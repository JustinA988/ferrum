use ferrum_core_types::{
    llama_config, llama_tensor, CausalModel, FerrumError, GgufModel, Kernels, KvPool, KvPoolConfig,
    LlamaConfig, MetadataView, RopeArgs, SlotId, OUTPUT, OUTPUT_NORM, TOKEN_EMBD,
};
use ferrum_kernels::scalar;
use ferrum_kvcache::PreallocatedKvPool;

use crate::weights::{dequant_weight, embed_token, linear, validate_weights};

struct MetaAdapter<'a, M: GgufModel>(&'a M);

impl<M: GgufModel> MetadataView for MetaAdapter<'_, M> {
    fn get(&self, key: &str) -> Option<&ferrum_core_types::MetadataValue> {
        self.0.metadata(key)
    }
}

/// Output projection name fixed at load (`output.weight` or tied `token_embd.weight`).
#[derive(Clone, Copy)]
enum OutputProj {
    Output,
    TiedEmbed,
}

impl OutputProj {
    fn name(self) -> &'static str {
        match self {
            Self::Output => OUTPUT,
            Self::TiedEmbed => TOKEN_EMBD,
        }
    }
}

/// RMS norm weights dequantized once at load.
struct NormWeights {
    attn: Vec<Vec<f32>>,
    ffn: Vec<Vec<f32>>,
    output: Vec<f32>,
}

struct WorkBuffers {
    norm: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    ffn: Vec<f32>,
    scores: Vec<f32>,
    head_acc: Vec<f32>,
    scratch: Vec<f32>,
}

impl WorkBuffers {
    fn new(config: &LlamaConfig, max_seq_len: usize) -> Self {
        let n_embd = config.n_embd as usize;
        let n_head = config.n_head as usize;
        let n_kv = config.n_head_kv as usize;
        let head_dim = config.head_dim as usize;
        let n_ff = config.n_ff as usize;
        Self {
            norm: vec![0.0; n_embd],
            q: vec![0.0; n_head * head_dim],
            k: vec![0.0; n_kv * head_dim],
            v: vec![0.0; n_kv * head_dim],
            attn: vec![0.0; n_head * head_dim],
            gate: vec![0.0; n_ff],
            up: vec![0.0; n_ff],
            ffn: vec![0.0; n_embd],
            scores: vec![0.0; max_seq_len],
            head_acc: vec![0.0; head_dim],
            scratch: vec![0.0; n_embd.max(n_ff)],
        }
    }
}

/// Teacher-forced Llama forward over a [`GgufModel`] weight source.
pub struct Llama<M> {
    model: M,
    config: LlamaConfig,
    kernels: Box<dyn Kernels>,
    kv: PreallocatedKvPool,
    output_proj: OutputProj,
    norms: NormWeights,
    work: WorkBuffers,
    hidden: Vec<f32>,
}

impl<M: GgufModel + Send + Sync> Llama<M> {
    /// Loads hyperparameters and validates weights. Uses [`scalar`] kernels for M4 acceptance;
    /// callers can rebuild with [`ferrum_kernels::select`] later.
    pub fn load(model: M, max_slots: usize, max_seq_len: usize) -> Result<Self, FerrumError> {
        let config = llama_config(&MetaAdapter(&model))?;
        let output_proj = match validate_weights(&model, &config)?.as_str() {
            OUTPUT => OutputProj::Output,
            TOKEN_EMBD => OutputProj::TiedEmbed,
            other => {
                return Err(FerrumError::model(format!(
                    "unexpected output weight name {other}"
                )))
            }
        };
        let kv = PreallocatedKvPool::new(KvPoolConfig {
            n_layers: config.n_layer as usize,
            n_kv_heads: config.n_head_kv as usize,
            head_dim: config.head_dim as usize,
            max_seq_len,
            max_slots,
        })?;
        let kernels = scalar();
        let norms = load_norm_weights(&model, kernels.as_ref(), &config)?;
        Ok(Self {
            work: WorkBuffers::new(&config, max_seq_len),
            hidden: vec![0.0; config.n_embd as usize],
            model,
            config,
            kernels,
            kv,
            output_proj,
            norms,
        })
    }

    pub fn model(&self) -> &M {
        &self.model
    }

    pub fn kernels(&self) -> &dyn Kernels {
        self.kernels.as_ref()
    }

    pub fn with_kernels(mut self, kernels: Box<dyn Kernels>) -> Self {
        self.kernels = kernels;
        self
    }

    fn embed(&mut self, token_id: u32) -> Result<(), FerrumError> {
        embed_token(&self.model, token_id, &self.config, &mut self.hidden)
    }

    fn forward_one_token(
        &mut self,
        slot: SlotId,
        pos: u32,
        logits_out: &mut [f32],
    ) -> Result<(), FerrumError> {
        let n_layer = self.config.n_layer as usize;
        let n_head = self.config.n_head as usize;
        let n_kv = self.config.n_head_kv as usize;
        let head_dim = self.config.head_dim as usize;
        let n_ff = self.config.n_ff as usize;
        let n_embd = self.config.n_embd as usize;
        let n_vocab = self.config.n_vocab as usize;
        if logits_out.len() != n_vocab {
            return Err(FerrumError::model("logits row length mismatch"));
        }
        self.kv.abandon_partial(slot)?;
        let pos_usize = pos as usize;
        let kv_group = n_head / n_kv;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let eps = self.config.rms_norm_eps;
        let rope_dim = self.config.rope_dim as usize;
        let rope_theta = self.config.rope_freq_base;
        let output_name = self.output_proj.name();

        let model = &self.model;
        let kernels = self.kernels.as_ref();
        let hidden = &mut self.hidden;
        let work = &mut self.work;
        let kv = &mut self.kv;
        let norms = &self.norms;

        for layer in 0..n_layer {
            kernels.rms_norm(hidden, &norms.attn[layer], eps, &mut work.norm)?;

            linear(
                model,
                kernels,
                &llama_tensor(layer as u32, "attn_q"),
                &work.norm,
                &mut work.q,
            )?;
            linear(
                model,
                kernels,
                &llama_tensor(layer as u32, "attn_k"),
                &work.norm,
                &mut work.k,
            )?;
            linear(
                model,
                kernels,
                &llama_tensor(layer as u32, "attn_v"),
                &work.norm,
                &mut work.v,
            )?;

            kernels.rope(RopeArgs {
                q: &mut work.q,
                k: &mut work.k,
                n_heads: n_head,
                n_kv_heads: n_kv,
                head_dim,
                rope_dim,
                pos,
                theta: rope_theta,
            })?;

            kv.append_layer(slot, layer, &work.k, &work.v)?;

            for h in 0..n_head {
                let q_off = h * head_dim;
                let q_head = &work.q[q_off..q_off + head_dim];
                let kv_h = h / kv_group;
                let k_off = kv_h * head_dim;
                let k_cur = &work.k[k_off..k_off + head_dim];
                let v_cur = &work.v[k_off..k_off + head_dim];

                let scores = &mut work.scores[..=pos_usize];
                for (p, score) in scores.iter_mut().enumerate().take(pos_usize) {
                    let k_p = &kv.k(slot, layer, p)?[k_off..k_off + head_dim];
                    *score = kernels.dot(q_head, k_p)? * scale;
                }
                scores[pos_usize] = kernels.dot(q_head, k_cur)? * scale;
                kernels.softmax_inplace(scores)?;

                work.head_acc.fill(0.0);
                for (p, &w) in scores.iter().enumerate().take(pos_usize) {
                    let v_p = &kv.v(slot, layer, p)?[k_off..k_off + head_dim];
                    for (d, acc) in work.head_acc.iter_mut().enumerate().take(head_dim) {
                        *acc += w * v_p[d];
                    }
                }
                for (d, acc) in work.head_acc.iter_mut().enumerate().take(head_dim) {
                    *acc += scores[pos_usize] * v_cur[d];
                }
                work.attn[q_off..q_off + head_dim].copy_from_slice(&work.head_acc);
            }

            linear(
                model,
                kernels,
                &llama_tensor(layer as u32, "attn_output"),
                &work.attn,
                &mut work.ffn,
            )?;
            kernels.add(hidden, &work.ffn, &mut work.scratch[..n_embd])?;
            hidden.copy_from_slice(&work.scratch[..n_embd]);

            kernels.rms_norm(hidden, &norms.ffn[layer], eps, &mut work.norm)?;

            linear(
                model,
                kernels,
                &llama_tensor(layer as u32, "ffn_gate"),
                &work.norm,
                &mut work.gate,
            )?;
            linear(
                model,
                kernels,
                &llama_tensor(layer as u32, "ffn_up"),
                &work.norm,
                &mut work.up,
            )?;
            kernels.silu(&work.gate, &mut work.scratch[..n_ff])?;
            kernels.mul(&work.scratch[..n_ff], &work.up, &mut work.gate)?;
            linear(
                model,
                kernels,
                &llama_tensor(layer as u32, "ffn_down"),
                &work.gate,
                &mut work.ffn,
            )?;
            kernels.add(hidden, &work.ffn, &mut work.scratch[..n_embd])?;
            hidden.copy_from_slice(&work.scratch[..n_embd]);
        }

        kernels.rms_norm(hidden, &norms.output, eps, &mut work.norm)?;
        linear(model, kernels, output_name, &work.norm, logits_out)?;

        kv.commit_token(slot)?;
        Ok(())
    }
}

fn load_norm_weights<M: GgufModel>(
    model: &M,
    kernels: &dyn Kernels,
    config: &LlamaConfig,
) -> Result<NormWeights, FerrumError> {
    let n_embd = config.n_embd as usize;
    let n_layer = config.n_layer as usize;
    let mut attn = Vec::with_capacity(n_layer);
    let mut ffn = Vec::with_capacity(n_layer);
    for layer in 0..n_layer {
        let mut w = vec![0.0; n_embd];
        dequant_weight(
            model,
            kernels,
            &llama_tensor(layer as u32, "attn_norm"),
            &mut w,
        )?;
        attn.push(w);
        let mut w = vec![0.0; n_embd];
        dequant_weight(
            model,
            kernels,
            &llama_tensor(layer as u32, "ffn_norm"),
            &mut w,
        )?;
        ffn.push(w);
    }
    let mut output = vec![0.0; n_embd];
    dequant_weight(model, kernels, OUTPUT_NORM, &mut output)?;
    Ok(NormWeights { attn, ffn, output })
}

impl<M: GgufModel + Send + Sync> CausalModel for Llama<M> {
    fn config(&self) -> &LlamaConfig {
        &self.config
    }

    fn alloc_slot(&mut self) -> Result<SlotId, FerrumError> {
        self.kv.alloc()
    }

    fn free_slot(&mut self, slot: SlotId) -> Result<(), FerrumError> {
        self.kv.free(slot)
    }

    fn seq_len(&self, slot: SlotId) -> Result<usize, FerrumError> {
        self.kv.seq_len(slot)
    }

    fn forward(
        &mut self,
        token_ids: &[u32],
        position0: u32,
        slot: SlotId,
        logits_out: &mut [f32],
    ) -> Result<(), FerrumError> {
        let n_vocab = self.config.n_vocab as usize;
        if logits_out.len() != token_ids.len().saturating_mul(n_vocab) {
            return Err(FerrumError::model("logits buffer length mismatch"));
        }
        let seq = self.kv.seq_len(slot)?;
        if position0 as usize != seq {
            return Err(FerrumError::model(format!(
                "position0 {position0} does not match sequence length {seq}"
            )));
        }

        for (idx, &token_id) in token_ids.iter().enumerate() {
            let pos = position0
                .checked_add(idx as u32)
                .ok_or_else(|| FerrumError::model("position overflow"))?;
            let base = idx * n_vocab;
            if let Err(err) = self.embed(token_id) {
                self.rewind_forward(slot, seq)?;
                return Err(err);
            }
            if let Err(err) =
                self.forward_one_token(slot, pos, &mut logits_out[base..base + n_vocab])
            {
                self.rewind_forward(slot, seq)?;
                return Err(err);
            }
        }
        Ok(())
    }
}

impl<M: GgufModel + Send + Sync> Llama<M> {
    fn rewind_forward(&mut self, slot: SlotId, committed: usize) -> Result<(), FerrumError> {
        while self.kv.seq_len(slot)? > committed {
            self.kv.uncommit_token(slot)?;
        }
        self.kv.abandon_partial(slot)
    }
}
