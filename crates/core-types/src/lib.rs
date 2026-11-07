//! Shared types for loading a GGUF model, running it, and serving completions.
//!
//! This is revision 1 of the ferrum interface: dtypes, errors, the kernel and
//! model traits, the KV cache, the scheduler, the tokenizer, and the OpenAI
//! request and response shapes.

#![forbid(unsafe_code)]

pub mod dtype;
pub mod engine;
pub mod error;
pub mod ids;
pub mod kernels;
pub mod kv;
pub mod metadata;
pub mod model;
pub mod schedule;
pub mod source;
pub mod tokenizer;

pub use dtype::{f16_to_f32, ggml_type_name, shape_elements, tensor_nbytes, DType};
pub use engine::{
    mock_completion, mock_completion_text, mock_prompt_tokens, mock_stream_chunks, ChunkChoice,
    Completion, CompletionChoice, CompletionChunk, CompletionEngine, CompletionRequest,
    CompletionRequestBody, CompletionUsage, FinishReason, Health, ModelCard, ModelList, PromptBody,
    PromptInput, StopBody, MOCK_COMPLETION_ID, MOCK_COMPLETION_TEXT, MOCK_MODEL_ID,
};
pub use error::{ApiErrorEnvelope, ApiErrorObject, FerrumError};
pub use ids::{RequestId, SlotId};
pub use kernels::{Backend, GemmArgs, Kernels, RopeArgs};
pub use kv::{KvPool, KvPoolConfig};
pub use metadata::{MetadataArray, MetadataType, MetadataValue, MetadataView};
pub use model::{
    argmax, llama_config, llama_tensor, CausalModel, LlamaConfig, OUTPUT, OUTPUT_NORM, TOKEN_EMBD,
};
pub use schedule::{SamplingParams, Scheduler, StepEvent, SubmitRequest};
pub use source::{format_gguf_dump, GgufModel, TensorDesc};
pub use tokenizer::{
    Tokenizer, TOKEN_BYTE, TOKEN_CONTROL, TOKEN_NORMAL, TOKEN_UNKNOWN, TOKEN_UNUSED,
    TOKEN_USER_DEFINED,
};

/// Bump only when a public signature, trait method, or wire format changes.
pub const INTERFACE_REVISION: u32 = 1;

pub const Q8_0_BLOCK: usize = dtype::Q8_0_BLOCK;
pub const Q8_0_BLOCK_BYTES: usize = dtype::Q8_0_BLOCK_BYTES;
