use crate::error::FerrumError;
use serde::{Deserialize, Serialize};
use std::sync::mpsc;

pub const MOCK_MODEL_ID: &str = "ferrum-mock";
pub const MOCK_COMPLETION_ID: &str = "cmpl-mock";
pub const MOCK_COMPLETION_TEXT: &str = "mock";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FinishReason {
    Stop,
    Length,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromptInput {
    Text(String),
    Tokens(Vec<u32>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: PromptInput,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub stream: bool,
    pub stop: Vec<String>,
    pub echo: bool,
    pub seed: u64,
}

#[derive(Debug, Deserialize)]
pub struct CompletionRequestBody {
    pub model: String,
    pub prompt: PromptBody,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stop: Option<StopBody>,
    #[serde(default)]
    pub echo: bool,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub n: Option<u32>,
}

fn default_max_tokens() -> u32 {
    16
}

fn default_temperature() -> f32 {
    1.0
}

fn default_top_p() -> f32 {
    1.0
}

/// JSON prompt. A string is text. An array of numbers is token ids.
/// An array of strings is rejected by [`CompletionRequestBody::validate`].
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum PromptBody {
    Text(String),
    Tokens(Vec<u32>),
    Texts(Vec<String>),
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum StopBody {
    One(String),
    Many(Vec<String>),
}

impl CompletionRequestBody {
    pub fn validate(self) -> Result<CompletionRequest, FerrumError> {
        if self.n.is_some_and(|n| n != 1) {
            return Err(FerrumError::invalid_request(
                "only n = 1 is supported",
                Some("n"),
            ));
        }
        if self.max_tokens == 0 {
            return Err(FerrumError::invalid_request(
                "max_tokens must be at least 1",
                Some("max_tokens"),
            ));
        }
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(FerrumError::invalid_request(
                "temperature must be finite and >= 0",
                Some("temperature"),
            ));
        }
        if !self.top_p.is_finite() || self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err(FerrumError::invalid_request(
                "top_p must be in (0, 1]",
                Some("top_p"),
            ));
        }
        let prompt = match self.prompt {
            PromptBody::Text(text) => PromptInput::Text(text),
            PromptBody::Tokens(ids) => {
                if ids.is_empty() {
                    return Err(FerrumError::invalid_request(
                        "prompt token list is empty",
                        Some("prompt"),
                    ));
                }
                PromptInput::Tokens(ids)
            }
            PromptBody::Texts(_) => {
                return Err(FerrumError::invalid_request(
                    "prompt string arrays are not supported",
                    Some("prompt"),
                ));
            }
        };
        let stop = match self.stop {
            None => Vec::new(),
            Some(StopBody::One(value)) => vec![value],
            Some(StopBody::Many(values)) => values,
        };
        Ok(CompletionRequest {
            model: self.model,
            prompt,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            stream: self.stream,
            stop,
            echo: self.echo,
            seed: self.seed.unwrap_or(0),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: u32,
    pub logprobs: Option<()>,
    pub finish_reason: FinishReason,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CompletionUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Completion {
    pub id: String,
    #[serde(rename = "object")]
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: CompletionUsage,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ChunkChoice {
    pub text: String,
    pub index: u32,
    pub logprobs: Option<()>,
    pub finish_reason: Option<FinishReason>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CompletionChunk {
    pub id: String,
    #[serde(rename = "object")]
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ModelCard {
    pub id: String,
    #[serde(rename = "object")]
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ModelList {
    #[serde(rename = "object")]
    pub object: String,
    pub data: Vec<ModelCard>,
}

impl ModelList {
    pub fn mock() -> Self {
        Self {
            object: "list".to_string(),
            data: vec![ModelCard {
                id: MOCK_MODEL_ID.to_string(),
                object: "model".to_string(),
                created: 0,
                owned_by: "ferrum".to_string(),
            }],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Health {
    pub status: String,
}

impl Health {
    pub fn ok() -> Self {
        Self {
            status: "ok".to_string(),
        }
    }
}

pub fn mock_prompt_tokens(prompt: &PromptInput) -> u32 {
    match prompt {
        PromptInput::Text(text) => u32::from(!text.is_empty()),
        PromptInput::Tokens(ids) => u32::try_from(ids.len()).unwrap_or(u32::MAX),
    }
}

pub fn mock_completion_text(prompt: &PromptInput, echo: bool) -> String {
    if !echo {
        return MOCK_COMPLETION_TEXT.to_string();
    }
    let prefix = match prompt {
        PromptInput::Text(text) => text.clone(),
        PromptInput::Tokens(ids) => ids.iter().map(u32::to_string).collect::<Vec<_>>().join(" "),
    };
    if prefix.is_empty() {
        MOCK_COMPLETION_TEXT.to_string()
    } else {
        format!("{prefix} {MOCK_COMPLETION_TEXT}")
    }
}

pub fn mock_completion(prompt: &PromptInput, echo: bool) -> Completion {
    let prompt_tokens = mock_prompt_tokens(prompt);
    Completion {
        id: MOCK_COMPLETION_ID.to_string(),
        object: "text_completion".to_string(),
        created: 0,
        model: MOCK_MODEL_ID.to_string(),
        choices: vec![CompletionChoice {
            text: mock_completion_text(prompt, echo),
            index: 0,
            logprobs: None,
            finish_reason: FinishReason::Stop,
        }],
        usage: CompletionUsage {
            prompt_tokens,
            completion_tokens: 1,
            total_tokens: prompt_tokens.saturating_add(1),
        },
    }
}

pub fn mock_stream_chunks() -> [CompletionChunk; 2] {
    let content = CompletionChunk {
        id: MOCK_COMPLETION_ID.to_string(),
        object: "text_completion".to_string(),
        created: 0,
        model: MOCK_MODEL_ID.to_string(),
        choices: vec![ChunkChoice {
            text: MOCK_COMPLETION_TEXT.to_string(),
            index: 0,
            logprobs: None,
            finish_reason: None,
        }],
    };
    let done = CompletionChunk {
        id: MOCK_COMPLETION_ID.to_string(),
        object: "text_completion".to_string(),
        created: 0,
        model: MOCK_MODEL_ID.to_string(),
        choices: vec![ChunkChoice {
            text: String::new(),
            index: 0,
            logprobs: None,
            finish_reason: Some(FinishReason::Stop),
        }],
    };
    [content, done]
}

/// Blocking engine. The server calls it from a dedicated OS thread.
pub trait CompletionEngine: Send + Sync {
    fn models(&self) -> ModelList;

    fn complete(&self, request: CompletionRequest) -> Result<Completion, FerrumError>;

    fn complete_stream(
        &self,
        request: CompletionRequest,
        tx: mpsc::Sender<Result<CompletionChunk, FerrumError>>,
    ) -> Result<(), FerrumError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::FerrumError;

    #[test]
    fn completion_wire_shape() {
        let completion = mock_completion(&PromptInput::Text("hi".into()), false);
        let value = serde_json::to_value(&completion).unwrap();
        assert_eq!(value["object"], "text_completion");
        assert_eq!(value["id"], "cmpl-mock");
        assert_eq!(value["choices"][0]["text"], "mock");
        assert_eq!(value["choices"][0]["finish_reason"], "stop");
        assert!(value["choices"][0]["logprobs"].is_null());
        assert_eq!(value["usage"]["prompt_tokens"], 1);
        assert_eq!(value["usage"]["completion_tokens"], 1);
        assert_eq!(value["usage"]["total_tokens"], 2);
        assert_eq!(
            mock_completion_text(&PromptInput::Text("hi".into()), true),
            "hi mock"
        );
        assert_eq!(
            mock_completion_text(&PromptInput::Tokens(vec![1, 2]), true),
            "1 2 mock"
        );
    }

    #[test]
    fn request_defaults_and_rejections() {
        let body: CompletionRequestBody =
            serde_json::from_str(r#"{"model":"ferrum-mock","prompt":"hi"}"#).unwrap();
        let request = body.validate().unwrap();
        assert_eq!(request.max_tokens, 16);
        assert_eq!(request.temperature, 1.0);
        assert!(!request.stream);

        let body: CompletionRequestBody =
            serde_json::from_str(r#"{"model":"ferrum-mock","prompt":[7,8]}"#).unwrap();
        assert!(matches!(
            body.validate().unwrap().prompt,
            PromptInput::Tokens(_)
        ));

        let body: CompletionRequestBody =
            serde_json::from_str(r#"{"model":"ferrum-mock","prompt":"hi","n":2}"#).unwrap();
        let err = body.validate().unwrap_err();
        assert_eq!(err.http_status(), 400);
        assert_eq!(err.api_error().error.param.as_deref(), Some("n"));
        assert_eq!(err.api_error().error.error_type, "invalid_request_error");
        assert!(err.api_error().error.code.is_none());

        let missing = FerrumError::not_found("unknown model ferrum");
        assert_eq!(missing.http_status(), 404);
        assert_eq!(missing.api_error().error.error_type, "not_found_error");
    }

    #[test]
    fn model_list_and_stream_chunks() {
        let list = serde_json::to_value(ModelList::mock()).unwrap();
        assert_eq!(list["object"], "list");
        assert_eq!(list["data"][0]["id"], MOCK_MODEL_ID);
        assert_eq!(list["data"][0]["object"], "model");
        let chunks = mock_stream_chunks();
        assert_eq!(chunks[0].choices[0].text, "mock");
        assert!(chunks[0].choices[0].finish_reason.is_none());
        assert_eq!(chunks[1].choices[0].finish_reason, Some(FinishReason::Stop));
        assert_eq!(serde_json::to_value(Health::ok()).unwrap()["status"], "ok");
    }
}
