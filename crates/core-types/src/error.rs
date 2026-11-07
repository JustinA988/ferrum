//! Errors returned by the library, and the JSON an HTTP handler sends.
//!
//! `FerrumError` is what loaders, kernels, the model, and the scheduler return.
//! `ApiErrorEnvelope` is the OpenAI error object: `error.message`, `error.type`,
//! `error.param`, and `error.code`. The server maps one onto the other.

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct ApiErrorEnvelope {
    pub error: ApiErrorObject,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct ApiErrorObject {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub param: Option<String>,
    pub code: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum FerrumError {
    #[error("{message}")]
    InvalidRequest {
        message: String,
        param: Option<String>,
    },
    #[error("{message}")]
    NotFound { message: String },
    #[error("{message}")]
    Unavailable { message: String },
    #[error("gguf: {0}")]
    Gguf(String),
    #[error("kernel: {0}")]
    Kernel(String),
    #[error("tokenizer: {0}")]
    Tokenizer(String),
    #[error("model: {0}")]
    Model(String),
    #[error("cache: {0}")]
    Cache(String),
    #[error("scheduler: {0}")]
    Scheduler(String),
    #[error("{0}")]
    Internal(String),
}

impl FerrumError {
    pub fn invalid_request(message: impl Into<String>, param: Option<&str>) -> Self {
        Self::InvalidRequest {
            message: message.into(),
            param: param.map(str::to_string),
        }
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self::NotFound {
            message: message.into(),
        }
    }

    pub fn unavailable(message: impl Into<String>) -> Self {
        Self::Unavailable {
            message: message.into(),
        }
    }

    pub fn gguf(message: impl Into<String>) -> Self {
        Self::Gguf(message.into())
    }

    pub fn kernel(message: impl Into<String>) -> Self {
        Self::Kernel(message.into())
    }

    pub fn tokenizer(message: impl Into<String>) -> Self {
        Self::Tokenizer(message.into())
    }

    pub fn model(message: impl Into<String>) -> Self {
        Self::Model(message.into())
    }

    pub fn cache(message: impl Into<String>) -> Self {
        Self::Cache(message.into())
    }

    pub fn scheduler(message: impl Into<String>) -> Self {
        Self::Scheduler(message.into())
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal(message.into())
    }

    /// HTTP status used by `/v1/*`. Health is not an error path.
    pub fn http_status(&self) -> u16 {
        match self {
            Self::InvalidRequest { .. } => 400,
            Self::NotFound { .. } => 404,
            Self::Unavailable { .. } => 503,
            Self::Gguf(_)
            | Self::Kernel(_)
            | Self::Tokenizer(_)
            | Self::Model(_)
            | Self::Cache(_)
            | Self::Scheduler(_)
            | Self::Internal(_) => 500,
        }
    }

    pub fn api_error(&self) -> ApiErrorEnvelope {
        let (error_type, param) = match self {
            Self::InvalidRequest { param, .. } => ("invalid_request_error", param.clone()),
            Self::NotFound { .. } => ("not_found_error", Some("model".to_string())),
            Self::Unavailable { .. } | Self::Internal(_) => ("server_error", None),
            Self::Gguf(_)
            | Self::Kernel(_)
            | Self::Tokenizer(_)
            | Self::Model(_)
            | Self::Cache(_)
            | Self::Scheduler(_) => ("server_error", None),
        };
        ApiErrorEnvelope {
            error: ApiErrorObject {
                message: self.to_string(),
                error_type: error_type.to_string(),
                param,
                code: None,
            },
        }
    }
}
