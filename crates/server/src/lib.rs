//! HTTP API. Owned by A6.
//!
//! Routes: `GET /health`, `GET /v1/models`, `POST /v1/completions`.
//! The mock engine returns [`ferrum_core_types::mock_completion`] and
//! [`ferrum_core_types::mock_stream_chunks`]. Inference calls, when a real
//! engine exists, run on a dedicated OS thread and send results back on a channel.

#![forbid(unsafe_code)]
