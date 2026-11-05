//! HTTP API. Owned by A6.
//!
//! Routes: `GET /health`, `GET /v1/models`, `POST /v1/completions`.
//! The mock calls [`ferrum_core_types::mock_completion`] and
//! [`ferrum_core_types::mock_stream_chunks`] inside the handler. A real
//! engine, when one exists, runs on a dedicated OS thread and sends results
//! back on a channel. This mock does not start that thread.

#![forbid(unsafe_code)]

use std::pin::Pin;
use std::task::{Context, Poll};

use actix_web::error::{InternalError, JsonPayloadError};
use actix_web::http::StatusCode;
use actix_web::{get, post, web, App, HttpRequest, HttpResponse, HttpServer};
use ferrum_core_types::{
    mock_completion, mock_stream_chunks, CompletionRequestBody, FerrumError, Health, ModelList,
    MOCK_MODEL_ID,
};
use futures_core::Stream;

/// Bind `addr` and serve the revision 1 mock until the process stops.
pub fn serve(addr: &str) -> std::io::Result<()> {
    let addr = addr.to_owned();
    actix_web::rt::System::new().block_on(async move {
        let server = HttpServer::new(|| App::new().configure(configure))
            .bind(addr.clone())
            .map_err(|err| std::io::Error::new(err.kind(), format!("bind {addr}: {err}")))?;
        eprintln!("ferrum: listening on {addr}");
        server.run().await
    })
}

fn configure(cfg: &mut web::ServiceConfig) {
    cfg.app_data(json_config())
        .service(health)
        .service(models)
        .service(completions);
}

fn json_config() -> web::JsonConfig {
    web::JsonConfig::default().error_handler(json_extract_error)
}

fn json_extract_error(err: JsonPayloadError, _req: &HttpRequest) -> actix_web::Error {
    let ferrum_err = match &err {
        JsonPayloadError::Deserialize(source) => {
            FerrumError::invalid_request(source.to_string(), None)
        }
        other => FerrumError::invalid_request(other.to_string(), None),
    };
    InternalError::from_response(err, error_response(&ferrum_err)).into()
}

fn error_response(err: &FerrumError) -> HttpResponse {
    let status =
        StatusCode::from_u16(err.http_status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    HttpResponse::build(status).json(err.api_error())
}

#[get("/health")]
async fn health() -> web::Json<Health> {
    web::Json(Health::ok())
}

#[get("/v1/models")]
async fn models() -> web::Json<ModelList> {
    web::Json(ModelList::mock())
}

#[post("/v1/completions")]
async fn completions(body: web::Json<CompletionRequestBody>) -> HttpResponse {
    let request = match body.into_inner().validate() {
        Ok(request) => request,
        Err(err) => return error_response(&err),
    };
    if request.model != MOCK_MODEL_ID {
        let err = FerrumError::not_found(format!("unknown model {}", request.model));
        return error_response(&err);
    }
    if request.stream {
        return stream_response();
    }
    HttpResponse::Ok().json(mock_completion(&request.prompt, request.echo))
}

fn stream_response() -> HttpResponse {
    let events = match sse_events() {
        Ok(events) => events,
        Err(err) => return error_response(&err),
    };
    HttpResponse::Ok()
        .content_type("text/event-stream")
        .streaming(SseBody {
            chunks: events.into_iter(),
        })
}

fn sse_events() -> Result<Vec<web::Bytes>, FerrumError> {
    let mut events = Vec::with_capacity(3);
    for chunk in mock_stream_chunks() {
        let json = serde_json::to_string(&chunk).map_err(|err| {
            FerrumError::internal(format!("could not encode stream chunk: {err}"))
        })?;
        events.push(web::Bytes::from(format!("data: {json}\n\n")));
    }
    events.push(web::Bytes::from_static(b"data: [DONE]\n\n"));
    Ok(events)
}

struct SseBody {
    chunks: std::vec::IntoIter<web::Bytes>,
}

impl Stream for SseBody {
    type Item = Result<web::Bytes, std::convert::Infallible>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Poll::Ready(self.get_mut().chunks.next().map(Ok))
    }
}

#[cfg(test)]
mod tests {
    use actix_web::http::header;
    use actix_web::http::StatusCode;
    use actix_web::{test, App};
    use ferrum_core_types::{
        mock_completion, mock_stream_chunks, Health, ModelList, PromptInput, MOCK_COMPLETION_ID,
        MOCK_MODEL_ID,
    };
    use serde_json::{json, Value};

    use super::configure;

    fn parse_json(bytes: &[u8]) -> Value {
        serde_json::from_slice(bytes).unwrap_or_else(|err| {
            panic!(
                "response is not json ({err}): {}",
                String::from_utf8_lossy(bytes)
            )
        })
    }

    fn assert_openai_error(body: &Value, error_type: &str, param: Option<&str>) {
        let error = body
            .get("error")
            .and_then(Value::as_object)
            .expect("error object");
        assert!(error.get("message").is_some_and(Value::is_string));
        assert_eq!(error.get("type").and_then(Value::as_str), Some(error_type));
        match param {
            Some(name) => assert_eq!(error.get("param").and_then(Value::as_str), Some(name)),
            None => assert!(error.get("param").is_some_and(Value::is_null)),
        }
        assert!(error.get("code").is_some_and(Value::is_null));
    }

    #[actix_web::test]
    async fn health_is_ok() {
        let app = test::init_service(App::new().configure(configure)).await;
        let req = test::TestRequest::get().uri("/health").to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = parse_json(&test::read_body(resp).await);
        assert_eq!(body, serde_json::to_value(Health::ok()).unwrap());
        assert_eq!(body, json!({"status": "ok"}));
    }

    #[actix_web::test]
    async fn models_lists_the_mock() {
        let app = test::init_service(App::new().configure(configure)).await;
        let req = test::TestRequest::get().uri("/v1/models").to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = parse_json(&test::read_body(resp).await);
        assert_eq!(body, serde_json::to_value(ModelList::mock()).unwrap());
        assert_eq!(body["data"][0]["id"], MOCK_MODEL_ID);
    }

    #[actix_web::test]
    async fn completion_returns_mock() {
        let app = test::init_service(App::new().configure(configure)).await;
        let req = test::TestRequest::post()
            .uri("/v1/completions")
            .set_json(json!({"model": MOCK_MODEL_ID, "prompt": "hi"}))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = parse_json(&test::read_body(resp).await);
        let expected =
            serde_json::to_value(mock_completion(&PromptInput::Text("hi".into()), false)).unwrap();
        assert_eq!(body, expected);
        assert_eq!(body["choices"][0]["text"], "mock");
        assert_eq!(body["choices"][0]["finish_reason"], "stop");
        assert_eq!(body["id"], MOCK_COMPLETION_ID);
        assert_eq!(body["id"], "cmpl-mock");
    }

    #[actix_web::test]
    async fn completion_echo_uses_mock_helper() {
        let app = test::init_service(App::new().configure(configure)).await;
        let req = test::TestRequest::post()
            .uri("/v1/completions")
            .set_json(json!({"model": MOCK_MODEL_ID, "prompt": "hi", "echo": true}))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = parse_json(&test::read_body(resp).await);
        let expected =
            serde_json::to_value(mock_completion(&PromptInput::Text("hi".into()), true)).unwrap();
        assert_eq!(body, expected);
        assert_eq!(body["choices"][0]["text"], "hi mock");
    }

    #[actix_web::test]
    async fn completion_stream_writes_sse() {
        let app = test::init_service(App::new().configure(configure)).await;
        let req = test::TestRequest::post()
            .uri("/v1/completions")
            .set_json(json!({"model": MOCK_MODEL_ID, "prompt": "hi", "stream": true}))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let content_type = resp
            .headers()
            .get(header::CONTENT_TYPE)
            .expect("content-type")
            .to_str()
            .expect("content-type text");
        assert!(content_type.contains("text/event-stream"), "{content_type}");
        let bytes = test::read_body(resp).await;
        let text = std::str::from_utf8(&bytes).expect("utf-8 body");
        let mut expected = String::new();
        for chunk in mock_stream_chunks() {
            let json = serde_json::to_string(&chunk).expect("chunk json");
            expected.push_str("data: ");
            expected.push_str(&json);
            expected.push_str("\n\n");
        }
        expected.push_str("data: [DONE]\n\n");
        assert_eq!(text, expected);
        assert!(text.contains("data: "));
        assert!(text.contains("[DONE]"));
    }

    #[actix_web::test]
    async fn unknown_model_is_not_found() {
        let app = test::init_service(App::new().configure(configure)).await;
        let req = test::TestRequest::post()
            .uri("/v1/completions")
            .set_json(json!({"model": "nope", "prompt": "hi"}))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let body = parse_json(&test::read_body(resp).await);
        assert_openai_error(&body, "not_found_error", Some("model"));
        let message = body["error"]["message"].as_str().expect("message");
        assert!(message.contains("nope"), "{message}");
    }

    #[actix_web::test]
    async fn n_other_than_one_is_rejected() {
        let app = test::init_service(App::new().configure(configure)).await;
        let req = test::TestRequest::post()
            .uri("/v1/completions")
            .set_json(json!({"model": MOCK_MODEL_ID, "prompt": "hi", "n": 2}))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = parse_json(&test::read_body(resp).await);
        assert_openai_error(&body, "invalid_request_error", Some("n"));
        assert_eq!(body["error"]["message"], "only n = 1 is supported");
    }

    #[actix_web::test]
    async fn unknown_json_field_is_ignored() {
        let app = test::init_service(App::new().configure(configure)).await;
        let req = test::TestRequest::post()
            .uri("/v1/completions")
            .set_json(json!({
                "model": MOCK_MODEL_ID,
                "prompt": "hi",
                "user": "openai",
                "suffix": "unused"
            }))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = parse_json(&test::read_body(resp).await);
        assert_eq!(body["choices"][0]["text"], "mock");
    }

    #[actix_web::test]
    async fn json_extract_failure_uses_openai_envelope() {
        let app = test::init_service(App::new().configure(configure)).await;
        let req = test::TestRequest::post()
            .uri("/v1/completions")
            .insert_header((header::CONTENT_TYPE, "application/json"))
            .set_payload("{")
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = parse_json(&test::read_body(resp).await);
        assert_openai_error(&body, "invalid_request_error", None);
    }
}
