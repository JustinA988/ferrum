//! Planned M6a mock HTTP acceptance. cargo test does not compile this file.
//!
//! Spawn `ferrum serve --mock [--bind 127.0.0.1:8080]`. The official OpenAI
//! client has not been run.

#![forbid(unsafe_code)]

use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use ferrum_core_types::{
    mock_completion, mock_stream_chunks, Health, ModelList, PromptInput, MOCK_COMPLETION_ID,
    MOCK_COMPLETION_TEXT, MOCK_MODEL_ID,
};

struct MockServer {
    child: Child,
    base: String,
}

impl Drop for MockServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn start_mock() -> MockServer {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("port");
    let port = listener.local_addr().expect("addr").port();
    drop(listener);
    let bin = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/debug/ferrum.exe");
    let bind = format!("127.0.0.1:{port}");
    let child = Command::new(&bin)
        .args(["serve", "--mock", "--bind", bind.as_str()])
        .spawn()
        .expect("ferrum serve --mock");
    let base = bind;
    let server = MockServer { child, base };
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        if let Ok((status, _, _)) = http(&server.base, "GET", "/health", None) {
            if status == 200 {
                return server;
            }
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    panic!("ferrum serve --mock did not answer /health");
}

fn http(
    base: &str,
    method: &str,
    path: &str,
    json: Option<&str>,
) -> std::io::Result<(u16, String, String)> {
    let mut stream = TcpStream::connect(base)?;
    stream.set_read_timeout(Some(Duration::from_secs(2)))?;
    let body = json.unwrap_or("");
    let mut request = format!("{method} {path} HTTP/1.1\r\nHost: {base}\r\nConnection: close\r\n");
    if json.is_some() {
        request.push_str("Content-Type: application/json\r\n");
        request.push_str(&format!("Content-Length: {}\r\n", body.len()));
    }
    request.push_str("\r\n");
    request.push_str(body);
    stream.write_all(request.as_bytes())?;
    let mut raw = Vec::new();
    let mut buf = [0u8; 1024];
    loop {
        match stream.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                raw.extend_from_slice(&buf[..n]);
                if raw.windows(4).any(|w| w == b"\r\n\r\n")
                    && (raw.windows(14).any(|w| w == b"data: [DONE]\n\n")
                        || content_length_ready(&raw))
                {
                    break;
                }
            }
            Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => break,
            Err(err) if err.kind() == std::io::ErrorKind::TimedOut => break,
            Err(err) => return Err(err),
        }
    }
    Ok(split_response(&raw))
}

fn content_length_ready(raw: &[u8]) -> bool {
    let text = String::from_utf8_lossy(raw);
    let Some((head, body)) = text.split_once("\r\n\r\n") else {
        return false;
    };
    let length = head.lines().find_map(|line| {
        let (name, value) = line.split_once(':')?;
        if name.eq_ignore_ascii_case("content-length") {
            value.trim().parse::<usize>().ok()
        } else {
            None
        }
    });
    match length {
        Some(length) => body.len() >= length,
        None => false,
    }
}

fn split_response(raw: &[u8]) -> (u16, String, String) {
    let text = String::from_utf8_lossy(raw).into_owned();
    let (head, body) = text.split_once("\r\n\r\n").unwrap_or(("", ""));
    let status = head
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|status| status.parse().ok())
        .unwrap_or(0);
    let content_type = head
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-type")
                .then(|| value.trim().to_string())
        })
        .unwrap_or_default();
    (status, body.to_string(), content_type)
}

#[test]
fn mock_health_models_completion_sse_and_errors() {
    let server = start_mock();
    let (status, body, _) = http(&server.base, "GET", "/health", None).expect("health");
    assert_eq!(status, 200);
    assert_eq!(body, serde_json::to_string(&Health::ok()).unwrap());
    assert_eq!(body, r#"{"status":"ok"}"#);

    let (status, body, _) = http(&server.base, "GET", "/v1/models", None).expect("models");
    assert_eq!(status, 200);
    assert_eq!(body, serde_json::to_string(&ModelList::mock()).unwrap());
    assert!(body.contains(MOCK_MODEL_ID));

    let (status, body, _) = http(
        &server.base,
        "POST",
        "/v1/completions",
        Some(r#"{"model":"ferrum-mock","prompt":"hi","user":"v1"}"#),
    )
    .expect("completion");
    assert_eq!(status, 200);
    let expected = mock_completion(&PromptInput::Text("hi".into()), false);
    assert_eq!(body, serde_json::to_string(&expected).unwrap());
    let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(parsed["choices"][0]["text"], MOCK_COMPLETION_TEXT);
    assert_eq!(parsed["id"], MOCK_COMPLETION_ID);
    assert_eq!(parsed["created"], 0);

    let echoed = mock_completion(&PromptInput::Text("hi".into()), true);
    assert_eq!(echoed.choices[0].text, "hi mock");
    let tokens = mock_completion(&PromptInput::Tokens(vec![1, 2]), true);
    assert_eq!(tokens.choices[0].text, "1 2 mock");
    let empty = mock_completion(&PromptInput::Text(String::new()), true);
    assert_eq!(empty.choices[0].text, "mock");

    let (status, body, content_type) = http(
        &server.base,
        "POST",
        "/v1/completions",
        Some(r#"{"model":"ferrum-mock","prompt":"hi","stream":true}"#),
    )
    .expect("stream");
    assert_eq!(status, 200);
    assert!(content_type
        .to_ascii_lowercase()
        .contains("text/event-stream"));
    let chunks = mock_stream_chunks();
    assert_eq!(chunks[0].created, 0);
    assert_eq!(chunks[0].id, MOCK_COMPLETION_ID);
    let first = format!("data: {}\n\n", serde_json::to_string(&chunks[0]).unwrap());
    let second = format!("data: {}\n\n", serde_json::to_string(&chunks[1]).unwrap());
    assert!(body.contains(&first), "first SSE event");
    assert!(body.contains(&second), "second SSE event");
    assert!(body.contains("data: [DONE]\n\n"), "SSE [DONE]");

    let (status, body, _) = http(
        &server.base,
        "POST",
        "/v1/completions",
        Some(r#"{"model":"missing-model","prompt":"hi"}"#),
    )
    .expect("unknown model");
    assert_eq!(status, 404);
    let err: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(err["error"]["type"], "not_found_error");
    assert_eq!(err["error"]["param"], "model");
    assert!(err["error"]["code"].is_null());
    assert!(err["error"]["message"]
        .as_str()
        .unwrap()
        .contains("missing-model"));

    let (status, body, _) = http(
        &server.base,
        "POST",
        "/v1/completions",
        Some(r#"{"model":"ferrum-mock","prompt":"hi","n":2}"#),
    )
    .expect("n=2");
    assert_eq!(status, 400);
    let err: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(err["error"]["type"], "invalid_request_error");
    assert_eq!(err["error"]["param"], "n");
    assert!(err["error"]["code"].is_null());
}

/// The official OpenAI client non-stream and stream calls are not run.
#[test]
#[ignore = "official OpenAI client has not been run"]
fn official_openai_client_completion_and_stream() {
    panic!("official OpenAI client has not been run");
}
