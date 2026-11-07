//! M6a acceptance for `ferrum serve --mock`.

#![forbid(unsafe_code)]

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStderr, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ferrum_core_types::{
    mock_completion, mock_stream_chunks, Health, ModelList, PromptInput, MOCK_COMPLETION_ID,
    MOCK_COMPLETION_TEXT, MOCK_MODEL_ID,
};

struct MockServer {
    child: Child,
    base: String,
    log: Arc<Mutex<String>>,
}

impl Drop for MockServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn ferrum_bin() -> PathBuf {
    let name = if cfg!(windows) {
        "ferrum.exe"
    } else {
        "ferrum"
    };
    let target_dir = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join("target")
        });
    target_dir.join("debug").join(name)
}

fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("ephemeral port");
    listener.local_addr().expect("local addr").port()
}

fn capture_stderr(stderr: ChildStderr) -> Arc<Mutex<String>> {
    let log = Arc::new(Mutex::new(String::new()));
    let slot = Arc::clone(&log);
    std::thread::spawn(move || {
        let mut text = String::new();
        let mut stderr = stderr;
        let _ = stderr.read_to_string(&mut text);
        if let Ok(mut guard) = slot.lock() {
            *guard = text;
        }
    });
    log
}

fn start_mock() -> MockServer {
    let bin = ferrum_bin();
    assert!(
        bin.is_file(),
        "missing {} (cargo build -p ferrum-cli)",
        bin.display()
    );
    let mut last_log = String::new();
    for _ in 0..8 {
        let port = free_port();
        let bind = format!("127.0.0.1:{port}");
        let mut child = Command::new(&bin)
            .args(["serve", "--mock", "--bind", bind.as_str()])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap_or_else(|err| panic!("spawn {}: {err}", bin.display()));
        let stderr = child.stderr.take().expect("stderr");
        let mut server = MockServer {
            child,
            base: bind,
            log: capture_stderr(stderr),
        };
        if wait_for_health(&mut server) {
            return server;
        }
        last_log = server
            .log
            .lock()
            .map(|guard| guard.clone())
            .unwrap_or_default();
        drop(server);
    }
    panic!("ferrum serve --mock did not answer /health\n{last_log}");
}

fn wait_for_health(server: &mut MockServer) -> bool {
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        if server.child.try_wait().ok().flatten().is_some() {
            return false;
        }
        if let Ok(response) = http(&server.base, "GET", "/health", None) {
            if response.status == 200 {
                return true;
            }
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    false
}

struct HttpResponse {
    status: u16,
    body: String,
    content_type: String,
}

fn http(base: &str, method: &str, path: &str, json: Option<&str>) -> std::io::Result<HttpResponse> {
    let mut stream = TcpStream::connect(base)?;
    stream.set_read_timeout(Some(Duration::from_secs(2)))?;
    stream.set_write_timeout(Some(Duration::from_secs(2)))?;
    let payload = json.unwrap_or("");
    let mut request = format!("{method} {path} HTTP/1.1\r\nHost: {base}\r\nConnection: close\r\n");
    if json.is_some() {
        request.push_str("Content-Type: application/json\r\n");
        request.push_str(&format!("Content-Length: {}\r\n", payload.len()));
    }
    request.push_str("\r\n");
    request.push_str(payload);
    stream.write_all(request.as_bytes())?;

    let mut raw = Vec::new();
    let header_end = loop {
        if let Some(end) = find_subsequence(&raw, b"\r\n\r\n") {
            break end + 4;
        }
        if read_some(&mut stream, &mut raw)? == 0 {
            break raw.len();
        }
    };
    let header = String::from_utf8_lossy(&raw[..header_end.min(raw.len())]).into_owned();
    let mut rest = raw.get(header_end..).unwrap_or_default().to_vec();
    let status = header
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|status| status.parse().ok())
        .unwrap_or(0);
    let content_type = header_value(&header, "content-type").unwrap_or_default();
    let body = if let Some(length) =
        header_value(&header, "content-length").and_then(|value| value.parse::<usize>().ok())
    {
        while rest.len() < length {
            if read_some(&mut stream, &mut rest)? == 0 {
                break;
            }
        }
        rest.truncate(length.min(rest.len()));
        rest
    } else if header_value(&header, "transfer-encoding")
        .is_some_and(|value| value.to_ascii_lowercase().contains("chunked"))
    {
        decode_chunks(&mut stream, &mut rest)?
    } else {
        loop {
            if read_some(&mut stream, &mut rest)? == 0 {
                break;
            }
        }
        rest
    };
    Ok(HttpResponse {
        status,
        body: String::from_utf8_lossy(&body).into_owned(),
        content_type,
    })
}

fn read_some(stream: &mut TcpStream, buf: &mut Vec<u8>) -> std::io::Result<usize> {
    let mut tmp = [0u8; 4096];
    match stream.read(&mut tmp) {
        Ok(0) => Ok(0),
        Ok(n) => {
            buf.extend_from_slice(&tmp[..n]);
            Ok(n)
        }
        Err(err)
            if err.kind() == std::io::ErrorKind::WouldBlock
                || err.kind() == std::io::ErrorKind::TimedOut =>
        {
            Ok(0)
        }
        Err(err) => Err(err),
    }
}

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn header_value(header: &str, name: &str) -> Option<String> {
    header.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        key.eq_ignore_ascii_case(name)
            .then(|| value.trim().to_string())
    })
}

fn decode_chunks(stream: &mut TcpStream, buf: &mut Vec<u8>) -> std::io::Result<Vec<u8>> {
    let mut body = Vec::new();
    loop {
        let line_end = loop {
            if let Some(index) = find_subsequence(buf, b"\r\n") {
                break index;
            }
            if read_some(stream, buf)? == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "chunk size",
                ));
            }
        };
        let size_line = String::from_utf8_lossy(&buf[..line_end]).into_owned();
        let size_hex = size_line.split(';').next().unwrap_or("").trim();
        let size = usize::from_str_radix(size_hex, 16).map_err(|err| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("chunk size: {err}"),
            )
        })?;
        buf.drain(..line_end + 2);
        if size == 0 {
            return Ok(body);
        }
        while buf.len() < size + 2 {
            if read_some(stream, buf)? == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "chunk data",
                ));
            }
        }
        body.extend_from_slice(&buf[..size]);
        buf.drain(..size + 2);
    }
}

fn parse_json(body: &str) -> serde_json::Value {
    serde_json::from_str(body.trim()).unwrap_or_else(|err| panic!("json ({err}): {body:?}"))
}

#[test]
fn mock_health_models_completion_sse_and_errors() {
    let server = start_mock();

    let health = http(&server.base, "GET", "/health", None).expect("health");
    assert_eq!(health.status, 200);
    assert_eq!(
        parse_json(&health.body),
        serde_json::to_value(Health::ok()).unwrap()
    );
    assert_eq!(health.body.trim(), r#"{"status":"ok"}"#);

    let models = http(&server.base, "GET", "/v1/models", None).expect("models");
    assert_eq!(models.status, 200);
    assert_eq!(
        parse_json(&models.body),
        serde_json::to_value(ModelList::mock()).unwrap()
    );
    assert!(models.body.contains(MOCK_MODEL_ID));

    let completion = http(
        &server.base,
        "POST",
        "/v1/completions",
        Some(r#"{"model":"ferrum-mock","prompt":"hi","user":"v1"}"#),
    )
    .expect("completion");
    assert_eq!(completion.status, 200);
    let expected = mock_completion(&PromptInput::Text("hi".into()), false);
    assert_eq!(
        parse_json(&completion.body),
        serde_json::to_value(&expected).unwrap()
    );
    assert_eq!(
        parse_json(&completion.body)["choices"][0]["text"],
        MOCK_COMPLETION_TEXT
    );
    assert_eq!(parse_json(&completion.body)["id"], MOCK_COMPLETION_ID);
    assert_eq!(parse_json(&completion.body)["created"], 0);

    let stream = http(
        &server.base,
        "POST",
        "/v1/completions",
        Some(r#"{"model":"ferrum-mock","prompt":"hi","stream":true}"#),
    )
    .expect("stream");
    assert_eq!(stream.status, 200);
    assert!(stream
        .content_type
        .to_ascii_lowercase()
        .contains("text/event-stream"));
    let mut sse = String::new();
    for chunk in mock_stream_chunks() {
        sse.push_str("data: ");
        sse.push_str(&serde_json::to_string(&chunk).unwrap());
        sse.push_str("\n\n");
    }
    sse.push_str("data: [DONE]\n\n");
    assert_eq!(stream.body, sse);
    assert!(stream.body.ends_with("data: [DONE]\n\n"));

    let missing = http(
        &server.base,
        "POST",
        "/v1/completions",
        Some(r#"{"model":"missing-model","prompt":"hi"}"#),
    )
    .expect("unknown model");
    assert_eq!(missing.status, 404);
    let err = parse_json(&missing.body);
    assert_eq!(err["error"]["type"], "not_found_error");
    assert_eq!(err["error"]["param"], "model");
    assert!(err["error"]["code"].is_null());
    assert!(err["error"]["message"]
        .as_str()
        .unwrap()
        .contains("missing-model"));

    let rejected = http(
        &server.base,
        "POST",
        "/v1/completions",
        Some(r#"{"model":"ferrum-mock","prompt":"hi","n":2}"#),
    )
    .expect("n=2");
    assert_eq!(rejected.status, 400);
    let err = parse_json(&rejected.body);
    assert_eq!(err["error"]["type"], "invalid_request_error");
    assert_eq!(err["error"]["param"], "n");
    assert!(err["error"]["code"].is_null());
}

/// Official `openai` Python client, non-stream and stream, against the mock.
///
/// Default `cargo test` skips this. Run it with `--ignored` where `python -c
/// "import openai"` works. The client is openai 3.x `completions.create`.
#[test]
#[ignore = "needs the official openai Python client; run with --ignored"]
fn official_openai_client_completion_and_stream() {
    let server = start_mock();
    let script = r#"
import os, sys
from openai import OpenAI
base = os.environ["FERRUM_BASE"]
client = OpenAI(base_url=f"http://{base}/v1", api_key="ferrum")
done = client.completions.create(model="ferrum-mock", prompt="hi", max_tokens=16)
text = done.choices[0].text
if text != "mock":
    sys.exit("non-stream text " + repr(text))
parts = []
for chunk in client.completions.create(
    model="ferrum-mock", prompt="hi", max_tokens=16, stream=True
):
    piece = chunk.choices[0].text or ""
    parts.append(piece)
joined = "".join(parts)
if joined != "mock":
    sys.exit("stream text " + repr(joined))
print("ok")
"#;
    let output = Command::new("python")
        .env("FERRUM_BASE", &server.base)
        .args(["-c", script])
        .output()
        .expect("spawn python");
    assert!(
        output.status.success(),
        "openai client failed\nstdout {}\nstderr {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
