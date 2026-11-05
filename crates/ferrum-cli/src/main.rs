#![forbid(unsafe_code)]

fn main() {
    eprintln!(
        "ferrum {version}: interface revision {revision} (server not built)",
        version = env!("CARGO_PKG_VERSION"),
        revision = ferrum_core_types::INTERFACE_REVISION
    );
    std::process::exit(2);
}
