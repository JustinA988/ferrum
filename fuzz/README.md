`fuzz/fuzz_targets/gguf_parse.rs` calls `ferrum_gguf::load` on arbitrary bytes and asserts that the call returns a `Result`. A panic is a failure.

This target has not been run. cargo-fuzz does not run on this Windows host. The 10-minute M1 fuzz gate is a Linux run. No 10-minute run is claimed.
