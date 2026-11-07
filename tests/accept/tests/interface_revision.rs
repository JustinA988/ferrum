//! Confirms the published interface revision is 1.

#![forbid(unsafe_code)]

#[test]
fn interface_revision_is_one() {
    assert_eq!(ferrum_core_types::INTERFACE_REVISION, 1);
}
