//! Identifiers that stay distinct across crates.
//!
//! `SlotId` names one sequence in the KV pool. `RequestId` names one completion
//! the scheduler accepted. Both are newtypes, so a slot cannot be passed where
//! a request is required.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct SlotId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct RequestId(pub u64);
