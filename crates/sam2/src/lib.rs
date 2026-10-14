//! SAM 2.1 video object tracker (Hiera backbone) for the Roto Brush.
//!
//! A Rust port of the Hugging Face `transformers` SAM 2 implementation (Apache-2.0; see
//! `NOTICE`). Native only: the crate is empty on wasm32 so L0–L4 still build for the web.
//! Nothing here loads weights unless asked to; weights are never bundled.
//!
//! Modules are added stage by stage, each checked against reference tensors produced by
//! `tools/sam2_reference.py`.
#![cfg(not(target_arch = "wasm32"))]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

use std::fmt;

pub mod decoder;
pub mod hiera;
pub mod memattn;
pub mod memory;
pub mod model;
pub mod neck;
pub mod nn;
pub mod prompt;
pub mod session;
pub mod weights;

/// Everything that can go wrong in this crate. Never a panic.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// The weights file is missing a tensor, has the wrong shape, or contains non-finite values.
    Weights(String),
    /// The input (frame size, prompt, frame index) is unusable.
    Input(String),
    /// The work was cancelled by the caller.
    Cancelled,
    /// A time or memory budget was exceeded.
    Budget(String),
    /// A tensor operation failed (shape mismatch, out of memory, device error).
    Compute(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Weights(m) => write!(f, "SAM 2 weights: {m}"),
            Error::Input(m) => write!(f, "SAM 2 input: {m}"),
            Error::Cancelled => write!(f, "SAM 2: cancelled"),
            Error::Budget(m) => write!(f, "SAM 2 budget: {m}"),
            Error::Compute(m) => write!(f, "SAM 2 compute: {m}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<candle_core::Error> for Error {
    fn from(e: candle_core::Error) -> Error {
        Error::Compute(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// The released SAM 2.1 sizes. Only Tiny is wired up first; the others are configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Size {
    Tiny,
    Small,
    BasePlus,
    Large,
}

impl Size {
    /// The Hugging Face repository the official weights come from.
    pub fn repo(self) -> &'static str {
        match self {
            Size::Tiny => "facebook/sam2.1-hiera-tiny",
            Size::Small => "facebook/sam2.1-hiera-small",
            Size::BasePlus => "facebook/sam2.1-hiera-base-plus",
            Size::Large => "facebook/sam2.1-hiera-large",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_name_their_official_repositories() {
        assert_eq!(Size::Tiny.repo(), "facebook/sam2.1-hiera-tiny");
        assert!(Size::Large.repo().ends_with("large"));
    }

    #[test]
    fn errors_say_what_failed() {
        assert!(Error::Input("empty frame".into()).to_string().contains("empty frame"));
        assert_eq!(Error::Cancelled.to_string(), "SAM 2: cancelled");
    }
}
