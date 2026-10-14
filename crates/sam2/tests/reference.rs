//! Stage-by-stage comparison with the PyTorch / transformers reference (`tools/sam2_reference.py`).
//!
//! Needs the `facebook/sam2.1-hiera-tiny` weights (default `<repo>/sam2-tiny`, or the folder in
//! `EFFECTCRAFT_SAM2_DIR`) and the fixture tensors (`<repo>/plan/sam2-ref/image.safetensors`, or
//! `EFFECTCRAFT_SAM2_REF`). Without them every test here skips, so CI stays green.
#![cfg(not(target_arch = "wasm32"))]

use std::collections::HashMap;
use std::path::PathBuf;

use candle_core::{Device, Tensor};
use effectcraft_sam2::hiera::{Hiera, HieraConfig};
use effectcraft_sam2::weights::Weights;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn weights_file() -> PathBuf {
    std::env::var_os("EFFECTCRAFT_SAM2_DIR").map_or_else(|| repo().join("sam2-tiny"), PathBuf::from).join("model.safetensors")
}

fn fixture_file() -> PathBuf {
    std::env::var_os("EFFECTCRAFT_SAM2_REF").map_or_else(|| repo().join("plan/sam2-ref/image.safetensors"), PathBuf::from)
}

/// The weights and the reference tensors, or `None` (the test then skips).
fn load() -> Option<(Weights, HashMap<String, Tensor>)> {
    let (wf, rf) = (weights_file(), fixture_file());
    if !wf.exists() || !rf.exists() {
        eprintln!("skipped: SAM 2 weights or reference tensors not found ({} / {})", wf.display(), rf.display());
        return None;
    }
    let w = Weights::open(&wf).unwrap();
    let r = candle_core::safetensors::load(&rf, &Device::Cpu).unwrap();
    Some((w, r))
}

fn max_abs(a: &Tensor, b: &Tensor) -> f32 {
    assert_eq!(a.dims(), b.dims(), "shape mismatch");
    (a - b).unwrap().abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap()
}

/// f32 activations differ from PyTorch by accumulation order only; the Hiera blocks stay well
/// inside this bound (measured max 2.6e-5, 2026-10-15).
const HIERA_TOL: f32 = 1e-4;

#[test]
fn hiera_matches_pytorch_block_by_block() {
    let Some((w, r)) = load() else { return };
    let hiera = Hiera::load(&w, &HieraConfig::tiny()).unwrap();
    let pixels = r.get("input.pixel_values").unwrap();
    let (emb, blocks) = hiera.forward_all(pixels).unwrap();
    let e = max_abs(&emb, r.get("vision_encoder.backbone.patch_embed.0").unwrap());
    eprintln!("patch_embed max abs error {e:e}");
    assert!(e < 1e-4, "patch embed error {e}");
    assert_eq!(blocks.len(), 12);
    for (i, b) in blocks.iter().enumerate() {
        let want = r.get(&format!("vision_encoder.backbone.blocks.{i}.0")).unwrap();
        let e = max_abs(b, want);
        eprintln!("block {i}: max abs error {e:e}");
        assert!(e < HIERA_TOL, "block {i} error {e} (tolerance {HIERA_TOL})");
    }
}

#[test]
fn hiera_is_deterministic_and_rejects_bad_sizes() {
    let Some((w, r)) = load() else { return };
    let hiera = Hiera::load(&w, &HieraConfig::tiny()).unwrap();
    let pixels = r.get("input.pixel_values").unwrap();
    let a = hiera.forward(pixels).unwrap();
    let b = hiera.forward(pixels).unwrap();
    assert_eq!(a.len(), 4);
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(max_abs(x, y), 0.0);
    }
    let odd = pixels.narrow(2, 0, 1022).unwrap();
    assert!(hiera.forward(&odd).is_err());
}
