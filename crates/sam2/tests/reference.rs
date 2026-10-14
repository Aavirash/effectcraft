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

/// The neck adds 1x1 convolutions and nearest-neighbour sums to the Hiera output; the sine
/// position encodings are exact functions of the grid size.
const NECK_TOL: f32 = 1e-4;

#[test]
fn neck_and_vision_encoder_match_pytorch() {
    let Some((w, r)) = load() else { return };
    let enc = effectcraft_sam2::neck::VisionEncoder::load_tiny(&w).unwrap();
    let pixels = r.get("input.pixel_values").unwrap();
    let stages = enc.hiera.forward(pixels).unwrap();
    let out = enc.neck.forward(&stages).unwrap();
    assert_eq!((out.features.len(), out.positions.len()), (4, 4));
    for (i, f) in out.features.iter().enumerate() {
        let e = max_abs(f, r.get(&format!("vision_encoder.neck.{i}")).unwrap());
        eprintln!("neck feature {i}: max abs error {e:e}");
        assert!(e < NECK_TOL, "neck feature {i} error {e}");
    }
    for (i, p) in out.positions.iter().enumerate() {
        let e = max_abs(p, r.get(&format!("vision_encoder.neck.{}", i + 4)).unwrap());
        eprintln!("neck position {i}: max abs error {e:e}");
        assert!(e < NECK_TOL, "neck position {i} error {e}");
    }
    // The vision encoder keeps the three highest-resolution levels, highest first.
    let feats = enc.forward(pixels).unwrap();
    let dims: Vec<usize> = feats.features.iter().map(|f| f.dim(2).unwrap()).collect();
    assert_eq!(dims, vec![256, 128, 64]);
    let e = max_abs(feats.features.first().unwrap(), r.get("vision_encoder.neck.3").unwrap());
    assert!(e < NECK_TOL);
}

/// Click-prompted single image: prompt embeddings and decoder outputs against `Sam2Model`.
const DECODER_TOL: f32 = 1e-3;

#[test]
fn prompt_encoder_and_decoder_match_pytorch() {
    use effectcraft_sam2::model::ImageModel;
    use effectcraft_sam2::prompt::Prompt;
    let Some((w, r)) = load() else { return };
    let model = ImageModel::load(&w).unwrap();
    let pixels = r.get("input.pixel_values").unwrap();
    let prompt = Prompt { points: vec![([460.0, 512.0], true)], bbox: None };

    let (sparse, dense) = model.embed_prompt(&prompt).unwrap();
    let want_sparse = r.get("prompt_encoder.0").unwrap().squeeze(0).unwrap();
    let e = max_abs(&sparse, &want_sparse);
    eprintln!("sparse prompt: max abs error {e:e}");
    assert!(e < 1e-5, "sparse prompt error {e}");
    let e = max_abs(&dense, r.get("prompt_encoder.1").unwrap());
    assert!(e < 1e-6, "dense prompt error {e}");

    let enc = model.encode(pixels).unwrap();
    let out = model.segment(&enc, &prompt, true).unwrap();
    let want_masks = r.get("mask_decoder.0").unwrap().squeeze(0).unwrap();
    let e = max_abs(&out.masks, &want_masks);
    eprintln!("masks: max abs error {e:e} (logit range {:?})", want_masks.flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>());
    assert!(e < DECODER_TOL, "mask logits error {e}");
    let e = max_abs(&out.iou, &r.get("mask_decoder.1").unwrap().squeeze(0).unwrap());
    eprintln!("iou: max abs error {e:e}");
    assert!(e < 1e-4, "iou error {e}");
    let e = max_abs(&out.object_score, &r.get("mask_decoder.3").unwrap().squeeze(0).unwrap());
    eprintln!("object score: max abs error {e:e}");
    assert!(e < 1e-4, "object score error {e}");
    let e = max_abs(&out.sam_tokens, &r.get("mask_decoder.2").unwrap().squeeze(0).unwrap());
    assert!(e < 1e-4, "sam token error {e}");
}

fn video_fixture_file() -> PathBuf {
    std::env::var_os("EFFECTCRAFT_SAM2_VIDEO_REF").map_or_else(|| repo().join("plan/sam2-ref/video.safetensors"), PathBuf::from)
}

/// The weights and the video reference tensors (`tools/sam2_video_reference.py`), or `None`.
fn load_video() -> Option<(Weights, HashMap<String, Tensor>)> {
    let (wf, rf) = (weights_file(), video_fixture_file());
    if !wf.exists() || !rf.exists() {
        eprintln!("skipped: SAM 2 weights or video reference tensors not found ({} / {})", wf.display(), rf.display());
        return None;
    }
    Some((Weights::open(&wf).unwrap(), candle_core::safetensors::load(&rf, &Device::Cpu).unwrap()))
}

/// Measured max 3.5e-6 (memory features), 2.6e-5 (memory attention), 2026-10-15.
const MEMORY_TOL: f32 = 1e-4;

#[test]
fn memory_encoder_matches_pytorch() {
    let Some((w, r)) = load_video() else { return };
    let enc = effectcraft_sam2::memory::MemoryEncoder::load(&w).unwrap();
    let (feat, pos) = enc.encode(r.get("mem.enc.vision_features").unwrap(), r.get("mem.enc.masks").unwrap()).unwrap();
    let e = max_abs(&feat, r.get("mem.enc.out_features").unwrap());
    eprintln!("memory features: max abs error {e:e}");
    assert!(e < MEMORY_TOL, "memory feature error {e}");
    let e = max_abs(&pos, r.get("mem.enc.out_pos").unwrap());
    eprintln!("memory position encoding: max abs error {e:e}");
    assert!(e < 1e-5, "memory position error {e}");
}

#[test]
fn memory_attention_matches_pytorch() {
    let Some((w, r)) = load_video() else { return };
    let att = effectcraft_sam2::memattn::MemoryAttention::load(&w).unwrap();
    let get = |k: &str| r.get(k).unwrap().squeeze(1).unwrap();
    let pointers = r.get("mem.attn.num_pointers").unwrap().to_vec1::<f32>().unwrap()[0] as usize;
    let out = att
        .forward(
            &get("mem.attn.current_vision_features"),
            &get("mem.attn.current_vision_position_embeddings"),
            &get("mem.attn.memory"),
            &get("mem.attn.memory_posision_embeddings"),
            pointers,
        )
        .unwrap();
    let want = r.get("mem.attn.out").unwrap().squeeze(0).unwrap().squeeze(0).unwrap();
    let e = max_abs(&out, &want);
    eprintln!("memory attention: max abs error {e:e} (pointers {pointers})");
    assert!(e < MEMORY_TOL, "memory attention error {e}");
}

/// Frames of the video fixture, served one at a time.
struct Clip(Tensor);

impl effectcraft_sam2::session::FrameSource for Clip {
    fn num_frames(&self) -> usize {
        self.0.dim(0).unwrap()
    }
    fn frame(&self, i: usize) -> effectcraft_sam2::Result<Tensor> {
        Ok(self.0.narrow(0, i, 1)?.contiguous()?)
    }
}

fn iou(a: &[f32], b: &[f32]) -> f64 {
    let (mut i, mut u) = (0usize, 0usize);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (*x > 0.0, *y > 0.0);
        i += usize::from(x && y);
        u += usize::from(x || y);
    }
    if u == 0 { 1.0 } else { i as f64 / u as f64 }
}

/// Through the clip the memory bank feeds back on itself (bfloat16 memories, rotary attention), so
/// the bound on logits is looser than for a single stage; the masks themselves must agree.
/// Measured max 0.0038 over 9 frames forwards and backwards, 2026-10-15.
const VIDEO_LOGIT_TOL: f32 = 0.02;
const VIDEO_MIN_IOU: f64 = 0.995;

#[test]
fn video_tracking_matches_sam2_video_model() {
    use effectcraft_sam2::prompt::Prompt;
    use effectcraft_sam2::session::{Cancel, Direction, ObjectId, VideoModel, VideoTracker};
    let Some((w, r)) = load_video() else { return };
    let model = std::sync::Arc::new(VideoModel::load(&w).unwrap());
    let clip = r.get("clip.pixel_values").unwrap().clone();
    let mut tracker = VideoTracker::new(model, Box::new(Clip(clip))).unwrap();
    let obj = ObjectId(1);
    let first = tracker.prompt(2, obj, &Prompt { points: vec![([416.0, 512.0], true)], bbox: None }).unwrap();

    let check = |tag: &str, m: &effectcraft_sam2::session::FrameMask| {
        let want = r.get(&format!("video.{tag}.{}.pred_masks", m.frame)).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let want_score = r.get(&format!("video.{tag}.{}.score", m.frame)).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
        let err = m.logits.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let j = iou(&m.logits, &want);
        eprintln!("{tag} frame {}: logit error {err:.4}, IoU {j:.4}, score {:.3} (want {want_score:.3})", m.frame, m.object_score);
        assert!(j >= VIDEO_MIN_IOU, "{tag} frame {} IoU {j}", m.frame);
        assert!(err < VIDEO_LOGIT_TOL, "{tag} frame {} logit error {err}", m.frame);
        assert!((m.object_score - want_score).abs() < 0.05, "{tag} frame {} score {} vs {want_score}", m.frame, m.object_score);
        assert_eq!(m.occluded, want_score <= 0.0);
    };
    check("prompt", &first);
    let cancel = Cancel::new();
    let mut fwd = vec![];
    tracker.propagate(2, Direction::Forward, None, &cancel, &mut |m| fwd.push(m)).unwrap();
    assert_eq!(fwd.iter().map(|m| m.frame).collect::<Vec<_>>(), vec![2, 3, 4, 5, 6]);
    for m in &fwd {
        check("fwd", m);
    }
    let mut rev = vec![];
    tracker.propagate(2, Direction::Backward, None, &cancel, &mut |m| rev.push(m)).unwrap();
    assert_eq!(rev.iter().map(|m| m.frame).collect::<Vec<_>>(), vec![2, 1, 0]);
    for m in &rev {
        check("rev", m);
    }
}

#[test]
fn tracker_rejects_hostile_input_and_honours_cancel() {
    use effectcraft_sam2::Error;
    use effectcraft_sam2::prompt::Prompt;
    use effectcraft_sam2::session::{Cancel, Direction, ObjectId, VideoModel, VideoTracker};
    let Some((w, r)) = load_video() else { return };
    let model = std::sync::Arc::new(VideoModel::load(&w).unwrap());
    let clip = r.get("clip.pixel_values").unwrap().narrow(0, 0, 3).unwrap();
    let mut t = VideoTracker::new(model.clone(), Box::new(Clip(clip.clone()))).unwrap();
    let click = Prompt { points: vec![([416.0, 512.0], true)], bbox: None };
    let obj = ObjectId(7);
    // outside the clip, empty and non-finite prompts, nothing prompted yet
    assert!(matches!(t.prompt(3, obj, &click), Err(Error::Input(_))));
    assert!(matches!(t.prompt(0, obj, &Prompt::default()), Err(Error::Input(_))));
    assert!(matches!(t.prompt(0, obj, &Prompt { points: vec![([f32::NAN, 1.0], true)], bbox: None }), Err(Error::Input(_))));
    let mut sink = |_m| {};
    assert!(matches!(t.propagate(0, Direction::Forward, None, &Cancel::new(), &mut sink), Err(Error::Input(_))));
    assert!(matches!(t.propagate(9, Direction::Forward, None, &Cancel::new(), &mut sink), Err(Error::Input(_))));
    // an empty clip is refused; a cancelled run stops with Cancelled and keeps the tracker usable
    assert!(VideoTracker::new(model.clone(), Box::new(Clip(clip.narrow(0, 0, 0).unwrap()))).is_err());
    t.prompt(0, obj, &click).unwrap();
    let cancel = Cancel::new();
    cancel.cancel();
    assert!(matches!(t.propagate(0, Direction::Forward, None, &cancel, &mut sink), Err(Error::Cancelled)));
    let mut seen = vec![];
    t.propagate(0, Direction::Forward, Some(1), &Cancel::new(), &mut |m| seen.push(m.frame)).unwrap();
    assert_eq!(seen, vec![0, 1]);
    // a frame source that returns the wrong size is an error, not a crash
    struct Bad;
    impl effectcraft_sam2::session::FrameSource for Bad {
        fn num_frames(&self) -> usize {
            2
        }
        fn frame(&self, _i: usize) -> effectcraft_sam2::Result<Tensor> {
            Ok(Tensor::zeros((1, 3, 64, 64), candle_core::DType::F32, &Device::Cpu)?)
        }
    }
    let mut bad = VideoTracker::new(model, Box::new(Bad)).unwrap();
    assert!(matches!(bad.prompt(0, obj, &click), Err(Error::Input(_))));
}

#[test]
fn sam2_works_as_a_mask_model() {
    use effectcraft_segment::{MaskModel, Prompt};
    let Some((w, _)) = load() else { return };
    let (fw, fh) = (256usize, 192usize);
    let mut rgb = vec![[0f32; 3]; fw * fh];
    let inside = |x: usize, y: usize| ((x as f32 - 120.0).powi(2) + (y as f32 - 96.0).powi(2)).sqrt() < 50.0;
    for y in 0..fh {
        for x in 0..fw {
            let bg = 0.3 + 0.2 * (x as f32 / 9.0).sin() * (y as f32 / 7.0).cos();
            rgb[y * fw + x] = if inside(x, y) { [0.85 + 0.1 * (x as f32 / 5.0).sin(), 0.4, 0.15] } else { [bg, bg * 0.8 + 0.1, bg * 0.6 + 0.3] };
        }
    }
    let model = effectcraft_sam2::mask::Sam2Mask::new(std::sync::Arc::new(effectcraft_sam2::session::VideoModel::load(&w).unwrap()));
    let prompt = Prompt { points: vec![([120.0, 96.0], true)], bbox: None, mask: None };
    let p = model.segment(&rgb, fw, fh, &prompt).unwrap();
    assert_eq!(p.len(), fw * fh);
    assert!(p.iter().all(|v| (0.0..=1.0).contains(v)));
    let (mut i, mut u) = (0usize, 0usize);
    for y in 0..fh {
        for x in 0..fw {
            let (a, b) = (p[y * fw + x] > 0.5, inside(x, y));
            i += usize::from(a && b);
            u += usize::from(a || b);
        }
    }
    let j = i as f64 / u as f64;
    eprintln!("SAM 2 as a MaskModel: disc IoU {j:.3}");
    assert!(j > 0.9, "IoU {j}");
    // The same frame again is served from the cached encoding, and bad prompts are errors.
    let again = model.segment(&rgb, fw, fh, &Prompt { points: vec![([20.0, 20.0], false), ([120.0, 96.0], true)], bbox: None, mask: None }).unwrap();
    assert_eq!(again.len(), fw * fh);
    assert!(model.segment(&rgb, fw, fh, &Prompt::default()).is_err());
    assert!(model.segment(&rgb, fw + 1, fh, &prompt).is_err());
}

#[test]
fn sam2_tracks_a_clip_through_the_object_tracker_interface() {
    use effectcraft_segment::{ClipFrame, MaskModel, Prompt, TrackRequest};
    let Some((w, _)) = load() else { return };
    let (fw, fh, n) = (128usize, 96usize, 6usize);
    let centre = |t: usize| (40.0 + 8.0 * t as f32, 48.0f32);
    let inside = |t: usize, x: usize, y: usize| {
        let c = centre(t);
        ((x as f32 - c.0).powi(2) + (y as f32 - c.1).powi(2)).sqrt() < 26.0
    };
    let frames: Vec<std::sync::Arc<ClipFrame>> = (0..n)
        .map(|t| {
            let mut rgb = vec![[0f32; 3]; fw * fh];
            for y in 0..fh {
                for x in 0..fw {
                    let bg = 0.3 + 0.2 * (x as f32 / 9.0).sin() * (y as f32 / 7.0).cos();
                    rgb[y * fw + x] =
                        if inside(t, x, y) { [0.85 + 0.1 * ((x as f32 - centre(t).0) / 5.0).sin(), 0.4, 0.15] } else { [bg, bg * 0.8 + 0.1, bg * 0.6 + 0.3] };
                }
            }
            std::sync::Arc::new(ClipFrame { rgb, w: fw, h: fh })
        })
        .collect();
    let model = effectcraft_sam2::mask::Sam2Mask::new(std::sync::Arc::new(effectcraft_sam2::session::VideoModel::load(&w).unwrap()));
    assert!(model.tracker().is_some());
    let c = centre(1);
    let req = TrackRequest { first: 100, count: n, prompts: vec![(101, Prompt { points: vec![([c.0, c.1], true)], bbox: None, mask: None })] };
    let mut seen = std::collections::BTreeMap::new();
    let mut get = |f: i64| frames.get((f - 100) as usize).cloned();
    model
        .tracker()
        .unwrap()
        .track(&req, &mut get, &|| false, &mut |f, t| {
            assert!(seen.insert(f, t).is_none(), "frame {f} reported twice");
        })
        .unwrap();
    assert_eq!(seen.keys().copied().collect::<Vec<_>>(), (100..100 + n as i64).collect::<Vec<_>>());
    for (f, t) in &seen {
        assert_eq!((t.w, t.h, t.prob.len()), (fw, fh, fw * fh));
        let k = (f - 100) as usize;
        let (mut i, mut u) = (0usize, 0usize);
        for y in 0..fh {
            for x in 0..fw {
                let (a, b) = (t.prob[y * fw + x] > 0.5, inside(k, x, y));
                i += usize::from(a && b);
                u += usize::from(a || b);
            }
        }
        let j = i as f64 / u.max(1) as f64;
        eprintln!("frame {f}: IoU {j:.3}, confidence {:.2}, occluded {}", t.confidence, t.occluded);
        assert!(j > 0.85 && !t.occluded && t.confidence > 0.3, "frame {f}: IoU {j}");
    }
    // Cancelling before the work starts, and bad requests, are errors.
    assert!(model.tracker().unwrap().track(&req, &mut get, &|| true, &mut |_, _| {}).is_err());
    let none = TrackRequest { first: 100, count: n, prompts: vec![] };
    assert!(model.tracker().unwrap().track(&none, &mut get, &|| false, &mut |_, _| {}).is_err());
    let outside = TrackRequest { first: 100, count: n, prompts: vec![(500, Prompt { points: vec![([1.0, 1.0], true)], bbox: None, mask: None })] };
    assert!(model.tracker().unwrap().track(&outside, &mut get, &|| false, &mut |_, _| {}).is_err());
}
