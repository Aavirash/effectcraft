// Modified work: wraps the ported SAM 2 image path behind EffectCraft's `MaskModel` interface
// (Apache-2.0, see NOTICE).
//! SAM 2 as a drop-in [`MaskModel`]: a frame and a prompt in, a foreground probability per pixel
//! out. The frame's encoding is cached, so further strokes on the same frame cost one decode.

use std::sync::{Arc, Mutex, PoisonError};

use candle_core::{Device, Tensor};
use effectcraft_segment::{MaskModel, ModelInfo, SAM2_TINY};

use crate::model::Encoded;
use crate::prompt::{INPUT_SIZE, Prompt};
use crate::session::VideoModel;
use crate::{Error, Result};

/// Largest frame accepted (pixels per side and in total).
const MAX_SIDE: usize = 16_384;
const MAX_PIXELS: usize = 100_000_000;

/// Resize `w x h` RGB (0..1) to the model input and normalise it: `[1, 3, 1024, 1024]`.
pub fn preprocess(rgb: &[[f32; 3]], w: usize, h: usize) -> Result<Tensor> {
    let pixels = w.checked_mul(h).filter(|n| *n > 0 && *n == rgb.len() && w <= MAX_SIDE && h <= MAX_SIDE && *n <= MAX_PIXELS);
    let Some(pixels) = pixels else {
        return Err(Error::Input(format!("a {w} x {h} frame needs {} pixels (at most {MAX_PIXELS}), got {}", w.saturating_mul(h), rgb.len())));
    };
    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD: [f32; 3] = [0.229, 0.224, 0.225];
    let mut planar = vec![0f32; 3 * pixels];
    for (i, px) in rgb.iter().enumerate() {
        for c in 0..3 {
            let v = px.get(c).copied().unwrap_or(0.0);
            if let Some(o) = planar.get_mut(c * pixels + i) {
                *o = if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 };
            }
        }
    }
    let x = Tensor::from_vec(planar, (1, 3, h, w), &Device::Cpu)?.upsample_bilinear2d(INPUT_SIZE, INPUT_SIZE, false)?;
    let mean = Tensor::from_slice(&MEAN, (1, 3, 1, 1), &Device::Cpu)?;
    let std = Tensor::from_slice(&STD, (1, 3, 1, 1), &Device::Cpu)?;
    Ok(x.broadcast_sub(&mean)?.broadcast_div(&std)?)
}

fn fingerprint(rgb: &[[f32; 3]], w: usize, h: usize) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64 ^ ((w as u64) << 32 | h as u64);
    for px in rgb {
        for v in px {
            hash = (hash ^ u64::from(v.to_bits())).wrapping_mul(0x0100_0000_01b3);
        }
    }
    hash
}

/// SAM 2 for single frames.
pub struct Sam2Mask {
    model: Arc<VideoModel>,
    last: Mutex<Option<(u64, Arc<Encoded>)>>,
}

impl Sam2Mask {
    pub fn new(model: Arc<VideoModel>) -> Sam2Mask {
        Sam2Mask { model, last: Mutex::new(None) }
    }

    /// The video model, for trackers built on the same weights.
    pub fn video_model(&self) -> &Arc<VideoModel> {
        &self.model
    }

    fn encoded(&self, rgb: &[[f32; 3]], w: usize, h: usize) -> Result<Arc<Encoded>> {
        let key = fingerprint(rgb, w, h);
        if let Some((k, e)) = self.last.lock().unwrap_or_else(PoisonError::into_inner).as_ref()
            && *k == key
        {
            return Ok(e.clone());
        }
        let enc = Arc::new(self.model.image.encode(&preprocess(rgb, w, h)?)?);
        *self.last.lock().unwrap_or_else(PoisonError::into_inner) = Some((key, enc.clone()));
        Ok(enc)
    }

    fn probability(&self, rgb: &[[f32; 3]], w: usize, h: usize, prompt: &effectcraft_segment::Prompt) -> Result<Vec<f32>> {
        let enc = self.encoded(rgb, w, h)?;
        let (sx, sy) = (INPUT_SIZE as f32 / w as f32, INPUT_SIZE as f32 / h as f32);
        let p = Prompt {
            points: prompt.points.iter().map(|(c, fg)| ([c[0] * sx, c[1] * sy], *fg)).collect(),
            bbox: prompt.bbox.map(|b| [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]),
        };
        if p.points.is_empty() && p.bbox.is_none() {
            return Err(Error::Input("SAM 2 needs at least one point or a box".into()));
        }
        let multimask = p.points.len() + 2 * usize::from(p.bbox.is_some()) <= 1;
        let out = self.model.image.segment(&enc, &p, multimask)?;
        let ious = out.iou.flatten_all()?.to_vec1::<f32>()?;
        let best = ious.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |b, (i, v)| if *v > b.1 { (i, *v) } else { b }).0;
        let logits = out.masks.narrow(1, best, 1)?.upsample_bilinear2d(h, w, false)?;
        Ok(candle_nn::ops::sigmoid(&logits)?.flatten_all()?.to_vec1::<f32>()?)
    }
}

impl MaskModel for Sam2Mask {
    fn info(&self) -> &'static ModelInfo {
        &SAM2_TINY
    }

    fn segment(&self, rgb: &[[f32; 3]], w: usize, h: usize, prompt: &effectcraft_segment::Prompt) -> effectcraft_segment::Result<Vec<f32>> {
        self.probability(rgb, w, h, prompt).map_err(|e| e.to_string())
    }
}

/// Verify the installed file against the registry and build the model.
pub fn load(bytes: &[u8]) -> std::result::Result<Arc<Sam2Mask>, String> {
    let got = effectcraft_segment::sha256::hex(bytes);
    if got != SAM2_TINY.sha256 {
        return Err(format!("{}: the file does not match (SHA-256 {got}, expected {})", SAM2_TINY.name, SAM2_TINY.sha256));
    }
    let w = crate::weights::Weights::from_bytes(bytes).map_err(|e| e.to_string())?;
    let model = VideoModel::load(&w).map_err(|e| e.to_string())?;
    Ok(Arc::new(Sam2Mask::new(Arc::new(model))))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preprocess_rejects_bad_frames_and_cleans_pixels() {
        assert!(preprocess(&[], 0, 0).is_err());
        assert!(preprocess(&[[0.0; 3]; 3], 2, 2).is_err());
        assert!(preprocess(&[[0.5; 3]; 4], usize::MAX, 2).is_err());
        let t = preprocess(&[[f32::NAN, 2.0, -1.0]; 4], 2, 2).unwrap();
        assert_eq!(t.dims(), [1, 3, INPUT_SIZE, INPUT_SIZE]);
        let v = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(v.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn a_wrong_file_is_refused_before_parsing() {
        let e = load(b"not the weights").err().unwrap_or_default();
        assert!(e.contains("does not match"), "{e}");
    }
}
