// Modified work: ported from Hugging Face `transformers` `models/sam2_video/modeling_sam2_video.py`
// (Sam2VideoModel and Sam2VideoInferenceSession: memory-conditioned single-frame inference,
// memory bank selection, object pointers, propagation; Apache-2.0, Copyright The HuggingFace
// Inc. team) to Rust on candle. See NOTICE.
//! The video tracker: per-object memory banks, prompts on any frame, propagation forwards and
//! backwards with streaming results and cancellation.

use std::collections::{BTreeMap, VecDeque};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use candle_core::{DType, Tensor};
use candle_nn::{Linear, ops::sigmoid};

use crate::decoder::FeedForward;
use crate::memattn::MemoryAttention;
use crate::memory::{Memory, MemoryEncoder};
use crate::model::ImageModel;
use crate::nn::{apply, linear};
use crate::prompt::{INPUT_SIZE, Prompt};
use crate::weights::Weights;
use crate::{Error, Result};

/// Mask logit used where the object is absent (the reference's `NO_OBJ_SCORE`).
const NO_OBJECT: f32 = -1024.0;
/// Frames of memory the attention looks back over.
const MEMORY_FRAMES: usize = 7;
/// Object pointers kept per query.
const MAX_POINTERS: usize = 16;
/// Largest clip the session accepts.
const MAX_FRAMES: usize = 1_000_000;
const GRID: usize = 64;

/// Frames of the clip, already resized to the model input and normalised: `[1, 3, 1024, 1024]`.
pub trait FrameSource: Send {
    fn num_frames(&self) -> usize;
    fn frame(&self, index: usize) -> Result<Tensor>;
}

/// Cooperative cancellation shared with the caller.
#[derive(Clone, Default)]
pub struct Cancel(Arc<AtomicBool>);

impl Cancel {
    pub fn new() -> Cancel {
        Cancel::default()
    }
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ObjectId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    Forward,
    Backward,
}

/// One object's mask on one frame.
#[derive(Clone, Debug, PartialEq)]
pub struct FrameMask {
    pub frame: usize,
    pub object: ObjectId,
    /// Mask logits at 256 x 256 (positive is inside the object); `-1024` everywhere when the
    /// object is absent.
    pub logits: Vec<f32>,
    /// The model's own estimate of the mask quality (0..1).
    pub iou: f32,
    /// Object-presence logit; at or below zero the object is not in the frame.
    pub object_score: f32,
    pub occluded: bool,
}

/// The weights of everything the video tracker needs.
pub struct VideoModel {
    pub image: ImageModel,
    memory_encoder: MemoryEncoder,
    memory_attention: MemoryAttention,
    temporal_pos: Vec<f32>,
    no_object_pointer: Tensor,
    pointer_proj: FeedForward,
    pointer_time_proj: Linear,
}

impl VideoModel {
    pub fn load(w: &Weights) -> Result<VideoModel> {
        Ok(VideoModel {
            image: ImageModel::load(w)?,
            memory_encoder: MemoryEncoder::load(w)?,
            memory_attention: MemoryAttention::load(w)?,
            temporal_pos: w.get("memory_temporal_positional_encoding", &[MEMORY_FRAMES, 1, 1, 64])?.flatten_all()?.to_vec1::<f32>()?,
            no_object_pointer: w.get("no_object_pointer", &[1, 256])?,
            pointer_proj: FeedForward::load(w, "object_pointer_proj", 256, 256, 256, 3, false)?,
            pointer_time_proj: linear(w, "temporal_positional_encoding_projection_layer", 256, 64)?,
        })
    }

    pub fn open(path: &Path) -> Result<VideoModel> {
        VideoModel::load(&Weights::open(path)?)
    }
}

/// What the tracker keeps for a frame it has processed.
struct FrameOut {
    memory: Memory,
    /// `[1, 256]`
    pointer: Tensor,
    /// The mask of a conditioning frame (returned again, unchanged, when propagation passes it).
    result: Option<FrameMask>,
}

#[derive(Default)]
struct ObjectState {
    cond: BTreeMap<usize, FrameOut>,
    non_cond: BTreeMap<usize, FrameOut>,
}

struct FrameFeatures {
    /// Lowest-resolution level `[1, 256, 64, 64]` and its position encoding.
    low: Tensor,
    pos: Tensor,
    high: (Tensor, Tensor),
}

pub struct VideoTracker {
    model: Arc<VideoModel>,
    frames: Box<dyn FrameSource>,
    objects: BTreeMap<ObjectId, ObjectState>,
    cache: VecDeque<(usize, Arc<FrameFeatures>)>,
}

fn sine_embedding_1d(positions: &[f32], dim: usize) -> Result<Tensor> {
    let half = dim / 2;
    let dim_t: Vec<f32> = (0..half).map(|i| 10000f32.powf(2.0 * (i / 2) as f32 / half as f32)).collect();
    let mut out = Vec::with_capacity(positions.len() * dim);
    for p in positions {
        out.extend(dim_t.iter().map(|d| (p / d).sin()));
        out.extend(dim_t.iter().map(|d| (p / d).cos()));
    }
    Ok(Tensor::from_vec(out, (positions.len(), dim), &candle_core::Device::Cpu)?)
}

fn scalar(t: &Tensor) -> Result<f32> {
    Ok(t.flatten_all()?.to_vec1::<f32>()?.first().copied().unwrap_or(0.0))
}

impl VideoTracker {
    pub fn new(model: Arc<VideoModel>, frames: Box<dyn FrameSource>) -> Result<VideoTracker> {
        let n = frames.num_frames();
        if n == 0 || n > MAX_FRAMES {
            return Err(Error::Input(format!("a clip needs between 1 and {MAX_FRAMES} frames, got {n}")));
        }
        Ok(VideoTracker { model, frames, objects: BTreeMap::new(), cache: VecDeque::new() })
    }

    pub fn num_frames(&self) -> usize {
        self.frames.num_frames()
    }

    /// Forget every object and cached frame (the clip stays).
    pub fn reset(&mut self) {
        self.objects.clear();
        self.cache.clear();
    }

    fn features(&mut self, frame: usize) -> Result<Arc<FrameFeatures>> {
        if let Some((_, f)) = self.cache.iter().find(|(i, _)| *i == frame) {
            return Ok(f.clone());
        }
        let pixels = self.frames.frame(frame)?;
        let enc = self.model.image.encode(&pixels)?;
        let (low, pos) = match (enc.features.features.get(2), enc.features.positions.get(2)) {
            (Some(l), Some(p)) => (l.clone(), p.clone()),
            _ => return Err(Error::Compute("vision encoder returned too few levels".into())),
        };
        let f = Arc::new(FrameFeatures { low, pos, high: enc.high });
        self.cache.push_back((frame, f.clone()));
        while self.cache.len() > 2 {
            self.cache.pop_front();
        }
        Ok(f)
    }

    /// The memory bank for `frame`: memory frames and object pointers with their position
    /// encodings, plus how many trailing tokens are pointers.
    fn memory_bank(&self, obj: &ObjectState, frame: usize, reverse: bool) -> Result<(Tensor, Tensor, usize)> {
        let (mut feats, mut poss) = (vec![], vec![]);
        let temporal = |off: usize| -> Result<Tensor> {
            // The reference indexes with `offset - 1`, so the conditioning frames (offset 0) use the last row.
            let row = if off == 0 { MEMORY_FRAMES - 1 } else { off - 1 };
            let v = self.model.temporal_pos.get(row * 64..(row + 1) * 64).ok_or_else(|| Error::Weights("temporal position table".into()))?;
            Ok(Tensor::from_slice(v, (1, 64), &candle_core::Device::Cpu)?)
        };
        for out in obj.cond.values() {
            feats.push(out.memory.features.clone());
            poss.push(out.memory.pos.broadcast_add(&temporal(0)?)?);
        }
        for off in (1..MEMORY_FRAMES).rev() {
            let prev = if reverse { frame.checked_add(off) } else { frame.checked_sub(off) };
            if let Some(out) = prev.and_then(|p| obj.non_cond.get(&p)) {
                feats.push(out.memory.features.clone());
                poss.push(out.memory.pos.broadcast_add(&temporal(off)?)?);
            }
        }
        // Object pointers: conditioning frames on the already-tracked side, then recent frames.
        let n_frames = self.frames.num_frames();
        let max_ptrs = n_frames.min(MAX_POINTERS);
        let (mut offsets, mut pointers): (Vec<f32>, Vec<Tensor>) = (vec![], vec![]);
        let sign = if reverse { -1.0f32 } else { 1.0 };
        for (t, out) in &obj.cond {
            if if reverse { *t >= frame } else { *t <= frame } {
                offsets.push((frame as f32 - *t as f32) * sign);
                pointers.push(out.pointer.clone());
            }
        }
        for off in 1..max_ptrs {
            let r = if reverse { frame.checked_add(off) } else { frame.checked_sub(off) };
            let Some(r) = r.filter(|r| *r < n_frames) else { break };
            if let Some(out) = obj.non_cond.get(&r) {
                offsets.push(off as f32);
                pointers.push(out.pointer.clone());
            }
        }
        let mut n_pointer_tokens = 0;
        if !pointers.is_empty() {
            let max_diff = (max_ptrs.saturating_sub(1)).max(1) as f32;
            let norm: Vec<f32> = offsets.iter().map(|o| o / max_diff).collect();
            let pe = apply(&self.model.pointer_time_proj, &sine_embedding_1d(&norm, 256)?)?;
            let ptrs = Tensor::cat(&pointers, 0)?; // [P, 256]
            let p = ptrs.dim(0)?;
            let tokens = ptrs.reshape((p * 4, 64))?;
            let pe = pe.unsqueeze(1)?.broadcast_as((p, 4, 64))?.reshape((p * 4, 64))?;
            n_pointer_tokens = p * 4;
            feats.push(tokens);
            poss.push(pe.contiguous()?);
        }
        if feats.is_empty() {
            return Err(Error::Input("the object has no conditioning frame; add a prompt first".into()));
        }
        Ok((Tensor::cat(&feats, 0)?, Tensor::cat(&poss, 0)?, n_pointer_tokens))
    }

    /// Run the model on one frame for one object and store the result.
    fn step(&mut self, id: ObjectId, frame: usize, prompt: Option<&Prompt>, reverse: bool, conditioning: bool) -> Result<FrameMask> {
        let feats = self.features(frame)?;
        let model = self.model.clone();
        let tokens = |t: &Tensor| -> Result<Tensor> { Ok(t.flatten_from(2)?.squeeze(0)?.t()?.contiguous()?) };
        let conditioned = if conditioning {
            feats.low.broadcast_add(model.image.no_memory())?
        } else {
            let obj = self.objects.get(&id).ok_or_else(|| Error::Input("unknown object".into()))?;
            let (mem, mem_pos, n_ptr) = self.memory_bank(obj, frame, reverse)?;
            let out = model.memory_attention.forward(&tokens(&feats.low)?, &tokens(&feats.pos)?, &mem, &mem_pos, n_ptr)?;
            out.t()?.contiguous()?.reshape((1, 256, GRID, GRID))?
        };
        let num_points = prompt.map_or(0, |p| p.points.len() + 2 * usize::from(p.bbox.is_some()));
        let multimask = num_points <= 1;
        let sparse = match prompt {
            Some(p) => model.image.prompt.sparse(p)?,
            None => model.image.prompt.sparse_none()?,
        };
        let dense = model.image.prompt.dense_empty()?;
        let out = model.image.decode_embedded(&conditioned, &feats.high, &sparse, &dense, multimask)?;

        let score = scalar(&out.object_score)?;
        let present = score > 0.0;
        let ious = out.iou.flatten_all()?.to_vec1::<f32>()?;
        let best = if multimask { ious.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |b, (i, v)| if *v > b.1 { (i, *v) } else { b }).0 } else { 0 };
        let low = out.masks.narrow(1, best, 1)?;
        let low = if present { low } else { Tensor::full(NO_OBJECT, low.dims(), low.device())? };
        let token = if multimask { out.sam_tokens.narrow(1, best, 1)? } else { out.sam_tokens.narrow(1, 0, 1)? }.squeeze(1)?;
        let pointer = if present { model.pointer_proj.forward(&token)? } else { model.no_object_pointer.clone() };

        let high = low.upsample_bilinear2d(INPUT_SIZE, INPUT_SIZE, false)?;
        let for_memory = if prompt.is_some() { high.gt(0f32)?.to_dtype(DType::F32)? } else { sigmoid(&high)? };
        let for_memory = ((for_memory * 20.0)? - 10.0)?;
        let memory = model.memory_encoder.memory(&feats.low, &for_memory, present)?;

        let iou = ious.get(best).copied().unwrap_or(0.0);
        let result = FrameMask { frame, object: id, logits: low.flatten_all()?.to_vec1::<f32>()?, iou, object_score: score, occluded: !present };
        let state = self.objects.entry(id).or_default();
        let entry = FrameOut { memory, pointer, result: conditioning.then(|| result.clone()) };
        if conditioning {
            state.non_cond.remove(&frame);
            state.cond.insert(frame, entry);
        } else {
            state.non_cond.insert(frame, entry);
            Self::prune(state, frame);
        }
        Ok(result)
    }

    /// Drop memories that no later frame in either direction can still look at.
    fn prune(state: &mut ObjectState, frame: usize) {
        let keep = |f: usize, cond: &BTreeMap<usize, FrameOut>| f.abs_diff(frame) <= MAX_POINTERS || cond.keys().any(|c| f.abs_diff(*c) <= MEMORY_FRAMES);
        let doomed: Vec<usize> = state.non_cond.keys().copied().filter(|f| !keep(*f, &state.cond)).collect();
        for f in doomed {
            state.non_cond.remove(&f);
        }
    }

    fn check_frame(&self, frame: usize) -> Result<()> {
        if frame >= self.frames.num_frames() {
            return Err(Error::Input(format!("frame {frame} is outside the clip of {} frames", self.frames.num_frames())));
        }
        Ok(())
    }

    /// Prompt an object on a frame (clicks and/or a box). The frame becomes one of the object's
    /// conditioning frames when it has not been tracked yet; the mask comes back at once.
    pub fn prompt(&mut self, frame: usize, object: ObjectId, prompt: &Prompt) -> Result<FrameMask> {
        self.check_frame(frame)?;
        prompt.validate()?;
        if prompt.points.is_empty() && prompt.bbox.is_none() {
            return Err(Error::Input("a prompt needs at least one point or a box".into()));
        }
        let tracked = self.objects.get(&object).is_some_and(|o| o.non_cond.contains_key(&frame));
        if tracked {
            // A correction on a tracked frame is conditioned on the memory like any tracked frame.
            let reverse = false;
            return self.step(object, frame, Some(prompt), reverse, false);
        }
        self.objects.entry(object).or_default();
        self.step(object, frame, Some(prompt), false, true)
    }

    /// Track every object from `start` through `max_frames` frames (the rest of the clip when
    /// `None`) in `direction`, calling `out` for each object on each frame as it is finished.
    pub fn propagate(&mut self, start: usize, direction: Direction, max_frames: Option<usize>, cancel: &Cancel, out: &mut dyn FnMut(FrameMask)) -> Result<()> {
        self.check_frame(start)?;
        let n = self.frames.num_frames();
        let reverse = direction == Direction::Backward;
        let span = max_frames.unwrap_or(n).min(n);
        let ids: Vec<ObjectId> = self.objects.iter().filter(|(_, o)| !o.cond.is_empty()).map(|(id, _)| *id).collect();
        if ids.is_empty() {
            return Err(Error::Input("no object has been prompted".into()));
        }
        for k in 0..=span {
            let frame = if reverse { start.checked_sub(k) } else { Some(start + k).filter(|f| *f < n) };
            let Some(frame) = frame else { break };
            for id in &ids {
                if cancel.is_cancelled() {
                    return Err(Error::Cancelled);
                }
                let cached = self.objects.get(id).is_some_and(|o| o.cond.contains_key(&frame));
                let stored = self.objects.get(id).and_then(|o| o.cond.get(&frame)).and_then(|o| o.result.clone());
                let mask = match stored {
                    Some(m) if cached => m,
                    _ => self.step(*id, frame, None, reverse, false)?,
                };
                out(mask);
            }
        }
        Ok(())
    }
}
