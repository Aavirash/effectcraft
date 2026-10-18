//! Layer masks: path coverage with expansion, feather, opacity, invert, combined by mode.

use effectcraft_effects::Buf;
use effectcraft_geom::{Mat3, vec2};
use effectcraft_keyframe::Value;
use effectcraft_path::{FillRule, Join};
use effectcraft_project::{FeatherFalloff, GroupKind, Layer, MaskMode, MaskMotionBlur};
use effectcraft_raster::{Mask, box_blur, gaussian_blur};
use rayon::prelude::*;

use crate::eval::EvalCtx;

/// What mask motion blur may do in a render (from the renderer's motion blur gate).
#[derive(Clone, Copy, Debug, Default)]
pub struct MaskBlur {
    /// Motion blur is on in this render: the render options, the comp's switch and the Motion
    /// Blur switches of the precomp layers above (when switches affect nested comps).
    pub allowed: bool,
    /// The layer's own Motion Blur switch, as those precomp layers limit it (Same As Layer).
    pub layer: bool,
    /// Sub-frame samples (the comp's Samples Per Frame; fewer in Draft).
    pub samples: usize,
}

/// Coverage of one mask in buffer pixels (before mode/opacity/invert). With mask motion blur
/// (Layer ▸ Mask ▸ Motion Blur: On, or Same As Layer with the layer's switch on) an animated
/// path is averaged over the comp's shutter.
fn coverage(ctx: &EvalCtx, layer: &Layer, g: &effectcraft_project::PropGroup, buf: &Buf, blur: MaskBlur) -> Option<Mask> {
    let mb = blur.allowed
        && match g.kind {
            GroupKind::Mask { motion_blur, .. } => match motion_blur {
                MaskMotionBlur::On => true,
                MaskMotionBlur::Off => false,
                MaskMotionBlur::SameAsLayer => blur.layer,
            },
            _ => false,
        };
    let animated = g.get("path").is_some_and(|p| p.keys.len() > 1 || p.has_expression());
    if mb && animated {
        let fd = ctx.comp.frame_duration().seconds();
        let (angle, phase) = (ctx.comp.shutter_angle / 360.0, ctx.comp.shutter_phase / 360.0);
        let mut acc: Option<Mask> = None;
        let n = blur.samples.max(2);
        let k = 1.0 / n as f32;
        for i in 0..n {
            let f = phase + angle * i as f64 / (n - 1) as f64;
            let sub = ctx.at(ctx.time + effectcraft_time::Tick::from_seconds_f64(f * fd));
            let Some(c) = coverage_at(&sub, layer, g, buf) else { continue };
            match &mut acc {
                None => {
                    let mut c = c;
                    c.data.iter_mut().for_each(|v| *v *= k);
                    acc = Some(c);
                }
                Some(a) => a.data.iter_mut().zip(&c.data).for_each(|(a, c)| *a += c * k),
            }
        }
        return acc;
    }
    coverage_at(ctx, layer, g, buf)
}

/// The largest feather taken, in buffer pixels (larger and non-finite values are clamped).
const MAX_FEATHER_PX: f64 = 1.0e6;
/// Up to this Gaussian sigma (box radius for a linear falloff: twice it) a feather is blurred at
/// the buffer's resolution. A wider one is blurred on a grid coarser per axis by the factor that
/// leaves it about `COARSE_SIGMA` grid pixels (2x for the box radius): a wide blur is smooth, so
/// the coarse grid loses nothing visible, and the padding stays about 50 grid pixels.
const FULL_RES_SIGMA: f64 = 32.0;
const COARSE_SIGMA: f64 = 16.0;
/// The largest padded grid (pixels); a bigger one (only reachable with a huge buffer) falls back
/// to blurring within the buffer.
const MAX_GRID_PX: usize = 1 << 26;
/// Padding for a variable-width feather, which is rasterised at the buffer's resolution.
const MAX_VARIABLE_PAD: usize = 256;

/// One axis of the grid a feathered mask is rasterised and blurred on: `step` buffer pixels per
/// grid pixel, `pad` grid pixels on each side of the buffer, and the blur in grid pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Axis {
    step: usize,
    pad: usize,
    /// Gaussian sigma, or the box radius of a linear falloff.
    blur: f64,
}

impl Axis {
    /// `feather` in buffer pixels. The pad covers the blur's whole support, so the coverage of
    /// the path outside the buffer reaches into it as on an unbounded plane.
    fn new(feather: f64, linear: bool, full_res_only: bool) -> Axis {
        let f = if feather.is_finite() { feather.clamp(0.0, MAX_FEATHER_PX) } else { 0.0 };
        if f <= 0.0 {
            return Axis { step: 1, pad: 0, blur: 0.0 };
        }
        if linear {
            // One box pass: a straight ramp `feather` pixels wide across the edge.
            let r = (f / 2.0).round().max(1.0);
            let step = if full_res_only || r <= 2.0 * FULL_RES_SIGMA { 1 } else { (r / (2.0 * COARSE_SIGMA)).ceil() as usize };
            let rg = (r / step as f64).round().max(1.0);
            let pad = rg as usize;
            Axis { step, pad: if full_res_only { pad.min(MAX_VARIABLE_PAD) } else { pad }, blur: rg }
        } else {
            let sigma = f / 2.0;
            let step = if full_res_only || sigma <= FULL_RES_SIGMA { 1 } else { (sigma / COARSE_SIGMA).ceil() as usize };
            let sg = sigma / step as f64;
            // Three box passes of radii summing to at most 1.5 * (sqrt(4 sigma^2 + 1) + 1) <= 3 sigma + 3.
            let pad = (3.0 * sg).ceil() as usize + 3;
            Axis { step, pad: if full_res_only { pad.min(MAX_VARIABLE_PAD) } else { pad }, blur: sg }
        }
    }

    /// Grid pixels for `n` buffer pixels.
    fn len(&self, n: u32) -> Option<usize> {
        (n as usize).div_ceil(self.step).checked_add(self.pad.checked_mul(2)?)
    }
}

/// Resamples a grid coverage (`gw` wide) back to the `w x h` buffer: bilinear between the grid
/// pixel centres (an exact copy when both steps are 1).
fn crop(grid: &Mask, ax: Axis, ay: Axis, w: u32, h: u32) -> Mask {
    let (gw, gh) = (grid.width as usize, grid.height as usize);
    let mut out = Mask::new(w, h, 0.0);
    let at = |x: usize, y: usize| grid.data.get(y.saturating_mul(gw).saturating_add(x)).copied().unwrap_or(0.0);
    // Grid coordinate of a buffer pixel centre, as (index, fraction) on that axis.
    let map = |i: u32, a: Axis, n: usize| -> (usize, usize, f32) {
        let u = (i as f64 + 0.5) / a.step as f64 + a.pad as f64 - 0.5;
        let u = u.clamp(0.0, n.saturating_sub(1) as f64);
        let i0 = u.floor() as usize;
        (i0, (i0 + 1).min(n.saturating_sub(1)), (u - i0 as f64) as f32)
    };
    let xs: Vec<(usize, usize, f32)> = (0..w).map(|x| map(x, ax, gw)).collect();
    out.data.par_chunks_mut(w.max(1) as usize).enumerate().for_each(|(y, row)| {
        let (y0, y1, fy) = map(y as u32, ay, gh);
        for (o, &(x0, x1, fx)) in row.iter_mut().zip(&xs) {
            let top = at(x0, y0) + (at(x1, y0) - at(x0, y0)) * fx;
            let bottom = at(x0, y1) + (at(x1, y1) - at(x0, y1)) * fx;
            *o = top + (bottom - top) * fy;
        }
    });
    out
}

fn coverage_at(ctx: &EvalCtx, layer: &Layer, g: &effectcraft_project::PropGroup, buf: &Buf) -> Option<Mask> {
    let Value::Path(sp) = ctx.group_value(layer, g, "path")? else { return None };
    let path = effectcraft_path::to_kurbo(&sp);
    let (w, h) = (buf.img.width, buf.img.height);
    let m = Mat3::translate(vec2(buf.offset[0], buf.offset[1])) * Mat3::scale(vec2(buf.scale, buf.scale));
    let exp = ctx.f(layer, g, "expansion", 0.0);
    let linear = matches!(g.kind, GroupKind::Mask { feather_falloff: FeatherFalloff::Linear, .. });
    // Variable-width feather (Mask Feather tool points): expansion and the per-point feather
    // are applied together from the signed distance to the path.
    let variable = !sp.feather.is_empty();
    let feather = ctx.v2(layer, g, "feather", [0.0; 2]);
    // The feather blurs the mask's own edge, also where the path lies outside the layer (After
    // Effects: a feathered mask just below a layer still fades into it). So a feathered mask is
    // rasterised on a grid padded by the blur's reach, blurred there and cropped to the buffer.
    // A variable feather scales its point radii by the mean scale: it stays at full resolution.
    let (ax, ay) = (Axis::new(feather[0] * buf.scale, linear, variable), Axis::new(feather[1] * buf.scale, linear, variable));
    let blurred = ax.blur > 0.0 || ay.blur > 0.0;
    let grid = match (ax.len(w), ay.len(h)) {
        (Some(gw), Some(gh)) if blurred && gw.checked_mul(gh).is_some_and(|n| n <= MAX_GRID_PX) => {
            let (Ok(gw), Ok(gh)) = (u32::try_from(gw), u32::try_from(gh)) else { return None };
            Some((gw, gh, ax, ay))
        }
        _ => None,
    };
    // Without the grid: blurred within the buffer at its resolution (the behaviour before the padding).
    let unpadded = |f: f64| Axis { pad: 0, ..Axis::new(f, linear, true) };
    let (gw, gh, gx, gy) = grid.unwrap_or_else(|| (w, h, unpadded(feather[0] * buf.scale), unpadded(feather[1] * buf.scale)));
    // Buffer pixels -> grid pixels.
    let to_grid = Mat3::scale(vec2(1.0 / gx.step as f64, 1.0 / gy.step as f64)) * Mat3::translate(vec2((gx.pad * gx.step) as f64, (gy.pad * gy.step) as f64));
    let mg = to_grid * m;
    let mut cov = if variable {
        effectcraft_path::feather::variable_feather_coverage(&sp, &mg, gw, gh, exp, linear)
    } else if exp.abs() > 0.01 {
        // Expansion offsets the outline itself (round joins). Offset Paths' geometry stays right
        // when the expansion exceeds a curvature radius, where a fill + stroke blend leaves a
        // ring around an uncovered middle (growing) or islands (shrinking).
        let grown = effectcraft_path::offset::offset_path(&path, exp, Join::Round, 4.0);
        effectcraft_path::fill_coverage(std::slice::from_ref(&grown), &mg, gw, gh, FillRule::NonZero)
    } else {
        effectcraft_path::fill_coverage(std::slice::from_ref(&path), &mg, gw, gh, FillRule::NonZero)
    };
    if blurred {
        let img = cov.to_image();
        let b = if linear { box_blur(&img, gx.blur as usize, gy.blur as usize, 1, false) } else { gaussian_blur(&img, gx.blur, gy.blur, false) };
        cov = Mask::from_alpha(&b);
    }
    if grid.is_some() {
        cov = crop(&cov, gx, gy, w, h);
    }
    Some(cov)
}

/// The layer's enabled masks at the context time, flattened to polylines in layer space (for
/// effects that use masks as paths: Stroke, Scribble, Inner/Outer Key, Reshape…).
pub fn shapes(ctx: &EvalCtx, layer: &Layer) -> Vec<effectcraft_effects::MaskShape> {
    let Some(masks) = layer.masks() else { return Vec::new() };
    let mut out = Vec::new();
    for g in masks.groups().filter(|g| g.enabled) {
        let GroupKind::Mask { inverted, .. } = g.kind else { continue };
        let Some(Value::Path(sp)) = ctx.group_value(layer, g, "path") else { continue };
        let path = effectcraft_path::to_kurbo(&sp);
        let mut pts: Vec<[f64; 2]> = Vec::new();
        let mut closed = false;
        kurbo::flatten(path.iter(), 0.25, |el| match el {
            effectcraft_path::PathEl::MoveTo(p) | effectcraft_path::PathEl::LineTo(p) if pts.last() != Some(&[p.x, p.y]) => {
                pts.push([p.x, p.y]);
            }
            effectcraft_path::PathEl::ClosePath => closed = true,
            _ => {}
        });
        if closed && pts.len() > 1 && pts.first() == pts.last() {
            pts.pop();
        }
        out.push(effectcraft_effects::MaskShape { name: g.name.clone(), points: pts, closed: closed || sp.closed, inverted });
    }
    out
}

/// Apply the layer's masks to its buffer.
/// Returns whether any mask was applied.
pub fn apply(ctx: &EvalCtx, layer: &Layer, buf: &mut Buf, blur: MaskBlur) -> bool {
    let Some(masks) = layer.masks() else { return false };
    // An open mask (a path the Pen is still drawing) makes no transparency: the layer shows
    // whole until the path is closed (After Effects; effects still use it as a path).
    let closed = |g: &effectcraft_project::PropGroup| matches!(ctx.group_value(layer, g, "path"), Some(Value::Path(sp)) if sp.closed);
    let list: Vec<_> = masks.groups().filter(|g| g.enabled && matches!(g.kind, GroupKind::Mask { mode, .. } if mode != MaskMode::None) && closed(g)).collect();
    if list.is_empty() {
        return false;
    }
    // Masks may extend past the source: the buffer keeps its size (pixels outside the layer stay
    // empty), but a feathered mask is blurred on a padded grid, so the feather of an edge outside
    // the buffer still reaches into it (`coverage_at`).
    let n = (buf.img.width * buf.img.height) as usize;
    let first_sub = matches!(list[0].kind, GroupKind::Mask { mode: MaskMode::Subtract, .. });
    let mut acc = vec![if first_sub { 1.0f32 } else { 0.0 }; n];
    // The first mask has nothing to combine with. Intersect and Darken would multiply or
    // minimise against that empty coverage and hide the layer; they cover like Add instead.
    // Lighten and Difference already do. Filling the accumulator first would make Add show
    // the whole layer.
    let mut first = true;
    for g in list {
        let GroupKind::Mask { mode, inverted, .. } = g.kind else { continue };
        let Some(mut cov) = coverage(ctx, layer, g, buf, blur) else { continue };
        let mode = if first && matches!(mode, MaskMode::Intersect | MaskMode::Darken) { MaskMode::Add } else { mode };
        first = false;
        let op = (ctx.f(layer, g, "opacity", 100.0) / 100.0) as f32;
        cov.data.par_iter_mut().for_each(|c| {
            if inverted {
                *c = 1.0 - *c;
            }
            *c *= op;
        });
        acc.par_iter_mut().zip(cov.data.par_iter()).for_each(|(a, &c)| {
            *a = match mode {
                MaskMode::Add => *a + c - *a * c,
                MaskMode::Subtract => *a * (1.0 - c),
                MaskMode::Intersect => *a * c,
                MaskMode::Lighten => a.max(c),
                MaskMode::Darken => a.min(c),
                MaskMode::Difference => (*a - c).abs(),
                MaskMode::None => *a,
            };
        });
    }
    buf.img.mul_mask(&acc);
    true
}
