//! A feathered mask blurs its own edge, also where the path lies outside the layer: the feather
//! of a mask partly or wholly outside the layer reaches into it (After Effects), instead of the
//! coverage stopping at the layer edge.

use effectcraft_color::Label;
use effectcraft_keyframe::{ShapePath, Value};
use effectcraft_project::build::{self, Ids};
use effectcraft_project::{BitDepth, Comp, FeatherFalloff, GroupKind, ItemId, ItemKind, Layer, LayerSource, MaskMode, Node, Project, Solid};
use effectcraft_time::{FrameRate, Tick};

use crate::render_frame;

fn solid(p: &mut Project, comp: &Comp, color: [f32; 3], w: u32, h: u32) -> Layer {
    let sid = p.add_item("Solid", Label::Red, None, ItemKind::Solid(Solid { color, width: w, height: h, pixel_aspect: 1.0 }));
    build::layer(p, comp, "Solid", LayerSource::Solid { item: sid }, (w, h), None)
}

fn add_mask(p: &mut Project, l: &mut Layer, rect: ShapePath, feather: [f64; 2], linear: bool) {
    let mut next = p.next_id;
    let mut m = build::mask(&mut Ids(&mut next), "Mask 1", rect, MaskMode::Add, [255, 255, 0]);
    p.next_id = next;
    m.get_mut("feather").unwrap().value = Value::Vec2(feather);
    if linear && let GroupKind::Mask { feather_falloff, .. } = &mut m.kind {
        *feather_falloff = FeatherFalloff::Linear;
    }
    l.props.sub_mut("masks").unwrap().children.push(Node::Group(m));
}

/// The standard normal distribution function (Abramowitz and Stegun 7.1.26, error < 2e-7).
fn phi(x: f64) -> f64 {
    let z = x.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.327_591_1 * z);
    let poly = t * (0.254_829_592 + t * (-0.284_496_736 + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let erf = 1.0 - poly * (-z * z).exp();
    if x >= 0.0 { 0.5 * (1.0 + erf) } else { 0.5 * (1.0 - erf) }
}

/// Coverage at pixel centre `x` of the span `a..b` blurred by a Gaussian of `sigma` (one axis).
fn span(x: f64, a: f64, b: f64, sigma: f64) -> f64 {
    phi((b - x) / sigma) - phi((a - x) / sigma)
}

/// A white 200×100 solid (32 bpc) with one Add mask.
fn white_with_mask(rect: ShapePath, feather: [f64; 2], linear: bool, scale: f64) -> effectcraft_raster::Image {
    let mut p = Project::default();
    p.settings.bit_depth = BitDepth::Bpc32;
    let comp = Comp::new(200, 100, FrameRate::FPS_30, Tick::from_seconds_f64(1.0));
    let cid = p.add_item("Comp", Label::Sandstone, None, ItemKind::Comp(comp.clone().into()));
    let mut l = solid(&mut p, &comp, [1.0; 3], 200, 100);
    add_mask(&mut p, &mut l, rect, feather, linear);
    p.comp_mut(cid).unwrap().layers.push(l);
    render_frame(&p, cid, Tick::ZERO, scale)
}

#[test]
fn a_feathered_mask_outside_a_solid_reaches_into_it() {
    // The mask spans x 210..260, right of the layer (x 0..200); feather 40 = Gaussian sigma 20.
    let img = white_with_mask(ShapePath::rect([235.0, 50.0], 50.0, 200.0), [40.0, 40.0], false, 1.0);
    for x in [150i64, 180, 199] {
        let want = span(x as f64 + 0.5, 210.0, 260.0, 20.0) * span(50.5, -50.0, 150.0, 20.0);
        let got = img.get(x, 50)[3] as f64;
        assert!((got - want).abs() < 0.01, "x {x}: alpha {got}, the Gaussian gives {want}");
    }
    assert!(img.get(199, 50)[3] > 0.25, "the feather reaches the layer edge");
    assert!(img.get(100, 50)[3] < 1e-3);
    // Without feather the mask hides the layer.
    let hard = white_with_mask(ShapePath::rect([235.0, 50.0], 50.0, 200.0), [0.0, 0.0], false, 1.0);
    assert!(hard.data.iter().all(|p| p[3] == 0.0));
    // A linear falloff ramps over the feather width: 20 px either side of the edge at x 210.
    let lin = white_with_mask(ShapePath::rect([235.0, 50.0], 50.0, 200.0), [40.0, 40.0], true, 1.0);
    let a = lin.get(199, 50)[3];
    assert!((a - 0.24).abs() < 0.04, "linear: {a}");
    assert!(lin.get(185, 50)[3] < 0.01);
    // At half resolution the same layer pixel gets the same coverage.
    let half = white_with_mask(ShapePath::rect([235.0, 50.0], 50.0, 200.0), [40.0, 40.0], false, 0.5);
    assert!((half.get(99, 25)[3] - img.get(198, 50)[3]).abs() < 0.02, "{} {}", half.get(99, 25)[3], img.get(198, 50)[3]);
}

#[test]
fn a_feathered_mask_larger_than_the_layer_keeps_the_layer_edges() {
    // The mask covers the layer and 100 px around it: the layer edges are inside the mask and
    // stay opaque (they used to fade, as if the layer edge were the mask edge).
    let img = white_with_mask(ShapePath::rect([100.0, 50.0], 400.0, 300.0), [40.0, 40.0], false, 1.0);
    for (x, y) in [(0i64, 0i64), (0, 50), (199, 99), (100, 0)] {
        assert!(img.get(x, y)[3] > 0.99, "({x}, {y}): {}", img.get(x, y)[3]);
    }
    // A mask edge inside the layer is feathered as before: half coverage on the edge.
    let inner = white_with_mask(ShapePath::rect([100.0, 50.0], 100.0, 400.0), [40.0, 40.0], false, 1.0);
    assert!((inner.get(50, 50)[3] - 0.5).abs() < 0.03, "{}", inner.get(50, 50)[3]);
    assert!(inner.get(100, 50)[3] > 0.98);
}

#[test]
fn huge_and_broken_feathers_stay_bounded() {
    for f in [1.0e5, 1.0e9, f64::INFINITY, f64::NAN, -5.0] {
        let img = white_with_mask(ShapePath::rect([235.0, 50.0], 50.0, 200.0), [f, f], false, 1.0);
        assert!(img.data.iter().all(|p| p[3].is_finite() && (0.0..=1.0).contains(&p[3])), "feather {f}");
        let img = white_with_mask(ShapePath::rect([235.0, 50.0], 50.0, 200.0), [f, 10.0], true, 1.0);
        assert!(img.data.iter().all(|p| p[3].is_finite() && (0.0..=1.0).contains(&p[3])), "linear feather {f}");
    }
}

/// Case 398 of the .aep corpus, built in EffectCraft's own model: a 1920×1080 comp at 8 bpc, a
/// red solid, above it an adjustment layer of comp size with a Gradient Ramp at its defaults and
/// one Add mask entirely below the layer (1920×200 centred at [960, 1250]) with feather 1000.
fn case_398(with_mask: bool) -> (Project, ItemId) {
    let mut p = Project::default();
    assert_eq!(p.settings.bit_depth, BitDepth::Bpc8, "After Effects' default depth");
    let comp = Comp::new(1920, 1080, FrameRate::FPS_30, Tick::from_seconds_f64(5.0));
    let cid = p.add_item("C", Label::Sandstone, None, ItemKind::Comp(comp.clone().into()));
    let red = solid(&mut p, &comp, [1.0, 0.0, 0.0], 1920, 1080);
    let mut adj = solid(&mut p, &comp, [1.0; 3], 1920, 1080);
    adj.switches.adjustment = true;
    let spec = effectcraft_effects::find("ec.generate.gradientramp").unwrap();
    let mut next = p.next_id;
    let ramp = effectcraft_effects::instantiate(spec, &mut Ids(&mut next), spec.name, [1920.0, 1080.0]);
    p.next_id = next;
    adj.props.sub_mut("effects").unwrap().children.push(ramp.into());
    if with_mask {
        add_mask(&mut p, &mut adj, ShapePath::rect([960.0, 1250.0], 1920.0, 200.0), [1000.0, 1000.0], false);
    }
    let c = p.comp_mut(cid).unwrap();
    c.layers.push(adj);
    c.layers.push(red);
    (p, cid)
}

fn rgb8(img: &effectcraft_raster::Image, x: i64, y: i64) -> [i32; 3] {
    let px = img.get(x, y);
    [0, 1, 2].map(|c| (px[c].clamp(0.0, 1.0) * 255.0).round() as i32)
}

#[test]
fn case_398_a_feathered_mask_below_an_adjustment_layer_reaches_into_it() {
    let t = Tick::from_seconds_f64(1.0);
    let (p, cid) = case_398(true);
    let img = render_frame(&p, cid, t, 1.0);
    // The ramp alone (no mask): what the adjustment layer mixes in.
    let (q, qid) = case_398(false);
    let ramp = render_frame(&q, qid, t, 1.0);
    // After Effects 2026, 8 bpc, the render of 398 at 1 s.
    let ae = [
        ((960, 0), [255, 0, 0]),
        ((960, 300), [254, 0, 0]),
        ((960, 540), [250, 5, 5]),
        ((960, 800), [248, 20, 20]),
        ((960, 1000), [252, 40, 40]),
        ((960, 1079), [255, 47, 47]),
        ((100, 1079), [255, 29, 29]),
    ];
    let mut report = Vec::new();
    for ((x, y), want_ae) in ae {
        let got = rgb8(&img, x, y);
        // 1. The fix: the coverage is the mask's Gaussian (sigma = feather / 2 = 500), blurred from
        //    the mask edge outside the layer; the result is the red mixed toward the ramp by it.
        let c = span(x as f64 + 0.5, 0.0, 1920.0, 500.0) * span(y as f64 + 0.5, 1150.0, 1350.0, 500.0);
        let r = ramp.get(x, y);
        let want = [0, 1, 2].map(|k| {
            let red = if k == 0 { 1.0 } else { 0.0 };
            ((red + (r[k] as f64 - red) * c) * 255.0).round() as i32
        });
        for k in 0..3 {
            assert!((got[k] - want[k]).abs() <= 3, "({x}, {y}): {got:?}, the Gaussian coverage {c:.4} gives {want:?}");
        }
        // 2. After Effects: the same shape of falloff, the feather reaching ~500 px into the layer.
        //    Its kernel is narrower than sigma = feather / 2 (the AE values fit sigma ~ 350), so
        //    near the mask EffectCraft is lower and far from it higher: 12 levels apart at most
        //    (green at the bottom edge), allowed 14.
        for k in 0..3 {
            assert!((got[k] - want_ae[k]).abs() <= 14, "({x}, {y}): EffectCraft {got:?}, After Effects {want_ae:?}");
        }
        report.push(format!("({x},{y}) ec {got:?} ae {want_ae:?}"));
    }
    println!("398: {}", report.join("; "));
    // The mask covers nothing of the layer itself, but its feather does: the top stays (nearly) red.
    assert!(rgb8(&img, 960, 0)[0] >= 253 && rgb8(&img, 960, 0)[1] == 0);
    assert!(rgb8(&img, 960, 1079)[1] > 25, "the feather reaches the bottom edge");
}
