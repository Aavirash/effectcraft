//! Layer ▸ Remove Silence…: find the silences in a footage layer's own audio and either create
//! markers for them, split the layer at them, or cut them out and close the gaps (Tighten),
//! optionally into a precomposition. The analysis reads the source audio in one-second blocks
//! and runs as a background job (Window ▸ Progress), like Scene Edit Detection; the edit itself
//! is one undo step.

use std::sync::Arc;

use effectcraft_keyframe::{Keyframe, Value as KV};
use effectcraft_project::{Comp, ItemId, ItemKind, Layer, LayerId, LayerSource, Marker, Project};
use effectcraft_render::FootageSource;
use effectcraft_time::Tick;
use serde_json::{Value, json};

use super::{CommandSpec, b_p, bad, f_p, layers_p, str_p};
use crate::jobs::Apply;
use crate::{EngineError, Result, Session, cmd};

/// The analysis sample rate. Speech thresholds are level-based, so this only needs to be a
/// normal audio rate; 48 kHz matches the app's mix rate.
const RATE: u32 = 48_000;
/// RMS window (seconds): 10 ms is short enough to find a word gap and long enough to be stable.
const WINDOW: f64 = 0.010;
/// Samples read per block: one second (100 windows).
const BLOCK: usize = RATE as usize;
/// Fade at each new cut (seconds), so a tight cut does not click.
const FADE: f64 = 0.005;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Markers,
    Split,
    Tighten,
}

#[derive(Clone, Copy, Debug)]
struct Opts {
    /// Silence threshold in dBFS.
    threshold: f32,
    /// A quieter run shorter than this is treated as speech (no cut).
    min_silence: f64,
    /// A louder run shorter than this is treated as silence (cut).
    min_speech: f64,
    /// Kept before and after each speech run.
    padding: f64,
}

impl Opts {
    fn from(p: &Value) -> Result<Opts> {
        let read = |k: &str, d: f64| -> Result<f64> {
            let v = f_p(p, k).unwrap_or(d);
            if !v.is_finite() {
                return Err(bad("layer.removeSilence", format!("`{k}` must be a number")));
            }
            Ok(v)
        };
        Ok(Opts {
            threshold: read("threshold", -35.0)?.clamp(-96.0, 0.0) as f32,
            min_silence: read("minSilence", 0.35)?.max(0.0),
            min_speech: read("minSpeech", 0.2)?.max(0.0),
            padding: read("padding", 0.1)?.max(0.0),
        })
    }
}

/// Whether `l` is a footage layer with audio the app can decode.
fn has_audio(project: &Project, l: &Layer) -> bool {
    l.switches.audio
        && matches!(&l.source, LayerSource::Footage { item } if matches!(
            project.item(*item).map(|i| &i.kind),
            Some(ItemKind::Footage(f)) if f.has_audio && !f.missing
        ))
}

fn enabled(s: &Session) -> std::result::Result<(), String> {
    let comp = s.active_comp().ok_or("no composition is open")?;
    if s.state.selected_layers.iter().filter_map(|id| comp.layer(*id)).any(|l| has_audio(&s.project, l)) {
        Ok(())
    } else {
        Err("select a footage or audio layer with sound".into())
    }
}

/// What a background job needs to read a layer's audio: the project and the decoder, both
/// snapshots taken when the job starts (the same contract as Scene Edit Detection).
#[derive(Clone)]
struct Probe {
    project: Arc<Project>,
    footage: Arc<dyn FootageSource>,
}

impl Probe {
    fn samples(&self, item: ItemId, footage: &effectcraft_project::Footage, t: Tick, frames: usize) -> Vec<f32> {
        self.footage.audio(item, footage, t, frames, RATE).unwrap_or_else(|| vec![0.0; frames * 2])
    }
}

/// The kept spans of comp time: the loud runs, padded, with runs shorter than `minSpeech`
/// dropped and gaps shorter than `minSilence` bridged.
fn keep_ranges(dbs: &[f32], o: &Opts) -> Vec<(f64, f64)> {
    let n = dbs.len();
    let per = WINDOW;
    let gap = ((o.min_silence / per).ceil() as usize).max(1);
    let speech = ((o.min_speech / per).ceil() as usize).max(1);
    let pad = ((o.padding / per).round() as usize).max(0);
    // Runs of windows at or above the threshold.
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < n {
        if dbs[i] >= o.threshold {
            let a = i;
            while i < n && dbs[i] >= o.threshold {
                i += 1;
            }
            runs.push((a, i));
        } else {
            i += 1;
        }
    }
    // Bridge short gaps (a pause inside a sentence is not a cut).
    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(runs.len());
    for (a, b) in runs {
        match merged.last_mut() {
            Some(last) if a - last.1 < gap => last.1 = b,
            _ => merged.push((a, b)),
        }
    }
    // Drop blips (a click, a breath) and pad each kept run, without overlapping the next one.
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(merged.len());
    for (a, b) in merged {
        if b - a < speech {
            continue;
        }
        let lo = a.saturating_sub(pad);
        let hi = (b + pad).min(n);
        let lo = out.last().map_or(lo, |last| lo.max(last.1));
        if hi > lo {
            out.push((lo, hi));
        }
    }
    out.into_iter().map(|(a, b)| (a as f64 * per, b as f64 * per)).collect()
}

/// Snapped, clamped kept spans of comp time, or an error when nothing is left to keep.
fn analyse(probe: &Probe, cid: ItemId, lid: LayerId, o: &Opts, progress: &dyn Fn(u64, u64) -> bool) -> std::result::Result<Vec<(Tick, Tick)>, String> {
    let comp = probe.project.comp(cid).ok_or("the composition is gone")?;
    let layer = comp.layer(lid).ok_or("the layer is gone")?;
    let (item, footage) = match &layer.source {
        LayerSource::Footage { item } => match probe.project.item(*item).map(|i| &i.kind) {
            Some(ItemKind::Footage(f)) if f.has_audio && !f.missing => (*item, f),
            _ => return Err("the layer has no audio to analyse".into()),
        },
        _ => return Err("Remove Silence needs a footage or audio layer".into()),
    };
    let start = layer.in_point.max(Tick::ZERO);
    let end = layer.out_point.min(comp.duration);
    if end <= start || (end - start).seconds() < WINDOW * 2.0 {
        return Err("the layer is too short to analyse".into());
    }
    let span = (end - start).seconds();
    let windows = ((span / WINDOW).ceil() as usize).max(1);
    let win = (RATE as f64 * WINDOW).round().max(1.0) as usize;
    let per_block = (BLOCK / win).max(1);
    let blocks = windows.div_ceil(per_block);
    let mut dbs: Vec<f32> = Vec::with_capacity(windows);
    for b in 0..blocks {
        let first = b * per_block;
        let count = per_block.min(windows - first);
        let frames = count * win;
        let t = start + Tick::from_seconds_f64(first as f64 * WINDOW);
        let buf = probe.samples(item, footage, layer.layer_time(t), frames);
        for w in 0..count {
            let a = w * win;
            let z = a + win;
            let mut sum = 0.0f64;
            let mut seen = 0usize;
            for i in a..z {
                let (Some(l), Some(r)) = (buf.get(i * 2), buf.get(i * 2 + 1)) else { break };
                sum += (f64::from(*l) * f64::from(*l) + f64::from(*r) * f64::from(*r)) * 0.5;
                seen += 1;
            }
            let rms = if seen == 0 { 0.0 } else { (sum / seen as f64).sqrt() };
            dbs.push((20.0 * rms.max(1e-9).log10()) as f32);
        }
        if !progress((b + 1) as u64, blocks as u64) {
            return Err(crate::render_queue::CANCELLED.to_string());
        }
    }
    let fr = comp.frame_rate;
    let snap = |t: f64| {
        let x = start + Tick::from_seconds_f64(t);
        fr.tick_of(fr.frame_at(fr.snap_nearest(x)))
    };
    let mut out: Vec<(Tick, Tick)> = Vec::new();
    for (a, b) in keep_ranges(&dbs, o) {
        let (a, b) = (snap(a).max(start), snap(b).min(end));
        if b <= a {
            continue;
        }
        match out.last_mut() {
            Some(last) if a <= last.1 => last.1 = b,
            _ => out.push((a, b)),
        }
    }
    if out.is_empty() {
        return Err("no speech found: the threshold or the minimum speech length may be wrong".into());
    }
    Ok(out)
}

/// One kept span of the tightened timeline: comp times `a..b` play at `new_a..`.
#[derive(Clone, Copy, Debug)]
struct Seg {
    a: Tick,
    b: Tick,
    new_a: Tick,
}

/// The piecewise map from the layer's timeline to the tightened one. Everything before the
/// layer and after it is kept where it is (shifted only by the total removed before it), so a
/// `ripple: all` edit moves the rest of the composition in step.
fn map_of(comp: &Comp, layer: &Layer, ranges: &[(Tick, Tick)]) -> Vec<Seg> {
    let mut segs = Vec::new();
    if layer.in_point > Tick::ZERO {
        segs.push(Seg { a: Tick::ZERO, b: layer.in_point, new_a: Tick::ZERO });
    }
    let mut at = layer.in_point;
    for (a, b) in ranges {
        if *b <= *a {
            continue;
        }
        segs.push(Seg { a: *a, b: *b, new_a: at });
        at += *b - *a;
    }
    let end = comp.layers.iter().map(|l| l.out_point).max().unwrap_or(comp.duration).max(comp.duration).max(layer.out_point);
    if end > layer.out_point {
        segs.push(Seg { a: layer.out_point, b: end, new_a: at });
    }
    segs
}

/// Fade the piece in and/or out with two keys on its Audio Levels, when they are not animated.
/// A cut in the middle of a waveform clicks; 5 ms is short enough to be inaudible.
fn fade_piece(piece: &mut Layer, fade_in: bool, fade_out: bool) {
    if !fade_in && !fade_out {
        return;
    }
    let Some(pr) = piece.props.sub("audio").and_then(|g| g.get("levels")) else { return };
    if !pr.keys.is_empty() {
        return;
    }
    let base = pr.value.as_vec2();
    let fade = Tick::from_seconds_f64(FADE);
    let a = piece.layer_time(piece.in_point);
    let b = piece.layer_time(piece.out_point);
    if b - a <= fade + fade {
        return;
    }
    let silent = KV::Vec2([-96.0, -96.0]);
    let mut keys = Vec::with_capacity(4);
    if fade_in {
        keys.push(Keyframe::new(a, silent.clone()));
        keys.push(Keyframe::new(a + fade, KV::Vec2(base)));
    }
    if fade_out {
        keys.push(Keyframe::new(b - fade, KV::Vec2(base)));
        keys.push(Keyframe::new(b, silent));
    }
    if let Some(pr) = piece.props.sub_mut("audio").and_then(|g| g.get_mut("levels")) {
        pr.keys = keys;
    }
}

/// The silences of `ranges` (its leading gap, the gaps between kept spans, its trailing gap).
fn gaps(layer: &Layer, ranges: &[(Tick, Tick)]) -> Vec<(Tick, Tick)> {
    let mut out = Vec::new();
    let mut at = layer.in_point;
    for (a, b) in ranges {
        if *a > at {
            out.push((at, *a));
        }
        at = *b;
    }
    if layer.out_point > at {
        out.push((at, layer.out_point));
    }
    out
}

fn mark(s: &mut Session, cid: ItemId, lid: LayerId, ranges: &[(Tick, Tick)]) -> Result<Value> {
    let comp = s.project.comp(cid).ok_or(EngineError::NoComp)?;
    let layer = comp.layer(lid).ok_or(EngineError::NoComp)?;
    let gaps = gaps(layer, ranges);
    let secs: Vec<f64> = gaps.iter().map(|(a, b)| (*b - *a).seconds()).collect();
    s.edit("Remove Silence", None, |proj, _| {
        let l = super::layer_mut(proj, cid, lid)?;
        let mut n = 0;
        for (a, _) in &gaps {
            let lt = l.layer_time(*a);
            if l.markers.iter().any(|m| m.time == lt) {
                continue;
            }
            n += 1;
            l.markers.push(Marker { time: lt, comment: format!("Silence {n}"), ..Default::default() });
        }
        l.markers.sort_by_key(|m| m.time);
        Ok(())
    })?;
    Ok(json!({"silences": secs}))
}

#[allow(clippy::too_many_arguments)]
fn apply(
    s: &mut Session,
    cid: ItemId,
    src: LayerId,
    mode: Mode,
    ranges: &[(Tick, Tick)],
    ripple_all: bool,
    keep_source: bool,
    precompose: bool,
) -> Result<Value> {
    let snap = s.project.clone();
    if mode == Mode::Markers {
        let r = mark(s, cid, src, ranges)?;
        let removed: f64 = r["silences"].as_array().into_iter().flatten().filter_map(Value::as_f64).sum();
        return Ok(json!({"layer": src.0, "mode": "markers", "silences": r["silences"], "removed": removed, "layers": [src.0]}));
    }
    let comp = s.project.comp(cid).ok_or(EngineError::NoComp)?.clone();
    let source = comp.layer(src).ok_or(EngineError::NoComp)?.clone();
    if ripple_all {
        if let Some(l) = comp.layers.iter().find(|l| l.track_matte.is_some() || l.parent.is_some()) {
            return Err(bad(
                "layer.removeSilence",
                format!("layer `{}` is parented or uses a track matte; remove the links, or choose Ripple: Selected layers", l.name),
            ));
        }
    } else if comp.layers.iter().any(|l| l.id != src && l.parent == Some(src)) {
        return Err(bad("layer.removeSilence", "other layers are parented to this one; choose Ripple: All unlocked layers or remove the parenting"));
    }
    if comp.layers.iter().any(|l| l.id != src && l.track_matte.is_some_and(|m| m.layer == src)) {
        return Err(bad("layer.removeSilence", "the layer is a track matte for another layer; remove the matte first"));
    }
    let mut segs = map_of(&comp, &source, ranges);
    if mode == Mode::Split {
        for s in &mut segs {
            s.new_a = s.a;
        }
    }
    let targets: Vec<LayerId> = if ripple_all { comp.layers.iter().filter(|l| !l.switches.locked).map(|l| l.id).collect() } else { vec![src] };
    let name = format!("{} (tight cut)", source.name);
    let new_ids = s.edit("Remove Silence", None, |proj, st| {
        let mut next = proj.next_id;
        let comp = proj.comp(cid).ok_or(EngineError::NoComp)?;
        let mut layers: Vec<Layer> = Vec::with_capacity(comp.layers.len());
        let mut ids: Vec<LayerId> = Vec::new();
        for l in &comp.layers {
            if !targets.contains(&l.id) {
                layers.push(l.clone());
                continue;
            }
            for seg in &segs {
                let lo = l.in_point.max(seg.a);
                let hi = l.out_point.min(seg.b);
                if hi <= lo {
                    continue;
                }
                let mut piece = l.clone();
                super::edit::reid(&mut piece, &mut next);
                let new_lo = if mode == Mode::Split { lo } else { seg.new_a + (lo - seg.a) };
                let len = hi - lo;
                piece.start_time = new_lo - (lo - l.start_time);
                piece.in_point = new_lo;
                piece.out_point = new_lo + len;
                piece.markers.retain(|m| m.time >= lo - l.start_time && m.time < hi - l.start_time);
                if mode == Mode::Tighten {
                    fade_piece(&mut piece, lo > l.in_point, hi < l.out_point);
                }
                ids.push(piece.id);
                layers.push(piece);
            }
        }
        let cm = proj.comp_mut(cid).ok_or(EngineError::NoComp)?;
        cm.layers = layers;
        st.selected_layers = ids.clone();
        st.selected_props.clear();
        st.selected_keys.clear();
        proj.next_id = next;
        Ok(ids)
    })?;
    let mut out = new_ids.clone();
    if mode == Mode::Tighten && precompose && !new_ids.is_empty() {
        let r = s.execute(
            "layer.precompose",
            json!({
                "comp": cid.0,
                "layers": new_ids.iter().map(|l| l.0).collect::<Vec<_>>(),
                "mode": "move",
                "adjustDuration": true,
                "name": name,
            }),
        )?;
        if let Some(l) = r.get("layer").and_then(Value::as_u64) {
            out = vec![LayerId(l)];
        }
    }
    if mode == Mode::Tighten
        && keep_source
        && let Some(source) = snap.comp(cid).and_then(|c| c.layer(src)).cloned()
    {
        s.edit("Remove Silence", None, |proj, _| {
            let mut next = proj.next_id;
            let mut l = source.clone();
            super::edit::reid(&mut l, &mut next);
            l.name = format!("{} (source)", l.name);
            l.switches.video = false;
            l.switches.audio = false;
            let c = proj.comp_mut(cid).ok_or(EngineError::NoComp)?;
            let at = c.layers.iter().position(|x| out.contains(&x.id)).map(|i| i + 1).unwrap_or(c.layers.len());
            c.layers.insert(at, l);
            proj.next_id = next;
            Ok(())
        })?;
    }
    s.collapse_undo("Remove Silence", &snap);
    let span = source.out_point - source.in_point;
    let kept = ranges.iter().fold(Tick::ZERO, |acc, (a, b)| acc + (*b - *a));
    Ok(json!({
        "layer": src.0,
        "mode": if mode == Mode::Split { "split" } else { "tighten" },
        "removed": (span - kept).seconds(),
        "kept": ranges.iter().map(|(a, b)| json!([a.seconds(), b.seconds()])).collect::<Vec<_>>(),
        "layers": out.iter().map(|l| l.0).collect::<Vec<_>>(),
    }))
}

fn run(s: &mut Session, p: &Value) -> Result<Value> {
    let c = "layer.removeSilence";
    let (cid, ids) = layers_p(s, p)?;
    let comp = s.project.comp(cid).ok_or(EngineError::NoComp)?;
    let src = ids
        .iter()
        .copied()
        .find(|id| comp.layer(*id).is_some_and(|l| has_audio(&s.project, l)))
        .ok_or_else(|| bad(c, "select a footage or audio layer with sound"))?;
    let layer = comp.layer(src).ok_or(EngineError::NoComp)?;
    if (layer.stretch - 100.0).abs() > 1e-9 {
        return Err(bad(c, "the layer is time-stretched; set Stretch back to 100% first"));
    }
    if !layer.switches.audio {
        return Err(bad(c, "the layer's Audio switch is off"));
    }
    let mode = match str_p(p, "mode").unwrap_or("tighten") {
        "markers" | "createMarkers" => Mode::Markers,
        "split" | "splitLayers" => Mode::Split,
        "tighten" | "tightenAndSplit" | "tightenPrecompose" => Mode::Tighten,
        x => return Err(bad(c, format!("mode: markers|split|tighten, not `{x}`"))),
    };
    let ripple_all = match str_p(p, "ripple").unwrap_or("selected") {
        "selected" | "layers" => false,
        "all" | "unlocked" => true,
        x => return Err(bad(c, format!("ripple: selected|all, not `{x}`"))),
    };
    let o = Opts::from(p)?;
    let keep_source = b_p(p, "keepSource").unwrap_or(true);
    let precompose = b_p(p, "precompose").unwrap_or(true);
    let wait = b_p(p, "wait").unwrap_or(false);
    let probe = Probe { project: s.project.clone(), footage: s.footage.clone() };
    let label = format!("Remove Silence: {}", layer.name);
    s.spawn_task("removeSilence", label, wait, move |ctl| {
        let ranges = analyse(&probe, cid, src, &o, &|done, total| ctl.progress(done, total))?;
        let apply: Apply = Box::new(move |s| apply(s, cid, src, mode, &ranges, ripple_all, keep_source, precompose));
        Ok(apply)
    })
}

pub fn specs() -> Vec<CommandSpec> {
    vec![cmd!(
        "layer.removeSilence",
        "Remove Silence...",
        ["Layer"],
        None,
        "{layer?|layers?, mode?: markers|split|tighten, threshold? (dBFS, -35), minSilence? (s, 0.35), minSpeech? (s, 0.2), padding? (s, 0.1), ripple?: selected|all, keepSource?: bool, precompose?: bool, wait?: bool}",
        enabled,
        run
    )]
}
