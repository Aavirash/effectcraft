//! Remove Silence and Generate Captions through the commands.

use std::sync::Arc;

use effectcraft_color::Label;
use effectcraft_keyframe::Value as KV;
use effectcraft_project::{Footage, FootageKind, ItemId, ItemKind, LayerId, LayerSource, build};
use effectcraft_raster::Image;
use effectcraft_render::FootageSource;
use effectcraft_time::{FrameRate, Tick};
use serde_json::json;

use crate::Session;

const FPS: u32 = 30;
/// Tone 0–1 s, silence 1–3 s, tone 3–4 s, silence 4–6 s: two silences of 2 s.
const SECS: f64 = 6.0;

/// A click track with the two silences above.
struct Tone;

impl FootageSource for Tone {
    fn frame(&self, _: ItemId, _: &Footage, _: Tick) -> Option<Arc<Image>> {
        None
    }
    fn audio(&self, _: ItemId, _: &Footage, t: Tick, frames: usize, rate: u32) -> Option<Vec<f32>> {
        let first = t.to_units_floor(i64::from(rate)).max(0);
        let mut out = Vec::with_capacity(frames * 2);
        for i in 0..frames {
            let secs = (first + i as i64) as f64 / f64::from(rate);
            let loud = secs < 1.0 || (3.0..4.0).contains(&secs);
            let s = if loud { (std::f64::consts::TAU * 440.0 * secs).sin() as f32 * 0.3 } else { 0.0 };
            out.push(s);
            out.push(s);
        }
        Some(out)
    }
}

/// A comp 640 × 360, `SECS` long, with one audio layer "Interview" (returns session, comp, layer).
fn setup() -> (Session, ItemId, LayerId) {
    let mut s = Session::default();
    s.footage = Arc::new(Tone);
    s.execute("comp.new", json!({"name": "Talk", "width": 640, "height": 360, "frameRate": FPS, "duration": SECS})).unwrap();
    let cid = s.active_comp_id().unwrap();
    let lid = s
        .edit("clip", None, |p, st| {
            let footage = Footage {
                path: "tone.wav".into(),
                kind: FootageKind::Audio,
                frame_rate: FrameRate::new(i64::from(FPS), 1),
                duration: Tick::from_seconds_f64(SECS),
                has_audio: true,
                ..Default::default()
            };
            let item = p.add_item("tone", Label::Aqua, None, ItemKind::Footage(footage));
            let comp = p.comp(cid).unwrap().clone();
            let mut l = build::layer(p, &comp, "Interview", LayerSource::Footage { item }, (comp.width, comp.height), None);
            l.start_time = Tick::ZERO;
            l.in_point = Tick::ZERO;
            l.out_point = Tick::from_seconds_f64(SECS);
            let id = l.id;
            p.comp_mut(cid).unwrap().layers.insert(0, l);
            st.selected_layers = vec![id];
            Ok(id)
        })
        .unwrap();
    (s, cid, lid)
}

fn spans(s: &Session, cid: ItemId) -> Vec<(f64, f64)> {
    let comp = s.project.comp(cid).unwrap();
    comp.layers.iter().map(|l| (l.in_point.seconds(), l.out_point.seconds())).collect()
}

#[test]
fn remove_silence_creates_markers_and_tightens() {
    let (mut s, cid, lid) = setup();
    // Markers: one per silence, at its start, in layer time.
    let r = s.execute("layer.removeSilence", json!({"mode": "markers", "minSilence": 0.3, "minSpeech": 0.1, "padding": 0.0, "wait": true})).unwrap();
    assert_eq!(r["silences"], json!([2.0, 2.0]));
    let l = s.project.comp(cid).unwrap().layer(lid).unwrap();
    let times: Vec<f64> = l.markers.iter().map(|m| m.time.seconds()).collect();
    assert_eq!(times, vec![1.0, 4.0]);
    assert_eq!(s.history.undo.last().unwrap().0, "Remove Silence");
    s.undo();
    assert!(s.project.comp(cid).unwrap().layer(lid).unwrap().markers.is_empty());

    // Tighten: two pieces, end to end, each still showing its own source range.
    let r = s
        .execute(
            "layer.removeSilence",
            json!({"mode": "tighten", "precompose": false, "keepSource": false, "minSilence": 0.3, "minSpeech": 0.1, "padding": 0.0, "wait": true}),
        )
        .unwrap();
    assert_eq!(r["removed"], json!(4.0));
    assert_eq!(spans(&s, cid), vec![(0.0, 1.0), (1.0, 2.0)]);
    let comp = s.project.comp(cid).unwrap();
    let pieces: Vec<&effectcraft_project::Layer> = comp.layers.iter().collect();
    assert_eq!(pieces.len(), 2);
    assert_eq!(pieces[0].layer_time(pieces[0].in_point).seconds(), 0.0);
    assert_eq!(pieces[1].layer_time(pieces[1].in_point).seconds(), 3.0);
    assert_eq!(r["layers"].as_array().unwrap().len(), 2);
    s.undo();
    assert_eq!(spans(&s, cid), vec![(0.0, 6.0)]);
}

#[test]
fn remove_silence_precomposes_and_keeps_the_source() {
    let (mut s, cid, _) = setup();
    s.execute(
        "layer.removeSilence",
        json!({"mode": "tighten", "precompose": true, "keepSource": true, "minSilence": 0.3, "minSpeech": 0.1, "padding": 0.0, "wait": true}),
    )
    .unwrap();
    let comp = s.project.comp(cid).unwrap();
    assert_eq!(comp.layers.len(), 2, "the tight cut and the disabled source");
    let pre = &comp.layers[0];
    assert!(matches!(pre.source, LayerSource::Comp { .. }), "the tight cut is a precomp layer");
    let src = &comp.layers[1];
    assert!(matches!(src.source, LayerSource::Footage { .. }));
    assert!(!src.switches.video && !src.switches.audio, "the source is disabled");
    assert!(src.name.contains("source"));
    assert_eq!(s.history.undo.last().unwrap().0, "Remove Silence");
    s.undo();
    assert_eq!(s.project.comp(cid).unwrap().layers.len(), 1);
}

#[test]
fn remove_silence_ripples_other_layers() {
    let (mut s, cid, _) = setup();
    s.execute("layer.newSolid", json!({"name": "Title"})).unwrap();
    s.edit("title", None, |p, _| {
        let cid = cid;
        let l = p.comp_mut(cid).unwrap().layers.iter_mut().find(|l| l.name == "Title").unwrap();
        l.start_time = Tick::from_seconds_f64(3.5);
        l.in_point = Tick::from_seconds_f64(3.5);
        l.out_point = Tick::from_seconds_f64(4.5);
        Ok(())
    })
    .unwrap();
    s.execute(
        "layer.removeSilence",
        json!({"mode": "tighten", "ripple": "all", "precompose": false, "keepSource": false, "minSilence": 0.3, "minSpeech": 0.1, "padding": 0.0, "wait": true}),
    )
    .unwrap();
    let comp = s.project.comp(cid).unwrap();
    let t = comp.layers.iter().find(|l| l.name == "Title").unwrap();
    assert!((t.in_point.seconds() - 1.5).abs() < 1e-9, "{:?}", t.in_point.seconds());
    assert!((t.out_point.seconds() - 2.0).abs() < 1e-9);
}

#[test]
fn captions_create_text_layers_per_cue() {
    let (mut s, cid, _) = setup();
    let cues = json!([
        {"start": 0.5, "end": 1.5, "text": "Hello there"},
        {"start": 2.0, "end": 3.0, "text": "Second line"}
    ]);
    let r = s.execute("captions.create", json!({"cues": cues, "style": "plain", "position": "bottom"})).unwrap();
    assert_eq!(r["cues"], json!(2));
    let comp = s.project.comp(cid).unwrap();
    let captions: Vec<&effectcraft_project::Layer> = comp.layers.iter().filter(|l| matches!(l.source, LayerSource::Text)).collect();
    assert_eq!(captions.len(), 2);
    assert_eq!(captions[0].name, "Caption 01");
    assert_eq!(captions[1].name, "Caption 02");
    assert_eq!((captions[0].in_point.seconds(), captions[0].out_point.seconds()), (0.5, 1.5));
    let text = |l: &effectcraft_project::Layer| match &l.props.prop("text/sourceText").unwrap().value {
        KV::Text(d) => d.text.clone(),
        _ => String::new(),
    };
    assert_eq!(text(captions[0]), "Hello there");
    assert_eq!(text(captions[1]), "Second line");
    assert_eq!(s.history.undo.last().unwrap().0, "Generate Captions");
    s.undo();
    assert_eq!(s.project.comp(cid).unwrap().layers.len(), 1, "only the footage layer is left");
}

#[test]
fn caption_styles_box_karaoke_and_pop() {
    let (mut s, cid, _) = setup();
    let cues = json!([{"start": 1.0, "end": 2.0, "text": "One two three"}]);
    s.execute("captions.create", json!({"cues": cues, "style": "boxed"})).unwrap();
    let comp = s.project.comp(cid).unwrap();
    assert_eq!(comp.layers.iter().filter(|l| matches!(l.source, LayerSource::Text)).count(), 1);
    assert_eq!(comp.layers.iter().filter(|l| matches!(l.source, LayerSource::Shape)).count(), 1, "the text and its box");
    let text = &comp.layers[0];
    let b = &comp.layers[1];
    assert!(matches!(text.source, LayerSource::Text));
    assert!(matches!(b.source, LayerSource::Shape));
    assert_eq!(b.parent, Some(text.id));
    s.undo();

    let cues = json!([{"start": 1.0, "end": 2.0, "text": "One two three"}]);
    s.execute("captions.create", json!({"cues": cues, "style": "karaoke"})).unwrap();
    let l = &s.project.comp(cid).unwrap().layers[0];
    let anims = l.props.group("text/animators").unwrap();
    let anim = anims.groups().next().unwrap();
    assert_eq!(anim.name, "Karaoke");
    let sel = anim.sub("selectors").and_then(|s| s.groups().next()).unwrap();
    assert!(sel.get("end").is_some_and(|p| p.keys.len() >= 2), "the reveal is keyed at the words");
    s.undo();

    let cues = json!([{"start": 1.0, "end": 2.0, "text": "One two three"}]);
    s.execute("captions.create", json!({"cues": cues, "style": "pop"})).unwrap();
    let comp = s.project.comp(cid).unwrap();
    let words: Vec<&effectcraft_project::Layer> = comp.layers.iter().filter(|l| matches!(l.source, LayerSource::Text)).collect();
    assert_eq!(words.len(), 3, "one word at a time");
    assert!(words.iter().all(|l| l.props.prop("transform/scale").is_some_and(|p| !p.keys.is_empty())));
}

#[test]
fn captions_import_reads_an_srt_file() {
    let (mut s, cid, _) = setup();
    let path = std::env::temp_dir().join(format!("effectcraft-captions-{}.srt", std::process::id()));
    std::fs::write(&path, "1\n00:00:01,000 --> 00:00:02,000\nFirst cue\n\n2\n00:00:03,000 --> 00:00:04,000\nSecond cue\n").unwrap();
    let r = s.execute("captions.import", json!({"path": path.display().to_string(), "style": "plain"})).unwrap();
    assert_eq!(r["cues"], json!(2));
    assert_eq!(s.project.comp(cid).unwrap().layers.iter().filter(|l| matches!(l.source, LayerSource::Text)).count(), 2);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn remove_silence_needs_audio_and_never_touches_a_locked_layer() {
    let (mut s, cid, lid) = setup();
    s.edit("lock", None, |p, _| {
        if let Some(l) = p.comp_mut(cid).and_then(|c| c.layer_mut(lid)) {
            l.switches.locked = true;
        }
        Ok(())
    })
    .unwrap();
    // Ripple all: the locked layer is not remapped and stays whole.
    s.execute(
        "layer.removeSilence",
        json!({"mode": "tighten", "ripple": "all", "precompose": false, "keepSource": false, "minSilence": 0.3, "minSpeech": 0.1, "padding": 0.0, "wait": true}),
    )
    .unwrap();
    assert_eq!(spans(&s, cid), vec![(0.0, 6.0)]);

    let mut none = Session::default();
    none.execute("comp.new", json!({"name": "Empty", "width": 64, "height": 64, "frameRate": FPS, "duration": 1})).unwrap();
    none.execute("layer.newSolid", json!({"name": "Flat"})).unwrap();
    assert!(none.execute("layer.removeSilence", json!({"wait": true})).is_err());
}

#[test]
fn silence_analysis_finds_gaps_of_any_threshold() {
    // The same tone with a stricter threshold still finds the two silences, and a looser one
    // finds the same spans: the tone is at -10 dBFS and the silence at the floor.
    let (mut s, _, _) = setup();
    let r = s
        .execute("layer.removeSilence", json!({"mode": "markers", "threshold": -20.0, "minSilence": 0.5, "minSpeech": 0.1, "padding": 0.0, "wait": true}))
        .unwrap();
    assert_eq!(r["silences"], json!([2.0, 2.0]));
}
