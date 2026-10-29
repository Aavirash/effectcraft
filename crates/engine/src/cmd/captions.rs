//! Layer ▸ Generate Captions…: turn a transcript (SRT, WebVTT or EffectCraft's JSON) into text
//! layers, one per cue, styled with one of the built-in caption styles. The layers are ordinary
//! text layers: they can be edited, animated, masked and restyled like any other, and agents can
//! pass cues straight to `captions.create` without a file.

use effectcraft_keyframe::{Interp, Justify, Keyframe, TextDoc, Value as KV};
use effectcraft_project::build::{self, Ids};
use effectcraft_project::{Comp, Layer, LayerId, LayerSource, Project};
use effectcraft_time::Tick;
use serde_json::{Value, json};

use super::{CommandSpec, b_p, bad, comp_id, f_p, str_p};
use crate::{EngineError, Result, Session, cmd};

/// The karaoke highlight colour.
const ACCENT: [f64; 4] = [1.0, 0.82, 0.08, 1.0];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Style {
    Plain,
    Boxed,
    Karaoke,
    Pop,
}

impl Style {
    fn parse(s: &str) -> Option<Style> {
        Some(match s {
            "plain" | "bold" | "boldWhite" | "simple" => Style::Plain,
            "boxed" | "box" => Style::Boxed,
            "karaoke" => Style::Karaoke,
            "pop" | "word" | "words" => Style::Pop,
            _ => return None,
        })
    }
}

#[derive(Clone, Debug)]
struct Word {
    start: f64,
    end: f64,
    text: String,
}

#[derive(Clone, Debug)]
struct Cue {
    start: f64,
    end: f64,
    text: String,
    words: Vec<Word>,
}

impl Cue {
    fn valid(&self) -> bool {
        self.start.is_finite() && self.end.is_finite() && self.end > self.start && !self.text.trim().is_empty()
    }
}

/// A number from JSON: a number, or a string holding one (some transcript exports quote times).
fn num(v: Option<&Value>) -> Option<f64> {
    match v? {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse::<f64>().ok(),
        _ => None,
    }
}

fn key<'a>(v: &'a Value, names: &[&str]) -> Option<&'a Value> {
    names.iter().find_map(|n| v.get(*n))
}

fn text_of(v: &Value) -> String {
    key(v, &["text", "content", "sentence", "caption"]).and_then(Value::as_str).unwrap_or_default().trim().to_string()
}

fn words_of(v: &Value) -> Vec<Word> {
    key(v, &["words", "tokens"])
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|w| {
                    let text = key(w, &["text", "word", "content"]).and_then(Value::as_str)?.trim().to_string();
                    let start = num(key(w, &["start", "startTime", "from"]))?;
                    let end = num(key(w, &["end", "endTime", "to"])).unwrap_or(start);
                    (!text.is_empty()).then_some(Word { start, end: end.max(start), text })
                })
                .collect()
        })
        .unwrap_or_default()
}

fn cue_of(v: &Value) -> Option<Cue> {
    let start = num(key(v, &["start", "startTime", "from"]))?;
    let end = num(key(v, &["end", "endTime", "to"]))?;
    let text = text_of(v);
    Some(Cue { start, end, text, words: words_of(v) })
}

/// `HH:MM:SS,mmm`, `HH:MM:SS.mmm`, `MM:SS.mmm` or plain seconds.
fn seconds(s: &str) -> Option<f64> {
    let s = s.trim().replace(',', ".");
    let parts: Vec<&str> = s.split(':').collect();
    let sec = |x: &str| x.trim().parse::<f64>().ok();
    match parts.as_slice() {
        [h, m, x] => Some(h.parse::<f64>().ok()? * 3600.0 + m.parse::<f64>().ok()? * 60.0 + sec(x)?),
        [m, x] => Some(m.parse::<f64>().ok()? * 60.0 + sec(x)?),
        [x] => sec(x),
        _ => None,
    }
}

/// Strip SRT/VTT markup (`<i>`, `<c.yellow>`, `{\an8}`) from a caption line.
fn plain(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut depth = 0usize;
    for c in s.chars() {
        match c {
            '<' | '{' => depth += 1,
            '>' | '}' if depth > 0 => depth -= 1,
            _ if depth == 0 => out.push(c),
            _ => {}
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn parse_srt(text: &str) -> Vec<Cue> {
    let normal = text.replace("\r\n", "\n").replace('\r', "\n");
    let mut cues = Vec::new();
    for block in normal.split("\n\n") {
        let Some(line) = block.lines().find(|l| l.contains("-->")) else { continue };
        let mut parts = line.split("-->");
        let (Some(a), Some(b)) = (parts.next().and_then(seconds), parts.next().map(|s| s.split_whitespace().next().unwrap_or("")).and_then(seconds)) else {
            continue;
        };
        let after = block.lines().skip_while(|l| *l != line).skip(1);
        let body = after.map(plain).filter(|l| !l.is_empty()).collect::<Vec<_>>().join(" ");
        if body.is_empty() {
            continue;
        }
        cues.push(Cue { start: a, end: b.max(a), text: body, words: Vec::new() });
    }
    cues.sort_by(|a, b| a.start.total_cmp(&b.start));
    cues
}

fn parse_json(text: &str) -> Result<Vec<Cue>> {
    let v: Value = serde_json::from_str(text).map_err(|e| bad("captions.import", format!("the transcript is not valid JSON: {e}")))?;
    let list = match &v {
        Value::Array(a) => a.clone(),
        Value::Object(_) => key(&v, &["cues", "segments", "captions", "transcript"]).and_then(Value::as_array).cloned().unwrap_or_default(),
        _ => Vec::new(),
    };
    let cues: Vec<Cue> = list.iter().filter_map(cue_of).collect();
    if cues.is_empty() {
        return Err(bad("captions.import", "the JSON has no cues (expected `{\"cues\": [{\"start\": 0, \"end\": 1, \"text\": \"…\"}]}`)"));
    }
    Ok(cues)
}

fn parse(text: &str) -> Result<Vec<Cue>> {
    let t = text.trim_start();
    let cues = if t.starts_with('{') || t.starts_with('[') { parse_json(t)? } else { parse_srt(text) };
    if cues.is_empty() {
        return Err(bad("captions.import", "no cues found; expected SRT (`00:00:01,000 --> 00:00:03,000`), WebVTT or JSON"));
    }
    Ok(cues)
}

/// Where the caption box sits and how big it is (comp pixels).
struct Layout {
    size: f64,
    bounds: [f64; 2],
    centre: [f64; 2],
}

fn layout(comp: &Comp, position: &str, margin: f64) -> Layout {
    let h = f64::from(comp.height);
    let w = f64::from(comp.width);
    let size = (h * 0.050).round().clamp(16.0, 400.0);
    let bounds = [(w * 0.76).max(size * 4.0), (size * 1.35 * 3.0).max(size * 2.0)];
    let m = h * (margin.clamp(0.0, 40.0) / 100.0).max(0.0);
    let y = match position {
        "top" => m + bounds[1] / 2.0,
        "center" | "middle" => h / 2.0,
        _ => h - m - bounds[1] / 2.0,
    };
    Layout { size, bounds, centre: [w / 2.0, y] }
}

fn doc(cue: &Cue, lay: &Layout) -> TextDoc {
    TextDoc {
        text: cue.text.clone(),
        font: "Inter".into(),
        style: "Bold".into(),
        size: lay.size,
        fill: [1.0, 1.0, 1.0, 1.0],
        stroke: [0.0, 0.0, 0.0, 1.0],
        stroke_width: lay.size * 0.09,
        apply_fill: true,
        apply_stroke: true,
        stroke_over_fill: false,
        justify: Justify::Center,
        box_size: Some(lay.bounds),
        box_pos: [-lay.bounds[0] / 2.0, -lay.bounds[1] / 2.0],
        ..Default::default()
    }
}

fn text_layer(proj: &mut Project, comp: &Comp, name: &str, cue: &Cue, lay: &Layout) -> Layer {
    let mut l = build::layer(proj, comp, name, LayerSource::Text, (comp.width, comp.height), None);
    if let Some(pr) = l.props.prop_mut("text/sourceText") {
        pr.value = KV::Text(Box::new(doc(cue, lay)));
    }
    if let Some(pr) = l.props.prop_mut("transform/position") {
        pr.value = KV::Vec3([lay.centre[0], lay.centre[1], 0.0]);
    }
    let t = |s: f64| Tick::from_seconds_f64(s);
    l.start_time = t(cue.start);
    l.in_point = t(cue.start);
    l.out_point = t(cue.end);
    l
}

/// The box behind a Boxed caption, parented to its text so it follows any move.
fn box_layer(proj: &mut Project, comp: &Comp, name: &str, cue: &Cue, lay: &Layout, text: LayerId) -> Option<Layer> {
    let mut paint = super::shape_tool::ShapeTool::default();
    paint.fill.color = [0.0, 0.0, 0.0];
    paint.fill.opacity = 72.0;
    paint.stroke_width = 0.0;
    let g = {
        let mut ids = Ids(&mut proj.next_id);
        super::shape_tool::shape_group(&mut ids, "rounded", [lay.bounds[0] + lay.size * 0.7, lay.bounds[1] + lay.size * 0.4], &paint)?
    };
    let mut l = build::layer(proj, comp, name, LayerSource::Shape, (comp.width, comp.height), None);
    if let Some(pr) = l.props.prop_mut("transform/position") {
        pr.value = KV::Vec3([0.0, 0.0, 0.0]);
    }
    l.props.sub_mut("contents")?.children.push(g.into());
    l.parent = Some(text);
    let t = |s: f64| Tick::from_seconds_f64(s);
    l.start_time = t(cue.start);
    l.in_point = t(cue.start);
    l.out_point = t(cue.end);
    Some(l)
}

/// The words of a cue with timings: the transcript's, or the cue's time split by characters.
fn words(cue: &Cue) -> Vec<Word> {
    if !cue.words.is_empty() {
        return cue.words.clone();
    }
    let parts: Vec<&str> = cue.text.split_whitespace().collect();
    let total: usize = parts.iter().map(|p| p.chars().count()).sum::<usize>().max(1);
    let dur = (cue.end - cue.start).max(0.0);
    let mut at = cue.start;
    let mut out = Vec::with_capacity(parts.len());
    for p in parts {
        let n = p.chars().count().max(1) as f64;
        let len = dur * n / total as f64;
        out.push(Word { start: at, end: at + len, text: p.to_string() });
        at += len;
    }
    out
}

fn hold(t: Tick, v: f64) -> Keyframe {
    let mut k = Keyframe::new(t, KV::Scalar(v));
    k.in_interp = Interp::Hold;
    k.out_interp = Interp::Hold;
    k
}

/// Highlight the cue word by word: the Range selector's End walks from 0 to 100% at the words'
/// starts, and the animator's Fill Color paints what it selects.
fn karaoke(proj: &mut Project, l: &mut Layer, cue: &Cue) {
    let ws = words(cue);
    let total = cue.text.chars().count().max(1) as f64;
    let mut ids = Ids(&mut proj.next_id);
    let props = build::text_anim_props(&mut ids, "fillColor", false);
    let mut anim = build::text_animator(&mut ids, "Karaoke", props);
    if let Some(pr) = anim.sub_mut("properties").and_then(|g| g.get_mut("fillColor")) {
        pr.value = KV::Color(ACCENT);
    }
    let mut chars = 0usize;
    let mut keys = vec![hold(Tick::ZERO, 0.0)];
    for (i, w) in ws.iter().enumerate() {
        if i > 0 {
            chars += 1;
        }
        chars += w.text.chars().count();
        let pct = (chars as f64 / total * 100.0).clamp(0.0, 100.0);
        let t = Tick::from_seconds_f64((w.start - cue.start).max(0.0));
        // The first word starts with the cue: a millisecond in, so the cue begins unhighlighted.
        let t = if t == Tick::ZERO && i == 0 { Tick::from_seconds_f64(0.001) } else { t };
        if keys.last().is_some_and(|k| k.time == t) {
            if let Some(last) = keys.last_mut() {
                last.value = KV::Scalar(pct);
            }
        } else if keys.last().is_some_and(|k| k.time < t) {
            keys.push(hold(t, pct));
        }
    }
    if keys.len() > 1
        && let Some(sel) = anim.sub_mut("selectors").and_then(|s| s.children.iter_mut().find_map(effectcraft_project::Node::as_group_mut))
        && let Some(pr) = sel.get_mut("end")
    {
        pr.keys = keys;
    }
    if let Some(anims) = l.props.sub_mut("text/animators") {
        anims.children.push(anim.into());
    }
}

/// A Pop caption: one word at a time, scaling up from nothing.
fn pop_layers(proj: &mut Project, comp: &Comp, name: &str, cue: &Cue, lay: &Layout) -> Vec<Layer> {
    let mut out = Vec::new();
    for (i, w) in words(cue).into_iter().enumerate() {
        let end = if w.end > w.start { w.end } else { w.start + 0.2 };
        let word = Cue { start: w.start, end, text: w.text.clone(), words: Vec::new() };
        let mut l = text_layer(proj, comp, &format!("{name}.{}", i + 1), &word, lay);
        if let Some(pr) = l.props.prop_mut("transform/scale") {
            pr.keys = vec![
                Keyframe::new(Tick::ZERO, KV::Vec3([0.0, 0.0, 100.0])),
                Keyframe::new(Tick::from_seconds_f64(0.08), KV::Vec3([118.0, 118.0, 100.0])),
                Keyframe::new(Tick::from_seconds_f64(0.16), KV::Vec3([100.0, 100.0, 100.0])),
            ];
        }
        if let Some(pr) = l.props.prop_mut("transform/opacity") {
            pr.keys = vec![Keyframe::new(Tick::ZERO, KV::Scalar(0.0)), Keyframe::new(Tick::from_seconds_f64(0.06), KV::Scalar(100.0))];
        }
        out.push(l);
    }
    out
}

/// Build the caption layers, top to bottom, in cue order.
fn build_layers(proj: &mut Project, comp: &Comp, cues: &[Cue], style: Style, lay: &Layout, name: &str) -> Vec<Layer> {
    let mut out = Vec::new();
    for (i, cue) in cues.iter().enumerate() {
        let name = format!("{name} {:02}", i + 1);
        match style {
            Style::Pop => out.extend(pop_layers(proj, comp, &name, cue, lay)),
            Style::Plain | Style::Boxed | Style::Karaoke => {
                let mut l = text_layer(proj, comp, &name, cue, lay);
                if style == Style::Karaoke {
                    karaoke(proj, &mut l, cue);
                }
                let id = l.id;
                out.push(l);
                if style == Style::Boxed
                    && let Some(b) = box_layer(proj, comp, &format!("{name} box"), cue, lay, id)
                {
                    out.push(b);
                }
            }
        }
    }
    out
}

/// Clamp, sort and drop cues outside the composition.
fn prepare(cues: &mut Vec<Cue>, comp: &Comp) {
    let end = comp.duration.seconds();
    cues.retain_mut(|c| {
        c.start = c.start.max(0.0);
        c.end = c.end.min(end);
        c.valid()
    });
    cues.sort_by(|a, b| a.start.total_cmp(&b.start));
}

fn generate(s: &mut Session, p: &Value, cues: &mut Vec<Cue>) -> Result<Value> {
    let c = if p.get("path").is_some() { "captions.import" } else { "captions.create" };
    let cid = comp_id(s, p)?;
    let comp = s.project.comp(cid).ok_or(EngineError::NoComp)?.clone();
    let style = match Style::parse(str_p(p, "style").unwrap_or("plain")) {
        Some(st) => st,
        None => return Err(bad(c, "style: plain|boxed|karaoke|pop")),
    };
    let position = str_p(p, "position").unwrap_or("bottom");
    if !matches!(position, "bottom" | "top" | "center" | "middle") {
        return Err(bad(c, "position: bottom|top|center"));
    }
    let margin = f_p(p, "margin").unwrap_or(8.0);
    if !margin.is_finite() {
        return Err(bad(c, "`margin` must be a number"));
    }
    let precompose = b_p(p, "precompose").unwrap_or(false);
    let name = str_p(p, "name").unwrap_or("Caption").to_string();
    prepare(cues, &comp);
    if cues.is_empty() {
        return Err(bad(c, "no cues fall inside the composition"));
    }
    let anchor = p.get("layer").and_then(Value::as_u64).map(LayerId);
    let lay = layout(&comp, position, margin);
    let ids = s.edit("Generate Captions", None, |proj, st| {
        let mut layers = build_layers(proj, &comp, cues, style, &lay, &name);
        let cm = proj.comp_mut(cid).ok_or(EngineError::NoComp)?;
        let at = anchor.and_then(|a| cm.layers.iter().position(|l| l.id == a)).unwrap_or(0);
        let ids: Vec<LayerId> = layers.iter().map(|l| l.id).collect();
        for l in layers.drain(..).rev() {
            cm.layers.insert(at, l);
        }
        st.selected_layers = ids.clone();
        st.selected_props.clear();
        st.selected_keys.clear();
        Ok(ids)
    })?;
    let mut out = ids.clone();
    if precompose {
        let r = s.execute(
            "layer.precompose",
            json!({
                "comp": cid.0,
                "layers": ids.iter().map(|l| l.0).collect::<Vec<_>>(),
                "mode": "move",
                "adjustDuration": false,
                "name": "Captions",
            }),
        )?;
        if let Some(l) = r.get("layer").and_then(Value::as_u64) {
            out = vec![LayerId(l)];
        }
    }
    Ok(json!({"layers": out.iter().map(|l| l.0).collect::<Vec<_>>(), "cues": cues.len(), "style": format!("{style:?}").to_lowercase()}))
}

fn create(s: &mut Session, p: &Value) -> Result<Value> {
    let c = "captions.create";
    let list = p.get("cues").and_then(Value::as_array).ok_or_else(|| bad(c, "missing `cues`"))?;
    let mut cues: Vec<Cue> = list.iter().filter_map(cue_of).collect();
    if cues.len() != list.len() {
        return Err(bad(c, "every cue needs `start`, `end` and `text`"));
    }
    generate(s, p, &mut cues)
}

fn import(s: &mut Session, p: &Value) -> Result<Value> {
    let c = "captions.import";
    let path = str_p(p, "path").ok_or_else(|| bad(c, "missing `path` (an SRT, VTT or JSON transcript)"))?;
    let text = std::fs::read_to_string(path).map_err(|e| bad(c, format!("cannot read {path}: {e}")))?;
    let mut cues = parse(&text)?;
    generate(s, p, &mut cues)
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        cmd!(
            "captions.import",
            "Generate Captions...",
            ["Layer"],
            None,
            "{comp?, layer?, path: .srt|.vtt|.json, style?: plain|boxed|karaoke|pop, position?: bottom|top|center, margin? (percent of the height, 8), precompose?: bool, name?}",
            |s: &Session| {
                let comp = s.active_comp().ok_or("no composition is open")?;
                if comp.layers.is_empty() { Err("the composition has no layers".into()) } else { Ok(()) }
            },
            import
        ),
        cmd!(
            "captions.create",
            "Create Captions",
            [],
            None,
            "{comp?, layer?, cues: [{start, end, text, words?}], style?: plain|boxed|karaoke|pop, position?: bottom|top|center, margin? (percent of the height, 8), precompose?: bool, name?}",
            |s: &Session| s.active_comp().map(|_| ()).ok_or_else(|| "no composition is open".to_string()),
            create
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srt_parses_cues_and_strips_markup() {
        let text = "1\r\n00:00:01,240 --> 00:00:03,080\r\n<i>So today</i> I want\r\nto show you.\r\n\r\n\
                    2\r\n00:00:03,120 --> 00:00:05,400\r\n{\\an8}How we built this.\r\n";
        let cues = parse(text).unwrap();
        assert_eq!(cues.len(), 2);
        assert!((cues[0].start - 1.24).abs() < 1e-9);
        assert_eq!(cues[0].text, "So today I want to show you.");
        assert!((cues[1].start - 3.12).abs() < 1e-9);
        assert_eq!(cues[1].text, "How we built this.");
    }

    #[test]
    fn vtt_parses_cues_with_settings_and_notes() {
        let text = "WEBVTT\n\nNOTE a comment\n\ncue-1\n00:00:01.000 --> 00:00:02.500 align:middle\nHello there\n\n\
                    00:00:03.000 --> 00:00:04.000\nSecond line\n";
        let cues = parse(text).unwrap();
        assert_eq!(cues.len(), 2);
        assert_eq!(cues[0].text, "Hello there");
        assert!((cues[1].end - 4.0).abs() < 1e-9);
    }

    #[test]
    fn json_parses_segments_and_words() {
        let text = r#"{"segments":[{"start":0.5,"end":1.5,"text":"Hi","words":[{"start":0.5,"end":0.9,"text":"Hi"}]}]}"#;
        let cues = parse(text).unwrap();
        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0].words.len(), 1);
        assert_eq!(cues[0].words[0].text, "Hi");
    }

    #[test]
    fn json_without_cues_is_an_error() {
        assert!(parse(r#"{"foo": 1}"#).is_err());
        assert!(parse("not a transcript").is_err());
    }
}
