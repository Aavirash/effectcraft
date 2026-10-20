//! Copied keyframes as After Effects keyframe text, for the system clipboard (#1908). Other
//! applications read this tab-separated format (AE2Blend brings it into Blender). Like After
//! Effects it covers one layer: its copied properties, keyframed ones as one row per key (frame
//! number in comp time, then the value) and static ones as a single row with no frame. Easing
//! and tangents aren't part of the format; EffectCraft's own paste keeps them.

use effectcraft_project::{AutoOrient, Comp, ItemKind, Layer, LayerId, LayerSource, Project, Property, Uid};

use crate::KeyRef;

/// Transform properties with the unit their columns are labelled with and, for vectors, the
/// third component a 2D layer shows. X / Y / Z Position (Separate Dimensions) have no unit.
const TRANSFORM: &[(&str, &str, Option<f64>)] = &[
    ("poi", "pixels", Some(0.0)),
    ("anchor", "pixels", Some(0.0)),
    ("position", "pixels", Some(0.0)),
    ("positionX", "", None),
    ("positionY", "", None),
    ("positionZ", "", None),
    ("scale", "percent", Some(100.0)),
    ("orientation", "degrees", None),
    ("rotationX", "degrees", None),
    ("rotationY", "degrees", None),
    ("rotation", "degrees", None),
    ("opacity", "percent", None),
];

/// How a copied property is written: its block's header line, column labels and values.
struct Channel {
    header: String,
    labels: Vec<String>,
    values: Values,
}

/// How a property's value becomes the numbers in a row.
#[derive(Clone, Copy)]
enum Values {
    Plain,
    /// A 2D layer's Transform vector: the third component is the one the layer shows (it may
    /// keep another from pasted 3D keys).
    Flat(f64),
}

impl Values {
    fn row(self, mut v: Vec<f64>) -> Vec<f64> {
        if let (Values::Flat(z), Some(c)) = (self, v.get_mut(2)) {
            *c = z;
        }
        v
    }
}

/// The text for copied keys (layer times) and unanimated properties. `None` unless they all come
/// from one layer and at least one property can be written.
pub(crate) fn text(project: &Project, comp: &Comp, keys: &[KeyRef], statics: &[(LayerId, Uid)]) -> Option<String> {
    let lid = keys.first().map(|k| k.layer).or_else(|| statics.first().map(|s| s.0))?;
    if keys.iter().any(|k| k.layer != lid) || statics.iter().any(|s| s.0 != lid) {
        return None;
    }
    let layer = comp.layer(lid)?;
    let fps = comp.frame_rate.as_f64();
    let mut blocks = String::new();
    layer.props.walk("", &mut |_, pr| {
        let mut times: Vec<_> = keys.iter().filter(|k| k.prop == pr.uid).map(|k| k.time).collect();
        times.sort();
        times.dedup();
        let rows: Vec<(String, Vec<f64>)> = if !times.is_empty() {
            let frame = |t| num(layer.comp_time(t).seconds() * fps);
            times.into_iter().filter_map(|t| Some((frame(t), pr.keys.iter().find(|k| k.time == t)?.value.components()))).collect()
        } else if statics.iter().any(|s| s.1 == pr.uid) {
            vec![(String::new(), pr.value.components())]
        } else {
            return;
        };
        let Some(ch) = channel(layer, pr).filter(|ch| !ch.labels.is_empty()) else { return };
        if rows.iter().any(|(_, v)| v.is_empty()) {
            return;
        }
        blocks.push_str(&format!("{}\n\tFrame\t{}\t\n", ch.header, ch.labels.join("\t")));
        for (frame, values) in rows {
            let values: Vec<String> = ch.values.row(values).into_iter().map(num).collect();
            blocks.push_str(&format!("\t{frame}\t{}\t\n", values.join("\t")));
        }
        blocks.push('\n');
    });
    if blocks.is_empty() {
        return None;
    }
    // Layers without a source of their own (text, shapes, cameras) take the comp's size and
    // pixel aspect.
    let (w, h) = match effectcraft_render::eval::source_size(project, layer) {
        (0, 0) => (comp.width, comp.height),
        size => size,
    };
    let par = match layer.source {
        LayerSource::Footage { item } | LayerSource::Solid { item } | LayerSource::Comp { item } => match project.item(item).map(|i| &i.kind) {
            Some(ItemKind::Footage(f)) => f.pixel_aspect,
            Some(ItemKind::Solid(s)) => s.pixel_aspect,
            Some(ItemKind::Comp(c)) => c.pixel_aspect,
            _ => comp.pixel_aspect,
        },
        _ => comp.pixel_aspect,
    };
    // The layout of current After Effects: a `Layer` line before the layer's properties (parsers
    // such as ae-keyframes reject properties without one), and a tab after every column.
    Some(format!(
        "Adobe After Effects 9.0 Keyframe Data\n\n\tUnits Per Second\t{}\n\tSource Width\t{w}\n\tSource Height\t{h}\n\tSource Pixel Aspect Ratio\t{}\n\tComp Pixel Aspect Ratio\t{}\n\nLayer\n{blocks}\nEnd of Keyframe Data\n",
        num(fps),
        num(par),
        num(comp.pixel_aspect)
    ))
}

/// How the layer's property `pr` is written, by where it sits in the layer: `None` for
/// properties the text doesn't carry.
fn channel(layer: &Layer, pr: &Property) -> Option<Channel> {
    let chain = layer.props.node_chain(pr.uid)?;
    match chain.first().and_then(|n| n.as_group()).map(|g| g.match_id.as_str()) {
        None if pr.match_id == "timeRemap" => Some(Channel { header: "Time Remap".into(), labels: labels("seconds", 1), values: Values::Plain }),
        Some("transform") => transform(layer, pr),
        _ => None,
    }
}

fn transform(layer: &Layer, pr: &Property) -> Option<Channel> {
    let (id, unit, z) = TRANSFORM.iter().find(|(id, ..)| *id == pr.match_id)?;
    if !active(layer, id) || (pr.three_d_only && !layer.is_3d()) {
        return None;
    }
    // After Effects labels only the first of Orientation's three columns, and names Z Rotation
    // plain Rotation (as AE2Blend expects).
    let labels = if *id == "orientation" { vec![format!("X {unit}")] } else { labels(unit, pr.value.dims()) };
    let values = match z {
        Some(z) if !layer.is_3d() => Values::Flat(*z),
        _ => Values::Plain,
    };
    Some(Channel { header: format!("Transform\t{}", pr.name), labels, values })
}

/// Column labels: the unit, or one per axis (`X pixels`, `Y pixels`).
fn labels(unit: &str, dims: usize) -> Vec<String> {
    match dims {
        0 => vec![],
        1 => vec![unit.to_string()],
        _ => ["X", "Y", "Z"].iter().take(dims).map(|axis| format!("{axis} {unit}").trim_end().to_string()).collect(),
    }
}

/// Whether the layer uses this Transform property: one-node cameras (and lights not aimed at
/// it) have no Point of Interest, and Separate Dimensions replaces Position with X / Y / Z
/// Position.
fn active(layer: &Layer, id: &str) -> bool {
    match id {
        "poi" => !(layer.is_camera() || layer.is_light()) || layer.auto_orient == AutoOrient::TowardsPointOfInterest,
        "position" => layer.transform().is_none_or(|tr| tr.get("positionX").is_none()),
        _ => true,
    }
}

/// A number with at most three decimals and no trailing zeros (`23.976`, `960`, `-0.5`).
fn num(v: f64) -> String {
    let s = format!("{:.3}", if v.is_finite() { v } else { 0.0 });
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s == "-0" { "0".into() } else { s.into() }
}
