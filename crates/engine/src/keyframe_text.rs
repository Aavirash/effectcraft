//! Copied keyframes as After Effects keyframe text, for the system clipboard (#1908). Other
//! applications read this tab-separated format (AE2Blend brings it into Blender). Like After
//! Effects it covers one layer: its Transform properties, keyframed ones as one row per key
//! (frame number in comp time, then the value) and static ones as a single row with no frame.
//! Easing and tangents aren't part of the format; EffectCraft's own paste keeps them.

use effectcraft_project::{Comp, LayerId, Project, Uid};

use crate::KeyRef;

/// Transform properties in After Effects' order, with the unit each column is labelled with.
const CHANNELS: &[(&str, &str)] = &[
    ("poi", "pixels"),
    ("anchor", "pixels"),
    ("position", "pixels"),
    ("scale", "percent"),
    ("orientation", "degrees"),
    ("rotationX", "degrees"),
    ("rotationY", "degrees"),
    ("rotation", "degrees"),
    ("opacity", "percent"),
];

/// The text for copied keys (layer times) and unanimated properties. `None` unless they all come
/// from one layer and at least one is a Transform property.
pub(crate) fn text(project: &Project, comp: &Comp, keys: &[KeyRef], statics: &[(LayerId, Uid)]) -> Option<String> {
    let lid = keys.first().map(|k| k.layer).or_else(|| statics.first().map(|s| s.0))?;
    if keys.iter().any(|k| k.layer != lid) || statics.iter().any(|s| s.0 != lid) {
        return None;
    }
    let layer = comp.layer(lid)?;
    let fps = comp.frame_rate.as_f64();
    let mut blocks = String::new();
    for (id, unit) in CHANNELS {
        let Some(pr) = layer.props.prop(&format!("transform/{id}")) else { continue };
        if pr.three_d_only && !layer.is_3d() {
            continue;
        }
        let mut times: Vec<_> = keys.iter().filter(|k| k.prop == pr.uid).map(|k| k.time).collect();
        times.sort();
        times.dedup();
        let rows: Vec<(String, Vec<f64>)> = if !times.is_empty() {
            let frame = |t| num(layer.comp_time(t).seconds() * fps);
            times.into_iter().filter_map(|t| Some((frame(t), pr.keys.iter().find(|k| k.time == t)?.value.components()))).collect()
        } else if statics.iter().any(|s| s.1 == pr.uid) {
            vec![(String::new(), pr.value.components())]
        } else {
            continue;
        };
        let Some(dims) = rows.first().map(|(_, v)| v.len()).filter(|n| *n > 0) else { continue };
        let columns = match dims {
            1 => vec![unit.to_string()],
            _ => ["X", "Y", "Z"].iter().take(dims).map(|axis| format!("{axis} {unit}")).collect(),
        };
        // 3D layers show Rotation as Z Rotation.
        let name = if *id == "rotation" && layer.is_3d() { "Z Rotation" } else { pr.name.as_str() };
        blocks.push_str(&format!("Transform\t{name}\n\tFrame\t{}\n", columns.join("\t")));
        for (frame, values) in rows {
            let values: Vec<String> = values.into_iter().map(num).collect();
            blocks.push_str(&format!("\t{frame}\t{}\n", values.join("\t")));
        }
        blocks.push('\n');
    }
    if blocks.is_empty() {
        return None;
    }
    let (w, h) = match effectcraft_render::eval::source_size(project, layer) {
        (0, 0) => (comp.width, comp.height),
        size => size,
    };
    Some(format!(
        "Adobe After Effects 8.0 Keyframe Data\n\n\tUnits Per Second\t{}\n\tSource Width\t{w}\n\tSource Height\t{h}\n\tSource Pixel Aspect Ratio\t1\n\tComp Pixel Aspect Ratio\t{}\n\n{blocks}End of Keyframe Data\n",
        num(fps),
        num(comp.pixel_aspect)
    ))
}

/// A number with at most three decimals and no trailing zeros (`23.976`, `960`, `-0.5`).
fn num(v: f64) -> String {
    let s = format!("{:.3}", if v.is_finite() { v } else { 0.0 });
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s == "-0" { "0".into() } else { s.into() }
}
