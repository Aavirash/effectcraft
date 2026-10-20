//! Copied keyframes as After Effects keyframe text for other applications (#1908).

use effectcraft_project::{LayerId, Uid};
use effectcraft_time::Tick;
use serde_json::json;

use crate::{KeyRef, Session};

/// A 1920×1080 23.976 fps comp with a 200×100 solid.
fn session() -> (Session, LayerId) {
    let mut s = Session::default();
    s.execute("comp.new", json!({"name": "Main", "width": 1920, "height": 1080, "frameRate": 23.976, "duration": 10})).unwrap();
    let id = s.execute("layer.newSolid", json!({"color": "#ff0000", "width": 200, "height": 100})).unwrap()["layer"].as_u64().unwrap();
    (s, LayerId(id))
}

fn frame(s: &Session, n: i64) -> Tick {
    Tick(s.active_comp().unwrap().frame_rate.frame_duration().0 * n)
}

fn uid(s: &Session, lid: LayerId, path: &str) -> Uid {
    s.active_comp().unwrap().layer(lid).unwrap().props.prop(path).unwrap().uid
}

fn edit_layer(s: &mut Session, lid: LayerId, f: impl FnOnce(&mut effectcraft_project::Layer)) {
    let cid = s.active_comp_id().unwrap();
    f(std::sync::Arc::make_mut(&mut s.project).comp_mut(cid).unwrap().layer_mut(lid).unwrap());
}

fn add_key(s: &mut Session, lid: LayerId, path: &str, time: Tick, value: serde_json::Value) {
    s.execute("prop.addKey", json!({"layer": lid.0, "path": path, "time": time.seconds(), "value": value})).unwrap();
}

#[test]
fn copied_keys_are_after_effects_text_in_comp_frames() {
    let (mut s, l) = session();
    // The layer starts 10 frames in: its keys at layer frames 2, 5 and 9 are comp frames 12, 15 and 19.
    let start = frame(&s, 10);
    edit_layer(&mut s, l, |l| l.start_time = start);
    for (n, v) in [(2, [960.0, 540.0]), (5, [1000.5, 520.25]), (9, [1100.0, 500.0])] {
        let t = frame(&s, n);
        add_key(&mut s, l, "transform/position", t, json!(v));
    }
    let t = frame(&s, 5);
    add_key(&mut s, l, "transform/rotation", t, json!(-45));
    let (position, rotation) = (uid(&s, l, "transform/position"), uid(&s, l, "transform/rotation"));
    // A subset: two of the three Position keys and the Rotation key.
    s.state.selected_keys = [(position, 2), (position, 9), (rotation, 5)].map(|(prop, n)| KeyRef { layer: l, prop, time: frame(&s, n) }).to_vec();
    s.execute("keys.copy", json!({})).unwrap();
    assert_eq!(
        s.state.keyframe_text.as_deref(),
        Some(
            "Adobe After Effects 8.0 Keyframe Data\n\n\tUnits Per Second\t23.976\n\tSource Width\t200\n\tSource Height\t100\n\tSource Pixel Aspect Ratio\t1\n\tComp Pixel Aspect Ratio\t1\n\n\
             Transform\tPosition\n\tFrame\tX pixels\tY pixels\tZ pixels\n\t12\t960\t540\t0\n\t19\t1100\t500\t0\n\n\
             Transform\tRotation\n\tFrame\tdegrees\n\t15\t-45\n\n\
             End of Keyframe Data\n"
        )
    );
    // The internal clipboard still has the keys themselves, relative to the earliest.
    let times: Vec<Vec<Tick>> = s.state.key_clipboard.iter().map(|c| c.keys.iter().map(|k| k.time).collect()).collect();
    assert_eq!(times, [vec![Tick::ZERO, frame(&s, 7)], vec![frame(&s, 3)]]);
}

#[test]
fn a_stretched_layer_writes_the_frames_its_keys_show_at() {
    let (mut s, l) = session();
    edit_layer(&mut s, l, |l| l.stretch = 200.0);
    let t = frame(&s, 6);
    add_key(&mut s, l, "transform/opacity", Tick::ZERO, json!(0));
    add_key(&mut s, l, "transform/opacity", t, json!(100));
    s.execute("prop.select", json!({"layer": l.0, "path": "transform/opacity"})).unwrap();
    s.execute("edit.copy", json!({})).unwrap();
    let text = s.state.keyframe_text.clone().unwrap();
    assert!(text.contains("Transform\tOpacity\n\tFrame\tpercent\n\t0\t0\n\t12\t100\n\n"), "{text}");
}

#[test]
fn copied_static_values_have_no_frame() {
    let (mut s, l) = session();
    s.execute("prop.set", json!({"layer": l.0, "path": "transform/position", "value": [100, 200]})).unwrap();
    s.execute("prop.select", json!({"layer": l.0, "path": "transform/position"})).unwrap();
    s.execute("edit.copy", json!({})).unwrap();
    let text = s.state.keyframe_text.clone().unwrap();
    assert!(text.contains("Transform\tPosition\n\tFrame\tX pixels\tY pixels\tZ pixels\n\t\t100\t200\t0\n\n"), "{text}");
    assert_eq!(s.state.value_clipboard.len(), 1, "the internal clipboard keeps the value");
}

#[test]
fn a_copied_transform_group_lists_what_the_layer_shows() {
    let (mut s, l) = session();
    let t = frame(&s, 24);
    add_key(&mut s, l, "transform/scale", Tick::ZERO, json!([50, 50, 100]));
    add_key(&mut s, l, "transform/scale", t, json!([150, 150, 100]));
    let transform = s.active_comp().unwrap().layer(l).unwrap().props.sub("transform").unwrap().uid;
    s.execute("prop.select", json!({"layer": l.0, "prop": transform, "selectKeys": false})).unwrap();
    s.execute("edit.copy", json!({})).unwrap();
    let text = s.state.keyframe_text.clone().unwrap();
    let names: Vec<&str> = text.lines().filter_map(|line| line.strip_prefix("Transform\t")).collect();
    assert_eq!(names, ["Anchor Point", "Position", "Scale", "Rotation", "Opacity"], "a 2D layer has no Orientation or X / Y Rotation");
    assert!(text.contains("\tFrame\tX percent\tY percent\tZ percent\n\t0\t50\t50\t100\n\t24\t150\t150\t100\n"), "{text}");
    // 3D: Orientation, X / Y Rotation, and Rotation is Z Rotation.
    edit_layer(&mut s, l, |l| l.switches.three_d = true);
    s.execute("prop.select", json!({"layer": l.0, "prop": transform, "selectKeys": false})).unwrap();
    s.execute("edit.copy", json!({})).unwrap();
    let text = s.state.keyframe_text.clone().unwrap();
    let names: Vec<&str> = text.lines().filter_map(|line| line.strip_prefix("Transform\t")).collect();
    assert_eq!(names, ["Anchor Point", "Position", "Scale", "Orientation", "X Rotation", "Y Rotation", "Z Rotation", "Opacity"]);
    assert!(text.contains("Transform\tOrientation\n\tFrame\tX degrees\tY degrees\tZ degrees\n\t\t0\t0\t0\n"), "{text}");
}

#[test]
fn keys_from_two_layers_have_no_text() {
    let (mut s, a) = session();
    let b = LayerId(s.execute("layer.newSolid", json!({"color": "#00ff00", "width": 200, "height": 100})).unwrap()["layer"].as_u64().unwrap());
    for l in [a, b] {
        add_key(&mut s, l, "transform/opacity", Tick::ZERO, json!(50));
    }
    s.state.selected_keys = [a, b].map(|l| KeyRef { layer: l, prop: uid(&s, l, "transform/opacity"), time: Tick::ZERO }).to_vec();
    s.execute("keys.copy", json!({})).unwrap();
    assert_eq!(s.state.keyframe_text, None, "After Effects copies one layer's keys as text");
    assert_eq!(s.state.key_clipboard.len(), 2, "EffectCraft's own clipboard has both");
    // A copied layer leaves no text behind.
    s.state.selected_keys = vec![KeyRef { layer: a, prop: uid(&s, a, "transform/opacity"), time: Tick::ZERO }];
    s.execute("keys.copy", json!({})).unwrap();
    assert!(s.state.keyframe_text.is_some());
    s.state.selected_keys.clear();
    s.execute("edit.copy", json!({"layers": [a.0]})).unwrap();
    assert_eq!(s.state.keyframe_text, None);
}
