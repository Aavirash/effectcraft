//! Scale Composition (`comp.scale`, #2208): the comp and everything in it grow or shrink by one
//! factor, so the frame looks the same at the new size.

use effectcraft_project::{Layer, LayerId};
use effectcraft_time::Tick;
use serde_json::json;

use crate::Session;

fn comp(w: u32, h: u32) -> Session {
    let mut s = Session::default();
    s.execute("comp.new", json!({"name": "Main", "width": w, "height": h, "frameRate": 10, "duration": 2})).unwrap();
    s
}

fn solid(s: &mut Session, color: &str, w: u32, h: u32) -> u64 {
    s.execute("layer.newSolid", json!({"color": color, "width": w, "height": h})).unwrap()["layer"].as_u64().unwrap()
}

fn layer(s: &Session, id: u64) -> Layer {
    s.active_comp().unwrap().layer(LayerId(id)).unwrap().clone()
}

fn v3(s: &Session, id: u64, path: &str, t: f64) -> [f64; 3] {
    layer(s, id).props.prop(path).unwrap().value_at(Tick::from_seconds_f64(t)).as_vec3()
}

/// Opaque pixels of `color_channel` (0 = red, 1 = green) at `t`: (x0, y0, x1, y1, count).
fn bounds(s: &Session, t: f64, channel: usize) -> (i64, i64, i64, i64, u32) {
    let img = s.render(s.active_comp_id().unwrap(), Tick::from_seconds_f64(t), Default::default());
    let (mut x0, mut y0, mut x1, mut y1, mut n) = (i64::MAX, i64::MAX, i64::MIN, i64::MIN, 0);
    for y in 0..img.height as i64 {
        for x in 0..img.width as i64 {
            let p = img.get(x, y);
            if p[3] > 0.5 && p[channel] > 0.5 {
                n += 1;
                (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x), y1.max(y));
            }
        }
    }
    (x0, y0, x1, y1, n)
}

/// Twice as large on every side, from a pixel-exact frame.
fn doubled(b: (i64, i64, i64, i64, u32)) -> (i64, i64, i64, i64, u32) {
    (b.0 * 2, b.1 * 2, b.2 * 2 + 1, b.3 * 2 + 1, b.4 * 4)
}

#[test]
fn scale_comp_doubles_the_frame_with_keys_and_parented_layers() {
    let mut s = comp(200, 100);
    // A red solid moving right (two Position keys), and a green solid parented to a null.
    let red = solid(&mut s, "#ff0000", 20, 10);
    s.execute("prop.addKey", json!({"layer": red, "path": "transform/position", "time": 0, "value": [30, 20]})).unwrap();
    s.execute("prop.addKey", json!({"layer": red, "path": "transform/position", "time": 1, "value": [70, 20]})).unwrap();
    let green = solid(&mut s, "#00ff00", 10, 10);
    let rig = s.execute("layer.newNull", json!({})).unwrap()["layer"].as_u64().unwrap();
    s.execute("prop.set", json!({"layer": rig, "path": "transform/position", "value": [150, 60]})).unwrap();
    s.execute("layer.setParent", json!({"layers": [green], "parent": rig})).unwrap();
    s.execute("prop.set", json!({"layer": green, "path": "transform/position", "value": [10, 15]})).unwrap();
    let before = s.project.clone();
    let (red0, red1, green0) = (bounds(&s, 0.0, 0), bounds(&s, 1.0, 0), bounds(&s, 0.0, 1));
    assert_eq!(red0, (20, 15, 39, 24, 200));

    let r = s.execute("comp.scale", json!({"factor": 2})).unwrap();
    assert_eq!((r["width"].as_u64(), r["height"].as_u64()), (Some(400), Some(200)));
    let c = s.active_comp().unwrap();
    assert_eq!((c.width, c.height), (400, 200));
    // Every key moves, not only the value at the current time.
    assert_eq!(bounds(&s, 0.0, 0), doubled(red0));
    assert_eq!(bounds(&s, 1.0, 0), doubled(red1));
    assert_eq!(v3(&s, red, "transform/scale", 1.0), [200.0; 3]);
    // The child keeps its own transform: its scaled parent carries it.
    assert_eq!(bounds(&s, 0.0, 1), doubled(green0));
    assert_eq!(v3(&s, green, "transform/position", 0.0)[..2], [10.0, 15.0]);
    assert_eq!(v3(&s, green, "transform/scale", 0.0), [100.0; 3]);

    // One undo step.
    s.execute("edit.undo", json!({})).unwrap();
    assert!(s.project == before, "undo restores the comp and every layer");
}

#[test]
fn scale_comp_to_a_width_keeps_the_aspect_and_scales_cameras_3d_and_guides() {
    let mut s = comp(640, 360);
    let card = solid(&mut s, "#ff0000", 100, 100);
    s.execute("layer.setSwitch", json!({"layers": [card], "switch": "threeD", "value": true})).unwrap();
    s.execute("prop.set", json!({"layer": card, "path": "transform/position", "value": [100, 50, 200]})).unwrap();
    let cam = s.execute("layer.newCamera", json!({})).unwrap()["layer"].as_u64().unwrap();
    s.execute("view.addGuide", json!({"orientation": "vertical", "position": 160})).unwrap();
    s.execute("view.addGuide", json!({"orientation": "horizontal", "position": 25, "percent": true})).unwrap();
    let zoom = |s: &Session| layer(s, cam).props.prop("cameraOptions/zoom").unwrap().value.as_f64();
    let (zoom0, cam0, poi0) = (zoom(&s), v3(&s, cam, "transform/position", 0.0), v3(&s, cam, "transform/poi", 0.0));
    let seen0 = bounds(&s, 0.0, 0);

    let r = s.execute("comp.scale", json!({"width": 960})).unwrap();
    assert_eq!((r["width"].as_u64(), r["height"].as_u64(), r["factor"].as_f64()), (Some(960), Some(540), Some(1.5)));
    assert_eq!(v3(&s, card, "transform/position", 0.0), [150.0, 75.0, 300.0]);
    assert_eq!(v3(&s, card, "transform/scale", 0.0), [150.0; 3]);
    assert!((zoom(&s) - zoom0 * 1.5).abs() < 1e-9, "the camera sees the same frame");
    assert_eq!(v3(&s, cam, "transform/position", 0.0), cam0.map(|v| v * 1.5));
    assert_eq!(v3(&s, cam, "transform/poi", 0.0), poi0.map(|v| v * 1.5));
    // Through the camera the card covers the same part of the frame.
    let seen = bounds(&s, 0.0, 0);
    let near = |a: i64, b: i64| (a as f64 - b as f64 * 1.5).abs() <= 2.0;
    assert!(near(seen.0, seen0.0) && near(seen.1, seen0.1) && near(seen.2, seen0.2) && near(seen.3, seen0.3), "{seen0:?} → {seen:?}");
    let c = s.active_comp().unwrap();
    assert_eq!(c.guides[0].px(c.width, c.height), 240.0);
    assert_eq!(c.guides[1].position, 25.0, "a percentage guide keeps its percent");

    // A height alone works the same way.
    let r = s.execute("comp.scale", json!({"height": 270})).unwrap();
    assert_eq!((r["width"].as_u64(), r["height"].as_u64()), (Some(480), Some(270)));
}

#[test]
fn scale_comp_refuses_a_bad_factor_and_changes_nothing() {
    let mut s = comp(640, 360);
    solid(&mut s, "#ff0000", 100, 100);
    let before = s.project.clone();
    for p in [json!({}), json!({"factor": 0}), json!({"factor": -2}), json!({"width": 0}), json!({"factor": 100}), json!({"factor": 0.001})] {
        assert!(s.execute("comp.scale", p.clone()).is_err(), "{p} is refused");
        assert!(s.project == before, "{p} leaves the project alone");
    }
}
