//! `keys.delete` reports the keys it removed, not the keys it was asked to remove.

use effectcraft_project::LayerId;
use serde_json::json;

use crate::{KeyRef, Session};

fn setup() -> (Session, u64, u64) {
    let mut s = Session::default();
    s.execute_checked("comp.new", json!({"width": 64, "height": 64, "frameRate": 30, "duration": 4})).unwrap();
    let layer = s.execute_checked("layer.newSolid", json!({"name": "Box", "color": "#fff"})).unwrap()["layer"].as_u64().unwrap();
    for t in 0..3 {
        s.execute_checked("prop.addKey", json!({"layer": layer, "path": "transform/opacity", "time": t, "value": t * 25})).unwrap();
    }
    let prop = s.active_comp().unwrap().layer(LayerId(layer)).unwrap().props.prop("transform/opacity").unwrap().uid;
    (s, layer, prop)
}

fn keys_of(s: &Session, layer: u64, prop: u64) -> Vec<KeyRef> {
    let keys = &s.active_comp().unwrap().layer(LayerId(layer)).unwrap().props.find(prop).unwrap().keys;
    keys.iter().map(|k| KeyRef { layer: LayerId(layer), prop, time: k.time }).collect()
}

#[test]
fn delete_keys_counts_only_the_keys_it_removed() {
    let (mut s, layer, prop) = setup();
    let keys = keys_of(&s, layer, prop);
    assert_eq!(keys.len(), 3);
    // Two real keys, one of them listed twice, and a key of a property that does not exist.
    let missing = KeyRef { layer: LayerId(layer), prop: u64::MAX, time: keys[1].time };
    s.state.selected_keys = vec![keys[0], keys[2], keys[0], missing];
    let removed = s.execute_checked("keys.delete", json!({})).unwrap();
    assert_eq!(removed, json!(2));
    assert_eq!(keys_of(&s, layer, prop), vec![keys[1]]);
}

#[test]
fn delete_keys_reports_zero_when_no_selected_key_exists() {
    let (mut s, layer, prop) = setup();
    let keys = keys_of(&s, layer, prop);
    s.state.selected_keys = vec![KeyRef { layer: LayerId(layer), prop: u64::MAX, time: keys[0].time }];
    let removed = s.execute_checked("keys.delete", json!({})).unwrap();
    assert_eq!(removed, json!(0));
    assert_eq!(keys_of(&s, layer, prop), keys);
}

/// A layer locked after its keys were selected keeps them: Delete and Edit Value leave them alone.
#[test]
fn a_locked_layers_keys_cannot_be_deleted_or_edited() {
    let (mut s, layer, prop) = setup();
    let keys = keys_of(&s, layer, prop);
    s.state.selected_keys = keys.clone();
    s.execute_checked("layer.setSwitch", json!({"layers": [layer], "switch": "locked", "value": true})).unwrap();
    let removed = s.execute_checked("keys.delete", json!({})).unwrap();
    assert_eq!(removed, json!(0));
    assert_eq!(keys_of(&s, layer, prop), keys);
    assert_eq!(s.state.selected_keys, keys, "the selection stays");
    let e = s.execute("keys.set", json!({"layer": layer, "prop": prop, "time": 0, "value": 90})).unwrap_err().to_string();
    assert!(e.contains("locked"), "{e}");
    let value = s.active_comp().unwrap().layer(LayerId(layer)).unwrap().props.find(prop).unwrap().keys[0].value.as_f64();
    assert_eq!(value, 0.0);
    // Unlocked again, both work.
    s.execute_checked("layer.setSwitch", json!({"layers": [layer], "switch": "locked", "value": false})).unwrap();
    s.execute_checked("keys.set", json!({"layer": layer, "prop": prop, "time": 0, "value": 90})).unwrap();
    assert_eq!(s.execute_checked("keys.delete", json!({})).unwrap(), json!(3));
}

/// Easy Ease, Toggle Hold and Time-Reverse leave a locked layer's keys as they are and keep them
/// selected; unlocked, they change them.
#[test]
fn selection_based_key_edits_skip_a_locked_layer() {
    let (mut s, layer, prop) = setup();
    let keys = keys_of(&s, layer, prop);
    let stored = |s: &Session| s.active_comp().unwrap().layer(LayerId(layer)).unwrap().props.find(prop).unwrap().keys.clone();
    let before = stored(&s);
    s.state.selected_keys = keys.clone();
    s.execute_checked("layer.setSwitch", json!({"layers": [layer], "switch": "locked", "value": true})).unwrap();
    for command in ["keys.easyEase", "keys.toggleHold", "keys.timeReverse"] {
        s.execute_checked(command, json!({})).unwrap();
        assert_eq!(stored(&s), before, "{command} changed a locked layer's keys");
        assert_eq!(s.state.selected_keys, keys, "{command} dropped the selection of a locked layer's keys");
    }
    s.execute_checked("layer.setSwitch", json!({"layers": [layer], "switch": "locked", "value": false})).unwrap();
    s.execute_checked("keys.easyEase", json!({})).unwrap();
    assert_ne!(stored(&s), before, "unlocked, Easy Ease changes the keys");
}
