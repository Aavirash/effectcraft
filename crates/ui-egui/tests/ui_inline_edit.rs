//! Inline text fields (a name, the current time) open with their whole text selected so typing
//! replaces it, commit on Enter and cancel on Escape (#170).

use effectcraft_engine::Session;
use effectcraft_ui_egui::EffectcraftApp;
use egui::{Event, Key, Modifiers, PointerButton, Pos2, pos2};
use egui_kittest::Harness;
use serde_json::json;

fn harness() -> (Harness<'static, EffectcraftApp>, u64) {
    let mut s = Session::default();
    s.execute("comp.new", json!({"name": "Main", "width": 320, "height": 180, "duration": 4, "fps": 24})).unwrap();
    let comp = s.active_comp_id().unwrap().0;
    let mut h = Harness::builder().with_size(egui::vec2(1600.0, 1000.0)).build_eframe(|_| EffectcraftApp::new(s));
    h.run_steps(4);
    (h, comp)
}

fn center(h: &Harness<'_, EffectcraftApp>, id: &str) -> Pos2 {
    let e = h.state().auto.find(id).unwrap_or_else(|| panic!("no {id}"));
    pos2(e.rect[0] + e.rect[2] / 2.0, e.rect[1] + e.rect[3] / 2.0)
}

fn click(h: &mut Harness<'_, EffectcraftApp>, p: Pos2, times: usize) {
    h.input_mut().events.push(Event::PointerMoved(p));
    h.step();
    // All in one frame (the harness' frames are longer apart than a double-click).
    for _ in 0..times {
        h.input_mut().events.push(Event::PointerButton { pos: p, button: PointerButton::Primary, pressed: true, modifiers: Modifiers::NONE });
        h.input_mut().events.push(Event::PointerButton { pos: p, button: PointerButton::Primary, pressed: false, modifiers: Modifiers::NONE });
    }
    h.run_steps(3);
}

fn key(h: &mut Harness<'_, EffectcraftApp>, key: Key) {
    h.input_mut().events.push(Event::Key { key, physical_key: None, pressed: true, repeat: false, modifiers: Modifiers::NONE });
    h.input_mut().events.push(Event::Key { key, physical_key: None, pressed: false, repeat: false, modifiers: Modifiers::NONE });
    h.run_steps(2);
}

fn type_text(h: &mut Harness<'_, EffectcraftApp>, text: &str) {
    assert!(h.ctx.egui_wants_keyboard_input(), "the field has the keyboard");
    h.input_mut().events.push(Event::Text(text.into()));
    h.step();
}

#[test]
fn project_rename_replaces_the_name() {
    let (mut h, comp) = harness();
    let name = |h: &Harness<'_, EffectcraftApp>| h.state().session.project.items.values().find(|i| i.id.0 == comp).unwrap().name.clone();
    let p = center(&h, &format!("project.item.{comp}.name"));
    click(&mut h, p, 1);
    key(&mut h, Key::Enter);
    type_text(&mut h, "Hero");
    key(&mut h, Key::Enter);
    assert_eq!(name(&h), "Hero");
    assert!(!h.ctx.egui_wants_keyboard_input(), "Enter releases the keyboard");
}

#[test]
fn timecode_field_replaces_commits_and_cancels() {
    let (mut h, _) = harness();
    let frame = |h: &Harness<'_, EffectcraftApp>| h.state().session.active_comp().unwrap().frame_rate.frame_at(h.state().session.time());
    let p = center(&h, "timeline.timecode");
    click(&mut h, p, 2);
    type_text(&mut h, "0:00:01:00");
    key(&mut h, Key::Enter);
    assert_eq!(frame(&h), 24, "Enter commits the typed time");
    assert!(!h.ctx.egui_wants_keyboard_input(), "Enter releases the keyboard");
    click(&mut h, p, 2);
    type_text(&mut h, "0:00:02:00");
    key(&mut h, Key::Escape);
    assert_eq!(frame(&h), 24, "Escape cancels");
    assert!(!h.ctx.egui_wants_keyboard_input(), "Escape releases the keyboard");
}

/// #284: dragging the expression pick whip while the expression is being edited puts the
/// reference at the cursor (After Effects), and the editing goes on after it; without an edit
/// in progress it replaces the expression.
#[test]
fn the_expression_pick_whip_inserts_at_the_cursor_while_editing() {
    let (mut h, _) = harness();
    let (layer, opacity, rotation, transform) = {
        let s = &mut h.state_mut().session;
        let l = s.execute("layer.newSolid", json!({"name": "Plate", "color": "#406080"})).unwrap()["layer"].as_u64().unwrap();
        s.execute("prop.setExpression", json!({"layer": l, "path": "transform/opacity", "expression": "50 + 0"})).unwrap();
        let tr = s.active_comp().unwrap().layer(effectcraft_engine::project::LayerId(l)).unwrap().props.sub("transform").unwrap().clone();
        (l, tr.get("opacity").unwrap().uid, tr.get("rotation").unwrap().uid, tr.uid)
    };
    h.state_mut().ui.timeline.open_layers.insert(layer);
    h.state_mut().ui.timeline.open_groups.insert(transform);
    h.run_steps(3);
    let expr = |h: &Harness<'_, EffectcraftApp>| {
        let comp = h.state().session.active_comp().unwrap();
        comp.layer(effectcraft_engine::project::LayerId(layer)).unwrap().props.find(opacity).unwrap().expr.as_ref().unwrap().text.clone()
    };
    let whip = |h: &mut Harness<'_, EffectcraftApp>| {
        let (from, to) = (center(h, &format!("timeline.prop.{opacity}.pickWhip")), center(h, &format!("timeline.prop.{rotation}.name")));
        h.input_mut().events.push(Event::PointerMoved(from));
        h.step();
        h.input_mut().events.push(Event::PointerButton { pos: from, button: PointerButton::Primary, pressed: true, modifiers: Modifiers::NONE });
        h.step();
        for k in 1..=8 {
            h.input_mut().events.push(Event::PointerMoved(from + (to - from) * (k as f32 / 8.0)));
            h.step();
        }
        h.input_mut().events.push(Event::PointerButton { pos: to, button: PointerButton::Primary, pressed: false, modifiers: Modifiers::NONE });
        h.run_steps(3);
    };
    // Editing: the cursor at the end after typing " * ".
    let p = center(&h, &format!("timeline.prop.{opacity}.expression"));
    click(&mut h, p, 1);
    key(&mut h, Key::End);
    type_text(&mut h, " * ");
    whip(&mut h);
    assert_eq!(expr(&h), "50 + 0 * transform.rotation");
    // The editor has the keyboard again, its cursor after the reference.
    type_text(&mut h, " + 1");
    // Ctrl/Cmd+Enter commits.
    h.input_mut().events.push(Event::ModifiersChanged(Modifiers::COMMAND));
    h.input_mut().events.push(Event::Key { key: Key::Enter, physical_key: None, pressed: true, repeat: false, modifiers: Modifiers::COMMAND });
    h.step();
    h.input_mut().events.push(Event::ModifiersChanged(Modifiers::NONE));
    h.run_steps(3);
    assert_eq!(expr(&h), "50 + 0 * transform.rotation + 1");
    // Not editing: the reference replaces the expression.
    whip(&mut h);
    assert_eq!(expr(&h), "transform.rotation");
}

/// A click at `p` with `modifiers` held.
fn click_with(h: &mut Harness<'_, EffectcraftApp>, p: Pos2, modifiers: Modifiers) {
    h.input_mut().events.push(Event::PointerMoved(p));
    h.input_mut().events.push(Event::ModifiersChanged(modifiers));
    h.step();
    h.input_mut().events.push(Event::PointerButton { pos: p, button: PointerButton::Primary, pressed: true, modifiers });
    h.input_mut().events.push(Event::PointerButton { pos: p, button: PointerButton::Primary, pressed: false, modifiers });
    h.step();
    h.input_mut().events.push(Event::ModifiersChanged(Modifiers::NONE));
    h.run_steps(3);
}

/// #370: Ctrl/Cmd-click on the current-time display (the Timeline's or the Composition
/// panel's) toggles the Time Display Style between Timecode and Frames, one undo step each, as
/// in After Effects; under the Timeline's display the other style shows.
#[test]
fn ctrl_click_on_the_time_display_toggles_timecode_and_frames() {
    use effectcraft_engine::project::TimeDisplayStyle;
    let (mut h, _) = harness();
    h.state_mut().session.execute("time.set", json!({"frame": 30})).unwrap();
    h.run_steps(2);
    let shown = |h: &Harness<'_, EffectcraftApp>, id: &str| h.state().auto.find(id).unwrap().label.clone();
    let style = |h: &Harness<'_, EffectcraftApp>| h.state().session.project.settings.time_display;
    let undo = h.state().session.history.undo.len();
    let at = center(&h, "timeline.timecode");
    click_with(&mut h, at, Modifiers::COMMAND);
    assert_eq!(style(&h), TimeDisplayStyle::Frames);
    assert_eq!(shown(&h, "timeline.timecode"), "00030");
    assert_eq!(h.state().session.history.undo.len(), undo + 1, "one undo step");
    assert!(!h.ctx.egui_wants_keyboard_input(), "no time entry opened");
    let at = center(&h, "viewer.timecode");
    click_with(&mut h, at, Modifiers::COMMAND);
    assert_eq!(style(&h), TimeDisplayStyle::Timecode);
    assert_eq!(shown(&h, "timeline.timecode"), "0:00:01:06");
    // A plain click changes nothing.
    click_with(&mut h, at, Modifiers::NONE);
    let at = center(&h, "timeline.timecode");
    click_with(&mut h, at, Modifiers::NONE);
    assert_eq!(style(&h), TimeDisplayStyle::Timecode);
    // Feet + Frames goes to Timecode.
    h.state_mut().session.execute("file.projectSettings", json!({"timeDisplay": "feet35"})).unwrap();
    h.run_steps(2);
    click_with(&mut h, at, Modifiers::COMMAND);
    assert_eq!(style(&h), TimeDisplayStyle::Timecode);
}
