//! Reproduction of issue #2436's report: an Environment light added through the real command
//! path (what the Light Settings dialog runs) must visibly light the scene in the viewer.

use effectcraft_engine::Session;
use effectcraft_engine::project::Renderer;
use effectcraft_ui_egui::EffectcraftApp;
use egui_kittest::Harness;
use serde_json::json;

fn app() -> EffectcraftApp {
    let mut s = Session::default();
    s.execute("file.openDemoProject", json!({})).unwrap();
    s.execute("comp.open", json!({"comp": "3D Showcase"})).unwrap();
    s.execute("time.set", json!({"time": 5.0})).unwrap();
    EffectcraftApp::new(s)
}

/// Step the UI until background frame renders have landed.
fn settle(h: &mut Harness<'_, EffectcraftApp>) {
    for _ in 0..600 {
        h.step();
        if h.state().frames.inflight() == 0 && h.state().frames.last_ms.lock().map(|v| *v > 0.0).unwrap_or(false) {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    for _ in 0..4 {
        h.step();
    }
}

fn changed_pixels(before: &[egui::Color32], after: &[egui::Color32]) -> usize {
    before.iter().zip(after).filter(|(a, b)| a != b).count()
}

/// An Environment light (red, full intensity, no source: the flat default environment) added to
/// a Classic 3D comp switches it to Advanced 3D and visibly changes the viewer's pixels.
#[test]
fn environment_light_lights_the_scene() {
    let mut h = Harness::builder().with_size(egui::vec2(1600.0, 1000.0)).build_eframe(|_| app());
    h.state_mut().session.execute("comp.settings", json!({"renderer": "classic3D"})).unwrap();
    h.run_steps(3);
    settle(&mut h);
    let before = h.state_mut().viewer_pixels().expect("frame before adding the light").pixels.clone();

    h.state_mut().session.execute("layer.newLight", json!({"kind": "Environment", "color": [1.0, 0.0, 0.0], "intensity": 100.0})).unwrap();

    assert_eq!(h.state().session.active_comp().unwrap().renderer, Renderer::Advanced3D, "the comp switched to Advanced 3D");
    settle(&mut h);
    let after = h.state_mut().viewer_pixels().expect("frame after adding the light").pixels.clone();
    assert_eq!(before.len(), after.len());
    let diff = changed_pixels(&before, &after);
    assert!(diff > 500, "a red Environment light at 100 % changed only {diff} pixels");
}

/// The literal action from the issue: default white Environment light, defaults everywhere —
/// even then the lighting model changes (Advanced 3D IBL instead of unlit/Classic), so pixels
/// must change somewhere in the comp frame.
#[test]
fn default_environment_light_changes_the_scene() {
    let mut h = Harness::builder().with_size(egui::vec2(1600.0, 1000.0)).build_eframe(|_| app());
    h.state_mut().session.execute("comp.settings", json!({"renderer": "classic3D"})).unwrap();
    h.run_steps(3);
    settle(&mut h);
    let before = h.state_mut().viewer_pixels().expect("frame before adding the light").pixels.clone();

    h.state_mut().session.execute("layer.newLight", json!({"kind": "Environment"})).unwrap();

    assert_eq!(h.state().session.active_comp().unwrap().renderer, Renderer::Advanced3D);
    settle(&mut h);
    let after = h.state_mut().viewer_pixels().expect("frame after adding the light").pixels.clone();
    assert_eq!(before.len(), after.len());
    let diff = changed_pixels(&before, &after);
    assert!(diff > 500, "a default white Environment light changed only {diff} pixels");
}
