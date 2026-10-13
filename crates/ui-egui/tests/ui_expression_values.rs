//! #1306 (from #1072): a property driven by an expression shows its value in red, the expression
//! colour, in the Timeline, Effect Controls and the Properties panel, as in After Effects: scalars,
//! every part of a vector (with its separator) and both parts of an angle, hovered too, in dark and
//! light themes. A disabled, removed or empty expression shows the ordinary colour again. The
//! tests read the colour the value text is painted with.

use effectcraft_engine::Session;
use effectcraft_engine::project::{LayerId, PropGroup};
use effectcraft_ui_egui::EffectcraftApp;
use effectcraft_ui_egui::dock::PanelKind;
use effectcraft_ui_egui::theme::ThemeKind;
use egui::epaint::Shape;
use egui::{Color32, Event, Pos2, Rect, pos2, vec2};
use egui_kittest::Harness;
use serde_json::json;

const PANELS: [&str; 3] = ["timeline.prop", "effectControls.prop", "properties.prop"];

/// A comp with one solid: (layer, transform Position / Rotation / Opacity / Scale uids, the
/// Transform effect's Position / Rotation / Opacity uids).
struct Ids {
    layer: u64,
    layer_props: [u64; 3],
    scale: u64,
    effect_props: [u64; 3],
}

fn setup() -> (Harness<'static, EffectcraftApp>, Ids) {
    let mut s = Session::default();
    s.execute("comp.new", json!({"name": "E", "width": 320, "height": 180, "duration": 4})).unwrap();
    let l = s.execute("layer.newSolid", json!({"color": "#406080"})).unwrap()["layer"].as_u64().unwrap();
    let fx = s.execute("effect.apply", json!({"layers": [l], "effect": "Transform"})).unwrap()["effects"][0].as_u64().unwrap();
    let layer = s.active_comp().unwrap().layer(LayerId(l)).unwrap().clone();
    let uid = |path: &str| layer.props.prop(path).unwrap_or_else(|| panic!("no {path}")).uid;
    let g: &PropGroup = layer.props.find_group(fx).unwrap();
    let fuid = |m: &str| g.get(m).unwrap_or_else(|| panic!("no effect {m}")).uid;
    let ids = Ids {
        layer: l,
        layer_props: [uid("transform/position"), uid("transform/rotation"), uid("transform/opacity")],
        scale: uid("transform/scale"),
        effect_props: [fuid("position"), fuid("rotation"), fuid("opacity")],
    };
    let transform = layer.props.sub("transform").unwrap().uid;
    s.state.selected_layers = vec![LayerId(l)];
    let mut app = EffectcraftApp::new(s);
    app.show_panel(PanelKind::EffectControls);
    app.ui.timeline.open_layers.insert(l);
    app.ui.timeline.open_groups.insert(transform);
    let mut h = Harness::builder().with_size(vec2(1600.0, 1000.0)).build_eframe(|_| app);
    h.run_steps(4);
    (h, ids)
}

fn set_expression(h: &mut Harness<'_, EffectcraftApp>, ids: &Ids, prop: u64, p: serde_json::Value) {
    let mut p = p;
    p["layer"] = json!(ids.layer);
    p["prop"] = json!(prop);
    h.state_mut().session.execute("prop.setExpression", p).unwrap();
    h.run_steps(3);
}

/// The value's area in `panel`: its value, revolutions and vector components together (so a
/// vector's separator and an angle's sign are inside).
fn value_rect(h: &Harness<'_, EffectcraftApp>, panel: &str, uid: u64) -> Option<Rect> {
    let ids = [
        format!("{panel}.{uid}.value"),
        format!("{panel}.{uid}.revolutions"),
        format!("{panel}.{uid}.value.0"),
        format!("{panel}.{uid}.value.1"),
        format!("{panel}.{uid}.value.2"),
    ];
    ids.iter()
        .filter_map(|id| h.state().auto.find(id))
        .map(|e| Rect::from_min_size(pos2(e.rect[0], e.rect[1]), vec2(e.rect[2], e.rect[3])))
        .reduce(|a, b| a.union(b))
}

/// The colours of the text painted inside `area` in the last frame.
fn text_colors(h: &Harness<'_, EffectcraftApp>, area: Rect) -> Vec<Color32> {
    fn walk(s: &Shape, area: Rect, out: &mut Vec<Color32>) {
        match s {
            Shape::Vec(v) => v.iter().for_each(|s| walk(s, area, out)),
            Shape::Text(t) if area.expand(1.0).contains_rect(t.visual_bounding_rect()) && !t.galley.text().trim().is_empty() => {
                out.push(t.override_text_color.unwrap_or(t.fallback_color));
            }
            _ => {}
        }
    }
    let mut out = vec![];
    for c in &h.output().shapes {
        walk(&c.shape, area, &mut out);
    }
    out
}

/// The colours `uid`'s value is painted with in every panel (each panel must show it).
fn value_colors(h: &Harness<'_, EffectcraftApp>, uid: u64) -> Vec<(&'static str, Vec<Color32>)> {
    PANELS
        .iter()
        .filter_map(|panel| {
            let area = value_rect(h, panel, uid)?;
            Some((*panel, text_colors(h, area)))
        })
        .collect()
}

fn assert_all(h: &Harness<'_, EffectcraftApp>, uids: &[u64], want: Color32, what: &str) {
    for &uid in uids {
        let shown = value_colors(h, uid);
        assert!(!shown.is_empty(), "{what}: @{uid} is shown in a panel");
        for (panel, colors) in shown {
            assert!(!colors.is_empty(), "{what}: @{uid} in {panel} paints its value");
            assert!(colors.iter().all(|c| *c == want), "{what}: @{uid} in {panel}: {colors:?}, want {want:?}");
        }
    }
}

/// Expression-driven values are red in every panel and both themes; other values stay blue.
#[test]
fn expression_driven_values_are_red_in_every_panel() {
    let (mut h, ids) = setup();
    for p in ids.layer_props.iter().chain(&ids.effect_props) {
        set_expression(&mut h, &ids, *p, json!({"expression": "value"}));
    }
    // Every value is shown somewhere: the layer's in the Timeline and Properties, the effect's in
    // Effect Controls.
    for p in ids.layer_props {
        assert!(value_rect(&h, "timeline.prop", p).is_some() && value_rect(&h, "properties.prop", p).is_some(), "@{p} in the Timeline and Properties");
    }
    for p in ids.effect_props {
        assert!(value_rect(&h, "effectControls.prop", p).is_some(), "@{p} in Effect Controls");
    }
    for theme in [ThemeKind::Dark, ThemeKind::Light] {
        let ctx = h.ctx.clone();
        h.state_mut().set_theme(&ctx, theme);
        h.run_steps(3);
        let t = h.state().tokens;
        assert_ne!(t.expr_text, t.hot_text);
        let all: Vec<u64> = ids.layer_props.iter().chain(&ids.effect_props).copied().collect();
        assert_all(&h, &all, t.expr_text, &format!("{theme:?} expression"));
        assert_all(&h, &[ids.scale], t.hot_text, &format!("{theme:?} no expression"));
    }
}

/// Disabled, removed or empty: the ordinary colour; enabled again: red.
#[test]
fn disabled_removed_and_empty_expressions_show_the_ordinary_colour() {
    let (mut h, ids) = setup();
    let [position, rotation, opacity] = ids.layer_props;
    for p in ids.layer_props {
        set_expression(&mut h, &ids, p, json!({"expression": "value"}));
    }
    let t = h.state().tokens;
    assert_all(&h, &ids.layer_props, t.expr_text, "enabled");
    set_expression(&mut h, &ids, position, json!({"enabled": false}));
    set_expression(&mut h, &ids, rotation, json!({"expression": ""}));
    set_expression(&mut h, &ids, opacity, json!({"expression": "   "}));
    assert_all(&h, &ids.layer_props, t.hot_text, "disabled, removed, empty");
    set_expression(&mut h, &ids, position, json!({"enabled": true}));
    assert_all(&h, &[position], t.expr_text, "enabled again");
}

fn hover(h: &mut Harness<'_, EffectcraftApp>, at: Pos2) {
    h.input_mut().events.push(Event::PointerMoved(at));
    h.run_steps(2);
}

/// Hovering a value (which turns an ordinary value's colour to the hover accent) keeps an
/// expression-driven one red.
#[test]
fn a_hovered_expression_value_stays_red() {
    let (mut h, ids) = setup();
    let [_, _, opacity] = ids.layer_props;
    set_expression(&mut h, &ids, opacity, json!({"expression": "value"}));
    let t = h.state().tokens;
    for panel in ["timeline.prop", "properties.prop"] {
        let area = value_rect(&h, panel, opacity).expect("the value");
        hover(&mut h, area.center());
        let colors = text_colors(&h, area);
        assert!(!colors.is_empty() && colors.iter().all(|c| *c == t.expr_text), "{panel} hovered: {colors:?}");
    }
    // An ordinary value does take the hover accent there.
    let area = value_rect(&h, "timeline.prop", ids.scale).expect("Scale");
    hover(&mut h, area.left_center() + vec2(6.0, 0.0));
    let colors = text_colors(&h, area);
    assert!(colors.contains(&t.accent_hover), "an ordinary hovered value: {colors:?}");
}
