//! Audio scrubbing: Ctrl/Cmd-dragging the current time plays one frame of the comp's mix at each
//! new frame (each snippet replacing what is still queued); holding still repeats nothing; silent
//! comps open no output; an idle scrub closes it. Previews with audio wait for frames.

use std::sync::{Arc, Mutex};

use effectcraft_engine::Session;
use effectcraft_engine::project::ItemId;
use effectcraft_engine::time::Tick;
use effectcraft_ui_egui::EffectcraftApp;
use effectcraft_ui_egui::audio::{AudioDevice, AudioFeed};
use egui::{Event, Modifiers, pos2};
use egui_kittest::Harness;
use serde_json::json;

/// An output that records the feed it is given and whether it was stopped.
#[derive(Clone, Default)]
struct Fake(Arc<Mutex<(Option<Arc<AudioFeed>>, bool)>>);

impl AudioDevice for Fake {
    fn sample_rate(&self) -> u32 {
        48000
    }
    fn start(&mut self, feed: Arc<AudioFeed>) -> Result<(), String> {
        self.0.lock().unwrap().0 = Some(feed);
        Ok(())
    }
    fn stop(&mut self) {
        self.0.lock().unwrap().1 = true;
    }
}

/// A 2 s, 30 fps comp whose solid sounds a tone (Audio switch on).
fn setup(audible: bool, fake: &Fake) -> (EffectcraftApp, ItemId) {
    setup_long(audible, fake, 2.0)
}

/// A `duration`-second, 30 fps comp whose solid sounds a tone (Audio switch on).
fn setup_long(audible: bool, fake: &Fake, duration: f64) -> (EffectcraftApp, ItemId) {
    let mut s = Session::default();
    s.execute("comp.new", json!({"name": "A", "width": 64, "height": 36, "frameRate": 30, "duration": duration})).unwrap();
    let cid = s.active_comp_id().unwrap();
    let l = s.execute("layer.newSolid", json!({"color": "#808080"})).unwrap()["layer"].as_u64().unwrap();
    if audible {
        s.execute("effect.apply", json!({"layers": [l], "effect": "ec.audio.tone"})).unwrap();
        s.execute("layer.setSwitch", json!({"layers": [l], "switch": "audio", "value": true})).unwrap();
    }
    let mut app = EffectcraftApp::new(s);
    let f = fake.clone();
    app.hooks.audio_device = Some(Box::new(move |_| Some(Box::new(f.clone()) as Box<dyn AudioDevice>)));
    (app, cid)
}

#[test]
fn scrubbing_plays_one_frame_per_new_frame() {
    let fake = Fake::default();
    let (mut app, cid) = setup(true, &fake);
    let t = |s: f64| Tick::from_seconds_f64(s);
    app.scrub_audio(cid, t(0.5), 0.0);
    let feed = fake.0.lock().unwrap().0.clone().expect("the output opened");
    // One 30 fps frame at 48 kHz, and not silence.
    assert_eq!(feed.queued_frames(), 1600);
    let mut out = vec![0.0f32; 3200];
    feed.pull(&mut out);
    assert!(out.iter().any(|v| v.abs() > 0.01), "the tone is there");
    // Faded in: the first sample is silent.
    assert_eq!(out[0], 0.0);
    // The same frame again plays nothing; the next frame replaces what is left.
    app.scrub_audio(cid, t(0.51), 0.1);
    assert_eq!(feed.queued_frames(), 0);
    feed.push(&[0.0; 1000]);
    app.scrub_audio(cid, t(0.6), 0.2);
    assert_eq!(feed.queued_frames(), 1600, "replaced, not piled up");
    // A silent comp opens nothing.
    let silent = Fake::default();
    let (mut quiet, qid) = setup(false, &silent);
    quiet.scrub_audio(qid, t(0.5), 0.0);
    assert!(silent.0.lock().unwrap().0.is_none() && quiet.scrub.is_none());
}

#[test]
fn cmd_dragging_the_current_time_scrubs_and_an_idle_scrub_closes() {
    let fake = Fake::default();
    let (app, _) = setup(true, &fake);
    let mut h = Harness::builder().with_size(egui::vec2(1400.0, 900.0)).build_eframe(|_| app);
    h.run_steps(3);
    let cti = h.state().auto.find("timeline.cti").expect("current time indicator").clone();
    let from = pos2(cti.rect[0] + cti.rect[2] / 2.0, cti.rect[1] + cti.rect[3] / 2.0);
    let cmd = Modifiers::COMMAND;
    h.event(Event::ModifiersChanged(cmd));
    h.event(Event::PointerMoved(from));
    h.step();
    h.event(Event::PointerButton { pos: from, button: egui::PointerButton::Primary, pressed: true, modifiers: cmd });
    h.step();
    for k in 1..=4 {
        h.event(Event::PointerMoved(from + egui::vec2(40.0 * k as f32, 0.0)));
        h.step();
    }
    assert!(h.state().scrub.is_some(), "Cmd-drag opened the scrub output");
    assert!(fake.0.lock().unwrap().0.as_ref().is_some_and(|f| f.queued_frames() > 0));
    h.event(Event::PointerButton { pos: from + egui::vec2(160.0, 0.0), button: egui::PointerButton::Primary, pressed: false, modifiers: cmd });
    h.step();
    h.event(Event::ModifiersChanged(Modifiers::NONE));
    h.step();
    // An idle second closes the output.
    for _ in 0..90 {
        h.step();
        std::thread::sleep(std::time::Duration::from_millis(15));
        if h.state().scrub.is_none() {
            break;
        }
    }
    assert!(h.state().scrub.is_none() && fake.0.lock().unwrap().1, "closed and stopped");
}

/// The comp's frame at the current time.
fn frame(h: &Harness<'_, EffectcraftApp>, cid: ItemId) -> i64 {
    let app = h.state();
    app.session.project.comp(cid).unwrap().frame_rate.frame_at(app.session.time())
}

/// The frame under the preview audio's first sample (its clock before the device pulls any).
fn audio_frame(h: &Harness<'_, EffectcraftApp>, cid: ItemId) -> Option<i64> {
    let app = h.state();
    let a = app.audio.as_ref()?;
    Some(effectcraft_ui_egui::audio::frame_of_sample(a.clock(), a.rate, app.session.project.comp(cid).unwrap().frame_rate))
}

/// Step until the preview's sound starts (or 60 s pass, for a loaded machine), checking the
/// playhead holds `first` meanwhile. Returns whether the sound started.
fn until_sound(h: &mut Harness<'_, EffectcraftApp>, cid: ItemId, first: i64) -> bool {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    while std::time::Instant::now() < deadline {
        h.step();
        if h.state().audio.is_some() {
            return true;
        }
        assert_eq!(frame(h, cid), first, "the playhead holds the first frame until the sound can start");
        assert!(h.state().playback.starting && h.state().playback.playing);
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    false
}

/// Starting a preview with sound before its frames are cached (#947): the playhead holds the
/// first frame while the frames ahead render, then sound and picture start together from it (the
/// sound used to start at a later frame, losing the opening). A frame that isn't cached later on
/// still stops the sound rather than being skipped (#103).
#[test]
fn preview_with_audio_holds_the_first_frame_and_sounds_from_it() {
    let fake = Fake::default();
    let (mut app, cid) = setup(true, &fake);
    // Several steps' worth of frames to cache before the sound can start (see the prefetch).
    let ahead = (app.frames.parallelism() * 2).max(4);
    app.session.execute("comp.settings", json!({"comp": cid.0, "duration": (4 * ahead) as f64 / 30.0})).unwrap();
    let mut h = Harness::builder().with_size(egui::vec2(1400.0, 900.0)).with_step_dt(1.0 / 31.0).build_eframe(|_| app);
    h.run_steps(2);
    let now = h.ctx.input(|i| i.time);
    h.state_mut().play(now);
    assert!(h.state().playback.starting && h.state().audio.is_none(), "nothing is cached: no sound yet");
    assert!(until_sound(&mut h, cid, 0), "the sound started once the frames ahead were cached");
    assert!(fake.0.lock().unwrap().0.is_some());
    assert_eq!(audio_frame(&h, cid), Some(0), "from the first frame, not a later one");
    assert!(!h.state().playback.starting);
    // A frame that isn't cached stops the sound rather than being skipped. (Frames render fast
    // enough here to be cached again before the check: keep them out of the cache.)
    h.state().frames.set_budget(0);
    h.state().frames.clear();
    h.step();
    assert!(h.state().audio.is_none() && h.state().playback.audio_held && h.state().playback.playing);
}

/// A long (30 s) preview sounds once the frames ahead of its first frame are cached, not the
/// whole work area (#208), and from that first frame (#947).
#[test]
fn preview_with_audio_sounds_without_caching_the_work_area() {
    let fake = Fake::default();
    let (app, cid) = setup_long(true, &fake, 30.0);
    let mut h = Harness::builder().with_size(egui::vec2(1400.0, 900.0)).with_step_dt(1.0 / 30.0).build_eframe(|_| app);
    h.run_steps(2);
    let now = h.ctx.input(|i| i.time);
    h.state_mut().play(now);
    assert!(until_sound(&mut h, cid, 0), "the sound started");
    assert_eq!(audio_frame(&h, cid), Some(0), "from the first frame");
}

/// Pull `frames` stereo frames from the output, waiting for the feeder thread to mix them.
fn pull(feed: &AudioFeed, frames: usize) -> Vec<f32> {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    while feed.queued_frames() < frames && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let mut out = vec![0.0f32; frames * 2];
    feed.pull(&mut out);
    out
}

/// The opening of the preview is heard (#947): a sound that lasts only the first 100 ms after the
/// start, from the beginning of the comp or from the middle, reaches the output device.
#[test]
fn a_sound_in_the_first_100_ms_is_heard() {
    for start in [0.0, 1.0] {
        let fake = Fake::default();
        let (mut app, cid) = setup_long(false, &fake, 4.0);
        // A tone only from `start` to `start` + 0.1 s.
        let l = app.session.execute("layer.newSolid", json!({"color": "#808080"})).unwrap()["layer"].as_u64().unwrap();
        app.session.execute("effect.apply", json!({"layers": [l], "effect": "ec.audio.tone"})).unwrap();
        app.session.execute("layer.setSwitch", json!({"layers": [l], "switch": "audio", "value": true})).unwrap();
        app.session.execute("layer.timing", json!({"layers": [l], "in": start, "out": start + 0.1})).unwrap();
        app.session.execute("time.set", json!({"time": start})).unwrap();
        let first = (start * 30.0) as i64;
        let mut h = Harness::builder().with_size(egui::vec2(1400.0, 900.0)).with_step_dt(1.0 / 30.0).build_eframe(|_| app);
        h.run_steps(2);
        let now = h.ctx.input(|i| i.time);
        h.state_mut().play(now);
        assert!(until_sound(&mut h, cid, first), "{start}: the sound started");
        assert_eq!(audio_frame(&h, cid), Some(first), "{start}: from the first frame");
        let feed = fake.0.lock().unwrap().0.clone().expect("the output opened");
        // The first 100 ms at 48 kHz.
        let out = pull(&feed, 4800);
        assert!(out.iter().any(|v| v.abs() > 0.01), "{start}: the opening tone reached the output");
    }
}

/// A preview without video starts its sound at once, from the first frame (unchanged), and so
/// does one whose frames are already cached.
#[test]
fn audio_only_previews_start_at_once() {
    let fake = Fake::default();
    let (app, cid) = setup(true, &fake);
    let mut h = Harness::builder().with_size(egui::vec2(1400.0, 900.0)).build_eframe(|_| app);
    h.run_steps(2);
    let now = h.ctx.input(|i| i.time);
    h.state_mut().play_with(now, effectcraft_engine::preview::PreviewShortcut::AltNumpad0);
    assert!(h.state().audio.is_some() && !h.state().playback.starting, "Video off: the sound starts at once");
    assert_eq!(audio_frame(&h, cid), Some(0));
}

/// Without an output device the preview still plays, from the first frame, once the frames ahead
/// are cached; Stop while it prepares leaves nothing playing.
#[test]
fn no_device_plays_the_picture_and_stop_ends_the_wait() {
    let fake = Fake::default();
    let (mut app, cid) = setup_long(true, &fake, 10.0);
    app.hooks.audio_device = Some(Box::new(|_| None));
    let mut h = Harness::builder().with_size(egui::vec2(1400.0, 900.0)).with_step_dt(1.0 / 30.0).build_eframe(|_| app);
    h.run_steps(2);
    let now = h.ctx.input(|i| i.time);
    h.state_mut().play(now);
    assert!(h.state().playback.starting);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    while h.state().playback.starting && std::time::Instant::now() < deadline {
        h.step();
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert!(!h.state().playback.starting && h.state().audio.is_none() && h.state().playback.playing, "no device: the picture plays alone");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while frame(&h, cid) == 0 && std::time::Instant::now() < deadline {
        h.step();
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert!(frame(&h, cid) > 0, "from the first frame on");
    // Stop while preparing.
    h.state_mut().stop();
    let now = h.ctx.input(|i| i.time);
    h.state_mut().hooks.audio_device = Some(Box::new(move |_| Some(Box::new(fake.clone()) as Box<dyn AudioDevice>)));
    h.state().frames.clear();
    h.state_mut().play(now);
    assert!(h.state().playback.starting);
    h.state_mut().stop();
    for _ in 0..10 {
        h.step();
    }
    assert!(!h.state().playback.playing && !h.state().playback.starting && h.state().audio.is_none(), "Stop while preparing: nothing plays");
}
