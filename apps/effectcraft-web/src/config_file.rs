//! `./effectcraft-config.json`, fetched once per page load and shared by whoever reads a key.

use std::cell::RefCell;

use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;

use crate::startup::{self, Startup};

thread_local! {
    /// `None`: not fetched yet; `Some(None)`: no usable file.
    static FILE: RefCell<Option<Option<Vec<u8>>>> = const { RefCell::new(None) };
}

async fn fetch() -> Option<Vec<u8>> {
    let init = web_sys::RequestInit::new();
    // Always ask the server: the file is the operator's switch, never a cached copy.
    init.set_cache(web_sys::RequestCache::NoCache);
    let req = web_sys::Request::new_with_str_and_init("effectcraft-config.json", &init).ok()?;
    let resp: web_sys::Response = JsFuture::from(web_sys::window()?.fetch_with_request(&req)).await.ok()?.dyn_into().ok()?;
    if resp.status() != 200 {
        return None;
    }
    let buf = JsFuture::from(resp.array_buffer().ok()?).await.ok()?;
    let a = js_sys::Uint8Array::new(&buf);
    (a.length() as usize <= startup::CONFIG_MAX).then(|| a.to_vec())
}

/// The config file's bytes (`None` when there is none), fetched on the first call.
pub async fn bytes() -> Option<Vec<u8>> {
    if let Some(done) = FILE.with(|f| f.borrow().clone()) {
        return done;
    }
    let got = fetch().await;
    FILE.with(|f| *f.borrow_mut() = Some(got.clone()));
    got
}

/// The configured startup behaviour.
pub async fn startup() -> Startup {
    let s = bytes().await.map(|b| startup::parse(&b)).unwrap_or_default();
    log::info!("startup: {s:?}");
    s
}
