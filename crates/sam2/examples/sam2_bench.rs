//! Timing of the SAM 2 image path on this machine: load, encode one frame, decode a click.
//!
//! `cargo run -p effectcraft-sam2 --example sam2_bench -- [MODEL_DIR]` (default `sam2-tiny`).
//! Prints a message and exits when the weights are not there.

use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use effectcraft_sam2::model::ImageModel;
use effectcraft_sam2::prompt::Prompt;

fn main() {
    if let Err(e) = run() {
        eprintln!("sam2_bench: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args().nth(1).unwrap_or_else(|| "sam2-tiny".to_string());
    let file = std::path::Path::new(&dir).join("model.safetensors");
    if !file.exists() {
        println!("no weights at {} (download facebook/sam2.1-hiera-tiny first)", file.display());
        return Ok(());
    }
    let t = Instant::now();
    let model = ImageModel::open(&file)?;
    println!("load: {:.2} s", t.elapsed().as_secs_f64());
    let pixels = Tensor::randn(0f32, 1.0, (1, 3, 1024, 1024), &Device::Cpu)?.to_dtype(DType::F32)?;
    for run in 0..3 {
        let t = Instant::now();
        let enc = model.encode(&pixels)?;
        let encode = t.elapsed().as_secs_f64();
        let prompt = Prompt { points: vec![([512.0, 512.0], true)], bbox: None };
        let t = Instant::now();
        let out = model.segment(&enc, &prompt, true)?;
        println!("run {run}: encode {encode:.2} s, click {:.0} ms, masks {:?}", t.elapsed().as_secs_f64() * 1e3, out.masks.dims());
    }
    // Where the encode time goes: patch embedding, then each Hiera block.
    let hiera = &model.vision.hiera;
    let t = Instant::now();
    let mut x = hiera.patch_embed(&pixels)?;
    println!("patch embed: {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
    for (i, b) in hiera.blocks.iter().enumerate() {
        let t = Instant::now();
        x = b.forward(&x)?;
        println!("block {i}: {:.0} ms, out {:?}", t.elapsed().as_secs_f64() * 1e3, x.dims());
    }
    println!("threads: {}", std::thread::available_parallelism().map_or(1, |n| n.get()));
    Ok(())
}
