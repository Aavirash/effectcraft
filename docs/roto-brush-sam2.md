# Roto Brush with SAM 2

Roto Brush can follow an object through a clip with Meta's **SAM 2.1 Tiny** (Hiera-Tiny, 39M parameters,
Apache-2.0). It is optional: nothing changes until you install the model and choose it.

## Using it
1. Settings ▸ Roto Brush ▸ SAM 2.1 Tiny ▸ **Install** (downloads `model.safetensors`, 155.9 MB, from the pinned
   Hugging Face commit; the SHA-256 is checked and the licence notice is kept next to the file). Or install from a
   file you downloaded. Agents: `roto.model.install {id: "sam2.1-hiera-tiny"}`.
2. Choose it (`roto.model.select {id: "sam2.1-hiera-tiny"}`).
3. Paint strokes on the base frame as usual, then **Propagate**. One tracking pass follows the object through the
   whole span in both directions, using SAM 2's streaming memory, so it keeps tracking through occlusion and when the
   object leaves the frame and returns. Each frame's probability map is then refined by the same graph cut as every other
   Roto Brush frame: your strokes stay hard constraints and edges snap to colour.
4. `roto.status {frame}` reports `source` (`tracker`, `model` for a frame the model segmented on its own, or
   `classical`) and, for tracked frames, `tracker.frames[f]`: `confidence` (0–1, the model's object score times its
   predicted mask quality) and `occluded` (the object is not in the frame: an empty matte). `model` says what Settings
   chose and whether it is installed, loaded and loading; `notice` says when it is not the one in use for the layer.
   `tracker.saved` names the model the saved frames came from.

Propagate never falls back to the classic engine without saying so. With SAM 2 chosen but not installed, or failing to
load, it refuses with the reason. While it is still loading, `wait: true` (agents) loads it first; without `wait` it
refuses, and the load goes on in the background. Deselecting SAM 2 (choosing Classic) runs Propagate exactly as before,
and Roto Brush 1.0 never uses a model.

Tracked frames are saved in the project with the instance, as Freeze's are, so a saved project renders the same when it
reopens, in the app or with `effectcraft-cli`, without the model being loaded or chosen there. They apply while the
strokes and span they came from are unchanged. Changing a stroke or the span makes them stale until the next Propagate.
Choosing another model, or the classic engine, leaves them in place and `roto.status` says so; a classic Propagate
replaces them with classic mattes. Object Matte saves nothing.

## Corrections re-track only what they change
A correction stroke changes the keys of the frames from it on (in both directions from the base), so Propagate tracks
only those again: the window starts at the nearest prompted frame at or before the first frame it needs (the corrected
frame, which restarts the object with its own prompt) and runs to the last frame it needs. Frames before the correction
keep their tracked results, and a Propagate with nothing changed does no tracking. The older non-conditioning memory
frames before the corrected frame are not rebuilt, so frames just after it can differ slightly from a from-scratch pass.
Tracked frames of a session are in memory; a project reopened in a new process re-tracks the whole span after a stroke
change (its saved tracking is stale as a whole).

## Memory and limits
The weights take about 156 MB. A model idle for 10 minutes is unloaded (a toast says so; the choice stays) and loads
again when a Propagate or a stroke needs it. The tracker holds the memory banks of the objects it follows (about 2 MB per
stored frame, pruned to the frames later ones can still look at) and refuses to grow past 2 GiB, or past a time limit when
one is set, with a clear error instead of running out of memory. Two runs of the same prompts give bit-identical masks on
the CPU.

## How it works
`crates/sam2` is a Rust port (on [candle](https://github.com/huggingface/candle), native only) of the Hugging Face
`transformers` SAM 2 implementation, Apache-2.0 (see its `NOTICE`). Every stage is checked against PyTorch by
`crates/sam2/tests/reference.rs` (they skip without the weights; reference tensors come from
`tools/sam2_reference.py` and `tools/sam2_video_reference.py`):

| Stage | Max abs error vs PyTorch |
|---|---|
| Hiera backbone, 12 blocks | 2.6e-5 |
| Neck features / position encodings | 9.7e-6 / 6e-8 |
| Prompt embeddings | 1.2e-7 |
| Mask decoder (logits up to 11.6) | 1.1e-4 |
| Memory encoder | 3.5e-6 |
| Memory attention (rotary) | 2.6e-5 |
| 7-frame clip, forwards and backwards (9 frame results) | logits 0.0038, IoU 1.0000 |

SAM 2 is prompted with at most three foreground and three background points per frame: it was trained on a handful
of clicks, and measured confidence collapses with many (0.99 with 1–7 points, 0.25 with 16, 0.00 with 32).

## Speed (this PC: 16 threads, CPU only, release build)
| Step | Time |
|---|---|
| Encode one 1024 × 1024 frame | about **2.4 s** (3.4 s before the speed work: a parallel GELU, cached position encodings, pooling without permutes) |
| A click on an encoded frame | about 105 ms |
| Memory attention on a tracked frame (full bank of 7 memory frames) | about **2.4 s** |
| A tracked frame in all | about **5 s** (the first six have fewer memories) |

A 300-frame clip therefore takes on the order of 25 minutes on this CPU. (An earlier version of this page said one encode
per frame; the memory attention is as large as the encode.) What was tried and did not help: chunking the queries of the
global and memory attention to keep scores in cache. Candle has no AMD/Vulkan backend, so a GPU path is future work (a Burn
or hand-written wgpu route); `cargo run --release -p effectcraft-sam2 --example sam2_bench` and `--example op_bench` time the
pieces.

## Benchmark
`effectcraft-cli bench --roto [--frames N] [--sam2 model.safetensors] [--mobilesam mobile_sam.pt]` runs synthetic clips
with known mattes through all three. 12 frames, debug build, this PC; mean IoU (mean IoU after the object returns), seconds
per frame:

| Clip | Classical | MobileSAM | SAM 2.1 Tiny |
|---|---|---|---|
| move | 0.982 | 0.992 | 0.993 |
| light | 0.981 | 0.992 | 0.992 |
| occlusion | 0.978 | 0.881 | 0.988 |
| reenter | 0.632 (0.000) | 0.499 (0.725) | 0.975 (0.967) |
| s/frame | 0.01 | 1.1 | 8.2 |

MobileSAM prompts every frame independently from the flow-warped matte, so it follows the object while the flow does and
loses it when it is hidden; SAM 2's streaming memory is what lets it re-acquire the object. The synthetic clips are easy
(one textured disc), so the "move" and "light" rows mostly show all three working; the re-entry row is the difference.
