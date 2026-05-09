# GAME ggml — Browser Demo

Drag a WAV/MP3/FLAC file into the page, get a MIDI transcription back.
**100 % client-side** — no server-side inference, no uploads.

![architecture](../assets/onnx-workflow.jpg)

## Quick start

```bash
cd ggml_backend/web-demo
python serve.py         # serves at http://127.0.0.1:8080
# open http://127.0.0.1:8080/index.html in Chrome / Safari / Edge
```

Pick a model (Q8_0 is the smallest ~16 MB download), click **Load model**,
drag an audio file in, click **Transcribe**.  The piano-roll renders below
and you can download the MIDI.

## Files

```
web-demo/
├── index.html           # the page
├── demo.js              # audio decode + inference orchestration + piano roll + MIDI writer
├── serve.py             # dev server that sends COOP/COEP (needed for WebGPU)
├── game_ggml.js         # emscripten loader (generated)
├── game_ggml.wasm       # wasm binary (generated, ~1.15 MB)
└── assets/              # one or more GGUFs to choose from in the UI
    ├── game_small_q8_0.gguf  (~16 MB)
    ├── game_small_f16.gguf   (~24 MB)
    └── game_small_f32.gguf   (~48 MB)
```

## Build

From the repo root:

```bash
# 1. Convert checkpoints → GGUFs (once)
python scripts/convert_pt_to_gguf.py --model-dir GAME-pt-1.0-small \
    -o assets/game_small_q8_0.gguf --dtype q8_0
python scripts/convert_pt_to_gguf.py --model-dir GAME-pt-1.0-small \
    -o assets/game_small_f16.gguf  --dtype f16
python scripts/convert_pt_to_gguf.py --model-dir GAME-pt-1.0-small \
    -o assets/game_small_f32.gguf  --dtype f32

# 2. Build the WASM module via Emscripten (needs emcmake in PATH)
#    Set GGML_WEBGPU=ON if you want the browser's GPU (optional, experimental).
emcmake cmake -S . -B build-wasm \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CCACHE=OFF \
    -DGGML_WEBGPU=ON
cmake --build build-wasm -j

# 3. Stage artifacts
cp build-wasm/bin/game_ggml.{js,wasm}   web-demo/
cp assets/game_small_*.gguf             web-demo/assets/
```

## WASM performance notes

WASM SIMD128 has **no native F16 or int8 ALU**, so quantized weights pay a
decode-to-F32 cost on every matmul.  For our ~50 MB model this usually
outweighs the DRAM-bandwidth savings you'd get on native CPU; the
counter-intuitive result on M-series (Node, single-threaded) is:

| Format | GGUF size | 2-s clip infer | RTF |
|---|---|---|---|
| Q8_0 | 16 MB | ~830 ms | 2.4× |
| F16  | 24 MB | ~890 ms | 2.3× |
| F32  | 48 MB | **~310 ms** | **6.5×** |

So pick your poison:

- **Mobile / slow network** → Q8_0 (smallest download)
- **Desktop / LAN**         → F32  (fastest inference)
- **Middle-of-the-road**    → F16

In the future, enabling `-pthread` (requires COOP/COEP, which `serve.py`
already sends) + WebGPU should give F16/Q8 a real advantage — pinning that
work until the ggml WebGPU backend stabilises for Emscripten.

## WebGPU

The CMake option `-DGGML_WEBGPU=ON` embeds ggml's experimental WebGPU
backend (via Emscripten's `emdawnwebgpu` port).  When the browser exposes
`navigator.gpu`, ggml will try to use it; otherwise it transparently falls
back to the CPU backend.

Tested targets (all CPU-fallback working; WebGPU-accelerated validation
pending browser support):
- Chrome 113+          (WebGPU stable)
- Safari 18+ (macOS)   (WebGPU stable)
- Edge 113+            (WebGPU stable)
- Firefox Nightly      (WebGPU behind flag)

## Known limitations

- **Decoder depends on the browser**: `AudioContext.decodeAudioData` handles
  most formats but there's no fallback for weird codecs.  Save weird files
  as WAV first.
- **Single-threaded CPU for now**.  Emscripten threads + WebWorker pool
  would take another day of work and COOP/COEP server config (already set
  by `serve.py`).
- **Model is 44.1 kHz**; the demo linearly resamples if needed.  Linear is
  "good enough" at the ratios we see (48→44.1, 22→44.1); for archival
  audio I'd use a proper sinc filter.
- **No streaming**: the whole audio is in memory at once.  For >5 min
  clips the page will use ~500 MB RAM peak.
