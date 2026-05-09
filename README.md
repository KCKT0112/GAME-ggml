# GAME ggml backend

[![ggml_backend CI](https://github.com/openvpi/GAME/actions/workflows/ggml_backend.yml/badge.svg)](https://github.com/openvpi/GAME/actions/workflows/ggml_backend.yml)

Native C++ inference for the [GAME](../) singing-voice-to-MIDI model, built on
[ggml](https://github.com/ggerganov/ggml).  Runs on CPU, Metal (default on
Apple Silicon), CUDA, or Vulkan.  Drop-in replacement for
`python infer.py extract` with no Python dependency at runtime.

## Highlights

- **End-to-end CLI** — WAV in, MIDI/TXT/CSV out (mirrors Python `extract`).
- **Small footprint** — ~50 MB GGUF for the 1.0-small checkpoint, ~12.7 M params.
- **Fast startup** — Metal binary-archive patch keeps first-run latency under a
  second on Apple Silicon.
- **Third-party integration** — clean PIMPL C++ API; `add_subdirectory` and link
  `game_ggml::game_ggml`.
- **Parity-tested** — full pipeline output matches the PyTorch reference bit-for-bit
  when the same RNG numbers are injected.

## Architecture

```
waveform (44100 Hz mono)
      │
      ▼
   MelExtractor (pocketfft STFT + librosa-compatible mel)
      │ mel [T, 80]
      ▼
   Encoder (EBFBackbone, 4 layers, dim=128)
      │ x_seg, x_est  each [T, 128]
      ▼
   D3PM loop (8 steps)
     ├─ remove_mutable_boundaries (stochastic)
     ├─ Segmenter (EBFBackbone, 8 layers; noise/time/lang embeddings)
     └─ decode_soft_boundaries (local-max)
      │ regions [T]  +  N
      ▼
   Estimator (JEBFBackbone, 4 layers; joint attention, mixed RoPE)
      │ pool_logits [N, 257]
      ▼
   Gaussian-blurred pitch decode → notes (offset, duration, pitch, voiced)
```

## Build

```bash
cmake -S ggml_backend -B ggml_backend/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DGAME_GGML_BUILD_TESTS=ON
cmake --build ggml_backend/build -j
```

Options:

| Option | Default | Meaning |
|---|---|---|
| `GAME_GGML_METAL`       | `ON` (Apple only) | Enable Metal backend |
| `GAME_GGML_CUDA`        | `OFF`             | Enable CUDA backend |
| `GAME_GGML_VULKAN`      | `OFF`             | Enable Vulkan backend |
| `GAME_GGML_BUILD_CLI`   | `ON`              | Build `game_ggml_cli` |
| `GAME_GGML_BUILD_TESTS` | `OFF`             | Build GoogleTest suite |

## Convert a PyTorch checkpoint

```bash
pip install -r ggml_backend/scripts/requirements.txt
python ggml_backend/scripts/convert_pt_to_gguf.py \
    --model-dir GAME-pt-1.0-small \
    -o ggml_backend/assets/game_small.gguf
```

The script reads `model.pt` + `config.yaml` + `lang_map.json` from the given
directory and writes a single GGUF file containing all 671 tensors (FP32) and
74 metadata KV pairs.

Inspect the result:

```bash
./ggml_backend/build/bin/game_ggml_cli inspect ggml_backend/assets/game_small.gguf
```

## Run inference

```bash
./ggml_backend/build/bin/game_ggml_cli extract input.wav \
    -m ggml_backend/assets/game_small.gguf \
    --output-formats mid,txt,csv \
    --output-dir out/ \
    --tempo 120 \
    --seed 42
```

### CLI → `infer.py extract` option mapping

| CLI flag | Python equivalent |
|---|---|
| `-m / --model`        | `-m` |
| `-l / --language`     | `-l` (takes numeric id — use `inspect` to see the mapping) |
| `--output-formats`    | `--output-formats` |
| `--output-dir`        | `--output-dir` |
| `--tempo`             | `--tempo` |
| `--seg-threshold`     | `--seg-threshold` |
| `--seg-radius`        | `--seg-radius` (in frames) |
| `--est-threshold`     | `--est-threshold` |
| `--t0` / `--nsteps`   | `--t0` / `--nsteps` |
| `--seed`              | *(new)* — 0 pulls a random seed from the OS |
| `--pitch-format`      | `--pitch-format` |
| `--round-pitch`       | `--round-pitch` |

## Using as a third-party library

```cmake
add_subdirectory(path/to/GAME/ggml_backend)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE game_ggml::game_ggml)
```

```cpp
// main.cpp
#include <game_ggml/model.h>
#include <vector>

int main() {
    auto model = game_ggml::Model::load("game_small.gguf");
    std::vector<float> waveform = /* ... load 44100 Hz mono ... */;

    game_ggml::InferParams params;
    params.language = 4;   // from lang_map: { "zh": 4 }
    params.seed     = 42;

    auto result = model.infer(waveform.data(), waveform.size(), params);
    for (const auto & n : result.notes) {
        if (!n.voiced) continue;
        printf("  %.2fs + %.2fs : %.2f\n",
               n.offset_seconds, n.duration_seconds, n.pitch_midi);
    }
}
```

The public header `<game_ggml/model.h>` uses PIMPL; consumers never transitively
include any ggml header.

See [`examples/external_consumer/`](examples/external_consumer/) for a minimal
standalone CMake project that builds against the library.

## Tests

```bash
ctest --test-dir ggml_backend/build --output-on-failure
```

The suite has 37 tests covering:

- Backend initialisation
- GGUF I/O round-trip
- Every op (RMSNorm, Linear, LayerScale, Embedding, GLU-FFN, CgMLP, RoPE in all
  three modes, Attention, PAC, EBF block)
- Encoder / Segmenter / Estimator end-to-end vs PyTorch reference dumps
- D3PM 8-step loop bit-exact with injected RNG (tolerates ≤ 2/100 boundary
  flips from Metal FP32 drift)
- Mel spectrogram vs `lib.feature.mel.StretchableMelSpectrogram`
- Slicer (short-clip + split-on-silence)
- MIDI writer (SMF type-0 structure)
- Text writers (TXT + CSV, note-name formatting)
- Full pipeline bit-exact E2E

Reference dumps are generated by `python scripts/dump_reference.py --category all`.
Dumps are gitignored — regenerate them as part of CI.

## Known limitations (v1)

- **44100 Hz mono WAV only** — other sample rates raise `InvalidWav`.  Resampling
  is deliberately out-of-scope to keep the footprint small.
- **FP32 weights only** — the converter emits FP32 GGUF.  Quantization is a
  later deliverable.
- **Only the shipped `1.0-small` config branch is supported**.  The estimator
  rejects `split` attention, learned pool merger, `region_token_num > 1`, and
  `use_region_bias=true` at load time with a clear `NotImplemented` message.
- **Batch size 1** per inference call — matches `infer.py extract`.  For
  parallel streams hold multiple `Model` instances.
- **Metal FP32 precision** — expected ~1e-3 per matmul; at boundary decoding
  this can flip one frame out of every few hundred vs the CPU reference.

## Dependencies

Everything is fetched at configure time by CMake; nothing is vendored.  Source
trees live under `build/_deps/<name>-src/` after the first configure.

| Dependency | Version pin | License | SPDX identifier |
|---|---|---|---|
| [ggml](https://github.com/ggerganov/ggml) | `v0.11.0` tag | MIT | MIT |
| [pocketfft](https://gitlab.mpcdf.mpg.de/mtr/pocketfft) | commit `32424d20` on `cpp` branch | BSD-3-Clause | BSD-3-Clause |
| [dr_libs](https://github.com/mackron/dr_libs) | commit `243e26ff` on `master` | Public Domain / MIT-0 (dual) | `Unlicense OR MIT-0` |
| [GoogleTest](https://github.com/google/googletest) | `v1.14.0` tag (tests only) | BSD-3-Clause | BSD-3-Clause |

Each upstream LICENSE file is preserved under `build/_deps/<name>-src/LICENSE*`
after download.  To update a dependency, change its `GIT_TAG` in
`cmake/Dependencies.cmake` and reconfigure.

## License

MIT — same as the parent [GAME project](../LICENSE).  Redistributions should
also carry the upstream license notices listed in the table above.
