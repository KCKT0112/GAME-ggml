// Plain C exports for the WASM build, used as an alternative to the embind
// class interface.  Why both?  embind-generated function dispatchers do NOT
// get wrapped with `WebAssembly.promising` under `-sJSPI=1`, so calling into
// them can't suspend — which breaks the WebGPU backend (which must suspend
// during `requestAdapter` / `onSubmittedWorkDone`).  Plain C exports ARE
// auto-promised, so JS code that uses `Module._gg_xxx()` can suspend safely.
//
// Layout of the notes buffer returned by `gg_infer`: tightly packed,
// 4 floats per note in the order (offset_seconds, duration_seconds,
// pitch_midi, voiced_as_float).  Caller frees with `gg_free`.

#include "game_ggml/model.h"
#include "game_ggml/errors.h"
#include "game_ggml/version.h"
#include "game_ggml/config.h"

#include "cli/slicer.h"

#include <emscripten.h>

#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

// Opaque state held on the heap; JS sees an integer handle.
struct WasmModelState {
    std::unique_ptr<game_ggml::Model> model;
};

WasmModelState * as_state(std::uintptr_t handle) {
    return reinterpret_cast<WasmModelState *>(handle);
}

}  // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE
std::uintptr_t gg_create_model(const void * gguf_ptr, std::size_t n_bytes) {
    try {
        auto state = new WasmModelState();
        state->model = std::make_unique<game_ggml::Model>(
            game_ggml::Model::load_from_memory(gguf_ptr, n_bytes));
        return reinterpret_cast<std::uintptr_t>(state);
    } catch (...) {
        return 0;
    }
}

EMSCRIPTEN_KEEPALIVE
void gg_destroy_model(std::uintptr_t handle) {
    delete as_state(handle);
}

EMSCRIPTEN_KEEPALIVE
void gg_set_threads(std::uintptr_t handle, int n) {
    auto * s = as_state(handle);
    if (s && s->model) s->model->set_n_threads(n);
}

EMSCRIPTEN_KEEPALIVE
int gg_sample_rate(std::uintptr_t handle) {
    auto * s = as_state(handle);
    return s && s->model ? s->model->config().inference.audio_sample_rate : 0;
}

// Slice a waveform on silence.  Writes up to `max_chunks` entries of 3 uint32
// values into `out_buf`: (offset_samples, length_samples, offset_seconds*1000).
// Returns the real number of chunks (may be > max_chunks, in which case the
// caller should retry with a larger buffer).
EMSCRIPTEN_KEEPALIVE
int gg_slice(std::uintptr_t handle,
             const float * wav, std::size_t n_samples,
             std::uint32_t * out_buf, int max_chunks) {
    auto * s = as_state(handle);
    if (!s || !s->model) return 0;
    game_ggml::cli::SlicerConfig slc;
    slc.sample_rate = s->model->config().inference.audio_sample_rate;
    auto chunks = game_ggml::cli::slice_waveform(wav, n_samples, slc);
    const int sr = slc.sample_rate;
    const int n = static_cast<int>(chunks.size());
    for (int i = 0; i < n && i < max_chunks; ++i) {
        const auto & ch = chunks[i];
        const std::uint32_t off_samp =
            static_cast<std::uint32_t>(ch.offset_seconds * sr + 0.5);
        out_buf[3 * i + 0] = off_samp;
        out_buf[3 * i + 1] = static_cast<std::uint32_t>(ch.waveform.size());
        out_buf[3 * i + 2] = static_cast<std::uint32_t>(ch.offset_seconds * 1000.0 + 0.5);
    }
    return n;
}

// Run inference on one chunk.  Allocates a float buffer of size `*out_count * 4`
// (caller must `gg_free()`).  Returns NULL on failure.
EMSCRIPTEN_KEEPALIVE
float * gg_infer(std::uintptr_t handle,
                 const float * wav, std::size_t n_samples,
                 int language,
                 std::uint32_t seed_lo, std::uint32_t seed_hi,
                 int nsteps,
                 std::size_t * out_count) {
    auto * s = as_state(handle);
    if (!s || !s->model) {
        if (out_count) *out_count = 0;
        return nullptr;
    }
    try {
        game_ggml::InferParams p;
        p.language    = language;
        p.d3pm_nsteps = nsteps;
        p.seed = (static_cast<std::uint64_t>(seed_hi) << 32) | seed_lo;

        auto res = s->model->infer(wav, n_samples, p);
        const std::size_t count = res.notes.size();
        float * buf = static_cast<float *>(std::malloc(count * 4 * sizeof(float)));
        if (!buf) {
            if (out_count) *out_count = 0;
            return nullptr;
        }
        for (std::size_t i = 0; i < count; ++i) {
            buf[4 * i + 0] = res.notes[i].offset_seconds;
            buf[4 * i + 1] = res.notes[i].duration_seconds;
            buf[4 * i + 2] = res.notes[i].pitch_midi;
            buf[4 * i + 3] = res.notes[i].voiced ? 1.0f : 0.0f;
        }
        if (out_count) *out_count = count;
        return buf;
    } catch (...) {
        if (out_count) *out_count = 0;
        return nullptr;
    }
}

EMSCRIPTEN_KEEPALIVE
void gg_free(void * ptr) {
    std::free(ptr);
}

// Return a pointer to a static, null-terminated version string.
EMSCRIPTEN_KEEPALIVE
const char * gg_version() {
    static const std::string v = game_ggml::version_string();
    return v.c_str();
}

EMSCRIPTEN_KEEPALIVE
const char * gg_ggml_version() {
    static const std::string v = game_ggml::ggml_version_string();
    return v.c_str();
}

}  // extern "C"
