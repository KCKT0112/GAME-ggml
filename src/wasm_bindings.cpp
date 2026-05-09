// Emscripten/WASM bindings for game_ggml.  Exposes:
//
//   Module.createGameGGML() -> Promise<Module>
//   const model = Module.createModelFromPtr(gguf_ptr, gguf_size)
//   const notes = model.infer(wav_ptr, n_samples, lang, seed_lo, seed_hi, nsteps)
//   notes[i] = { offset, duration, pitch, voiced }
//   model.delete()
//
// JS side allocates two buffers via Module._malloc, writes bytes / samples
// into them, then passes their pointers in.  Integer pointers are exposed
// to JS as `uintptr_t` (32-bit in WASM32).

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include "game_ggml/model.h"
#include "game_ggml/errors.h"
#include "game_ggml/version.h"
#include "game_ggml/config.h"

#include <cstdint>
#include <memory>
#include <string>

using namespace emscripten;

namespace {

// ---- Thin wrapper that holds the Model + exposes JS-friendly methods ----

class ModelJs {
public:
    explicit ModelJs(std::uintptr_t gguf_ptr, std::size_t n_bytes)
        : model_(std::make_unique<game_ggml::Model>(
              game_ggml::Model::load_from_memory(
                  reinterpret_cast<const void *>(gguf_ptr), n_bytes))) {}

    val infer(std::uintptr_t wav_ptr, std::size_t n_samples,
              int language, std::uint32_t seed_lo, std::uint32_t seed_hi,
              int nsteps) const {
        game_ggml::InferParams p;
        p.language    = language;
        p.d3pm_nsteps = nsteps;
        p.seed = (static_cast<std::uint64_t>(seed_hi) << 32) | seed_lo;

        auto res = model_->infer(
            reinterpret_cast<const float *>(wav_ptr), n_samples, p);

        val out = val::array();
        for (const auto & n : res.notes) {
            val note = val::object();
            note.set("offset",   n.offset_seconds);
            note.set("duration", n.duration_seconds);
            note.set("pitch",    n.pitch_midi);
            note.set("voiced",   n.voiced);
            out.call<void>("push", note);
        }
        return out;
    }

    int sample_rate() const { return model_->config().inference.audio_sample_rate; }
    int n_mels()      const { return model_->config().inference.n_mels;            }
    int embedding_dim() const { return model_->config().embedding_dim;             }
    int estimator_out_dim() const { return model_->config().estimator_out_dim;     }

    std::string arch() const { return model_->config().architecture; }

    void set_num_threads(int n) { model_->set_n_threads(n); }

private:
    std::unique_ptr<game_ggml::Model> model_;
};

// Exposed as a factory (JS land doesn't like C++ ctor arguments).
ModelJs * create_model_from_ptr(std::uintptr_t ptr, std::size_t size) {
    return new ModelJs(ptr, size);
}

std::string version_str() { return game_ggml::version_string(); }
std::string ggml_version_str() { return game_ggml::ggml_version_string(); }

}  // namespace

EMSCRIPTEN_BINDINGS(game_ggml) {
    class_<ModelJs>("Model")
        .function("infer",             &ModelJs::infer)
        .function("setNumThreads",     &ModelJs::set_num_threads)
        .function("sampleRate",        &ModelJs::sample_rate)
        .function("nMels",             &ModelJs::n_mels)
        .function("embeddingDim",      &ModelJs::embedding_dim)
        .function("estimatorOutDim",   &ModelJs::estimator_out_dim)
        .function("arch",              &ModelJs::arch);

    function("createModelFromPtr", &create_model_from_ptr,
             allow_raw_pointers());
    function("version",            &version_str);
    function("ggmlVersion",        &ggml_version_str);
}
