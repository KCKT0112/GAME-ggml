// Emscripten/WASM bindings for game_ggml.
//
// Exports via embind:
//
//   Module.createGameGGML() -> Promise<Module>
//   const model = Module.createModelFromPtr(ggufPtr, ggufSize)
//
//   // One-shot inference, for clips that fit comfortably in memory:
//   const notes = model.infer(wavPtr, nSamples, lang, seedLo, seedHi, nsteps)
//
//   // Silence-based slicing + per-chunk inference + offset stitching.
//   // Use this for long clips (> ~60 s) where a single graph would OOM:
//   const notes = model.inferLong(wavPtr, nSamples, lang, seedLo, seedHi, nsteps)
//
//   notes[i] = { offset, duration, pitch, voiced }
//   model.delete()

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include "game_ggml/model.h"
#include "game_ggml/errors.h"
#include "game_ggml/version.h"
#include "game_ggml/config.h"

#include "../src/cli/slicer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

using namespace emscripten;

namespace {

class ModelJs {
public:
    explicit ModelJs(std::uintptr_t gguf_ptr, std::size_t n_bytes)
        : model_(std::make_unique<game_ggml::Model>(
              game_ggml::Model::load_from_memory(
                  reinterpret_cast<const void *>(gguf_ptr), n_bytes))) {}

    val infer(std::uintptr_t wav_ptr, std::size_t n_samples,
              int language, std::uint32_t seed_lo, std::uint32_t seed_hi,
              int nsteps) const {
        auto p = make_params(language, seed_lo, seed_hi, nsteps);
        auto res = model_->infer(
            reinterpret_cast<const float *>(wav_ptr), n_samples, p);
        return notes_to_val(res.notes);
    }

    // Slice on silence (same algorithm as infer.py extract / the CLI), run
    // the model on each chunk, and stitch results back into the full-clip
    // timeline.  Avoids OOM on long clips.
    val inferLong(std::uintptr_t wav_ptr, std::size_t n_samples,
                   int language, std::uint32_t seed_lo, std::uint32_t seed_hi,
                   int nsteps) const {
        game_ggml::cli::SlicerConfig slc;
        slc.sample_rate = model_->config().inference.audio_sample_rate;
        const float * wav = reinterpret_cast<const float *>(wav_ptr);
        auto chunks = game_ggml::cli::slice_waveform(wav, n_samples, slc);

        auto p = make_params(language, seed_lo, seed_hi, nsteps);
        std::vector<game_ggml::Note> all;
        for (auto & ch : chunks) {
            auto res = model_->infer(ch.waveform.data(), ch.waveform.size(), p);
            for (auto & n : res.notes) {
                n.offset_seconds += static_cast<float>(ch.offset_seconds);
                all.push_back(n);
            }
        }
        return notes_to_val(all);
    }

    int sample_rate()       const { return model_->config().inference.audio_sample_rate; }
    int n_mels()            const { return model_->config().inference.n_mels;            }
    int embedding_dim()     const { return model_->config().embedding_dim;               }
    int estimator_out_dim() const { return model_->config().estimator_out_dim;           }
    std::string arch()      const { return model_->config().architecture;                }

    void set_num_threads(int n) { model_->set_n_threads(n); }

    // Return chunk boundaries (offset/length in samples) + time offset in
    // seconds.  Lets JS drive the per-chunk loop and yield to the browser
    // between chunks so the UI stays responsive on long clips.
    val sliceWaveform(std::uintptr_t wav_ptr, std::size_t n_samples) const {
        game_ggml::cli::SlicerConfig slc;
        slc.sample_rate = model_->config().inference.audio_sample_rate;
        const float * wav = reinterpret_cast<const float *>(wav_ptr);
        auto chunks = game_ggml::cli::slice_waveform(wav, n_samples, slc);

        // `slice_waveform` copies the waveform into each chunk — but for the
        // JS-driven path we only need the boundaries.  Walk the chunks and
        // recover (offset_samples, length_samples) from offset_seconds.
        val out = val::array();
        const int sr = slc.sample_rate;
        for (auto & ch : chunks) {
            val obj = val::object();
            const std::size_t off_samp =
                static_cast<std::size_t>(ch.offset_seconds * sr + 0.5);
            obj.set("offsetSamples", static_cast<unsigned>(off_samp));
            obj.set("offsetSeconds", ch.offset_seconds);
            obj.set("lengthSamples", static_cast<unsigned>(ch.waveform.size()));
            out.call<void>("push", obj);
        }
        return out;
    }

private:
    static game_ggml::InferParams make_params(int language,
                                               std::uint32_t seed_lo,
                                               std::uint32_t seed_hi,
                                               int nsteps) {
        game_ggml::InferParams p;
        p.language    = language;
        p.d3pm_nsteps = nsteps;
        p.seed = (static_cast<std::uint64_t>(seed_hi) << 32) | seed_lo;
        return p;
    }

    static val notes_to_val(const std::vector<game_ggml::Note> & notes) {
        val out = val::array();
        for (const auto & n : notes) {
            val note = val::object();
            note.set("offset",   n.offset_seconds);
            note.set("duration", n.duration_seconds);
            note.set("pitch",    n.pitch_midi);
            note.set("voiced",   n.voiced);
            out.call<void>("push", note);
        }
        return out;
    }

    std::unique_ptr<game_ggml::Model> model_;
};

ModelJs * create_model_from_ptr(std::uintptr_t ptr, std::size_t size) {
    return new ModelJs(ptr, size);
}

std::string version_str()      { return game_ggml::version_string();      }
std::string ggml_version_str() { return game_ggml::ggml_version_string(); }

}  // namespace

EMSCRIPTEN_BINDINGS(game_ggml) {
    class_<ModelJs>("Model")
        .function("infer",             &ModelJs::infer)
        .function("inferLong",         &ModelJs::inferLong)
        .function("sliceWaveform",     &ModelJs::sliceWaveform)
        .function("setNumThreads",     &ModelJs::set_num_threads)
        .function("sampleRate",        &ModelJs::sample_rate)
        .function("nMels",             &ModelJs::n_mels)
        .function("embeddingDim",      &ModelJs::embedding_dim)
        .function("estimatorOutDim",   &ModelJs::estimator_out_dim)
        .function("arch",              &ModelJs::arch);

    function("createModelFromPtr", &create_model_from_ptr, allow_raw_pointers());
    function("version",            &version_str);
    function("ggmlVersion",        &ggml_version_str);
}
