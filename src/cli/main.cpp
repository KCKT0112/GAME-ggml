// Command-line entry point for game_ggml.
//
// Subcommands:
//   --version / --help                  (task 1)
//   inspect <gguf>                      (task 2)
//   extract <wav> -m <gguf> [options]   (task 10)

#include "game_ggml/game_ggml.h"
#include "game_ggml/version.h"
#include "game_ggml/config.h"
#include "game_ggml/model.h"

#include "../backend.h"
#include "../gguf_io.h"
#include "../model_impl.h"
#include "../rng.h"
#include "wav_io.h"
#include "slicer.h"
#include "midi_writer.h"
#include "text_writer.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

// ---- tiny argument parser helpers ---------------------------------------

struct ArgView {
    int argc;
    char ** argv;
    int cursor = 0;

    bool has_next() const { return cursor < argc; }
    std::string next() { return argv[cursor++]; }
    std::string peek() const { return cursor < argc ? argv[cursor] : std::string{}; }

    std::string consume_or(const std::string & flag, const std::string & fallback) {
        if (peek() == flag && cursor + 1 < argc) {
            cursor += 2;
            return argv[cursor - 1];
        }
        return fallback;
    }
};

std::set<std::string> split_set(const std::string & s, char sep = ',') {
    std::set<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == sep) {
            if (!cur.empty()) out.insert(cur);
            cur.clear();
        } else cur += c;
    }
    if (!cur.empty()) out.insert(cur);
    return out;
}

// ---- --version ----------------------------------------------------------

void print_usage(const char * argv0) {
    std::fprintf(stdout,
        "Usage: %s <command> [options]\n"
        "\n"
        "Commands:\n"
        "  --version                              Print library + ggml version\n"
        "  --help                                 Show this message\n"
        "  inspect <gguf>                         Show GGUF metadata + tensor summary\n"
        "  extract <wav> -m <gguf> [options]      Transcribe a WAV file to MIDI/TXT/CSV\n"
        "\n"
        "Extract options:\n"
        "  -m, --model <path>                     Path to .gguf model                     (required)\n"
        "  -l, --language <id>                    Language id (0 = unknown)                (default: 0)\n"
        "  --output-formats <mid,txt,csv>         Comma-separated output formats            (default: mid)\n"
        "  --output-dir <dir>                     Directory for outputs                     (default: alongside input)\n"
        "  --tempo <bpm>                          MIDI tempo                               (default: 120)\n"
        "  --seg-threshold <float>                Boundary decoding threshold              (default: 0.2)\n"
        "  --seg-radius <frames>                  Boundary decoding radius                 (default: 2)\n"
        "  --est-threshold <float>                Note presence threshold                  (default: 0.2)\n"
        "  --t0 <float>                           D3PM initial t                           (default: 0.0)\n"
        "  --nsteps <int>                         D3PM sampling steps                      (default: 8)\n"
        "  --seed <uint64>                        RNG seed (0 = random_device)             (default: 0)\n"
        "  --pitch-format name|number             Text output pitch format                 (default: name)\n"
        "  --round-pitch                          Round pitch to integer in text output    (default: false)\n"
        "  --rng-replay <path>                    Feed float32 uniform samples from file    (parity vs PyTorch)\n",
        argv0);
}

void print_version() {
    std::fprintf(stdout, "game_ggml %s · ggml %s · backends: [",
        game_ggml::version_string(), game_ggml::ggml_version_string());
    const int n = game_ggml::available_backends_count();
    const char * const * names = game_ggml::available_backends();
    for (int i = 0; i < n; ++i) {
        std::fprintf(stdout, "%s%s", names[i], i + 1 < n ? ", " : "");
    }
    std::fprintf(stdout, "]\n");
    try {
        auto * b = game_ggml::internal::init_best_backend();
        std::fprintf(stdout, "active backend: %s\n", game_ggml::internal::backend_name(b));
        game_ggml::internal::free_backend(b);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "warning: %s\n", e.what());
    }
}

// ---- inspect ------------------------------------------------------------

int cmd_inspect(int argc, char ** argv) {
    if (argc < 1) { std::fprintf(stderr, "usage: inspect <path.gguf>\n"); return 1; }
    std::string path = argv[0];
    int tensor_limit = 20;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "-n" || a == "--tensor-limit") && i + 1 < argc) {
            tensor_limit = std::atoi(argv[++i]);
        }
    }

    auto file = game_ggml::internal::GgufFile::open(path);
    auto cfg  = game_ggml::internal::load_config(file);
    std::fprintf(stdout, "== %s ==\n", path.c_str());
    std::fprintf(stdout, "  architecture : %s\n", cfg.architecture.c_str());
    std::fprintf(stdout, "  name         : %s\n", cfg.name.c_str());
    std::fprintf(stdout, "  embedding_dim: %d\n", cfg.embedding_dim);
    std::fprintf(stdout, "  encoder      : %d layers, %d heads\n", cfg.encoder.num_layers, cfg.encoder.num_heads);
    std::fprintf(stdout, "  segmenter    : %d layers, latent@%d\n", cfg.segmenter.num_layers, cfg.segmenter.latent_layer_idx);
    std::fprintf(stdout, "  estimator    : %d layers, %s attn, R=%d\n", cfg.estimator.num_layers,
                 cfg.estimator.attn_type.c_str(), cfg.estimator.region_token_num);
    auto tensors = file.list_tensors();
    std::fprintf(stdout, "  tensors      : %zu\n", tensors.size());
    for (std::size_t i = 0; i < tensors.size() && static_cast<int>(i) < tensor_limit; ++i) {
        std::fprintf(stdout, "    %-60s size=%zu\n", tensors[i].name.c_str(), tensors[i].size_bytes);
    }
    return 0;
}

// ---- extract ------------------------------------------------------------

int cmd_extract(int argc, char ** argv) {
    using namespace game_ggml;
    using namespace game_ggml::cli;

    if (argc < 1) { std::fprintf(stderr, "usage: extract <wav> -m <gguf> [options]\n"); return 1; }
    const std::string input = argv[0];

    std::string model_path;
    std::string output_dir;
    std::set<std::string> output_formats = {"mid"};
    int  language     = 0;
    int  tempo        = 120;
    float seg_thr     = 0.2f;
    int   seg_radius  = 2;
    float est_thr     = 0.2f;
    float t0          = 0.0f;
    int   nsteps      = 8;
    std::uint64_t seed = 0;
    std::string pitch_format = "name";
    bool round_pitch = false;
    std::string rng_replay_path;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const std::string & flag) -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s requires a value\n", flag.c_str()); std::exit(1); }
            return argv[++i];
        };
        if      (a == "-m" || a == "--model")          model_path = next(a);
        else if (a == "-l" || a == "--language")       language = std::atoi(next(a).c_str());
        else if (a == "--output-formats")              output_formats = split_set(next(a));
        else if (a == "--output-dir")                  output_dir = next(a);
        else if (a == "--tempo")                       tempo      = std::atoi(next(a).c_str());
        else if (a == "--seg-threshold")               seg_thr    = std::stof(next(a));
        else if (a == "--seg-radius")                  seg_radius = std::atoi(next(a).c_str());
        else if (a == "--est-threshold")               est_thr    = std::stof(next(a));
        else if (a == "--t0")                          t0 = std::stof(next(a));
        else if (a == "--nsteps")                      nsteps = std::atoi(next(a).c_str());
        else if (a == "--seed")                        seed = std::strtoull(next(a).c_str(), nullptr, 10);
        else if (a == "--pitch-format")                pitch_format = next(a);
        else if (a == "--round-pitch")                 round_pitch = true;
        else if (a == "--rng-replay")                  rng_replay_path = next(a);
        else {
            std::fprintf(stderr, "unknown option: %s\n", a.c_str());
            return 1;
        }
    }
    if (model_path.empty()) { std::fprintf(stderr, "error: -m/--model is required\n"); return 1; }

    // Load model.
    std::fprintf(stderr, "loading model: %s\n", model_path.c_str());
    auto model = Model::load(model_path);

    // Load WAV.
    std::fprintf(stderr, "loading wav: %s\n", input.c_str());
    auto wav = load_wav_mono_f32(input, model.config().inference.audio_sample_rate);

    // Slice + inference per chunk.
    SlicerConfig slc_cfg;
    slc_cfg.sample_rate = model.config().inference.audio_sample_rate;
    auto chunks = slice_waveform(wav.samples.data(), wav.samples.size(), slc_cfg);
    std::fprintf(stderr, "sliced into %zu chunk(s)\n", chunks.size());

    InferParams p;
    p.language            = language;
    p.boundary_threshold  = seg_thr;
    p.boundary_radius     = seg_radius;
    p.note_threshold      = est_thr;
    p.d3pm_t0             = t0;
    p.d3pm_nsteps         = nsteps;
    p.seed                = seed;

    std::vector<Note> all_notes;

    // Optional bit-exact RNG replay.  When a file is provided we load the
    // whole sequence at once and reuse a single InjectedRng across every
    // chunk — matches the sequential consumption order used by PyTorch when
    // run with batch-size=1.
    std::unique_ptr<game_ggml::internal::IRandomSource> replay_rng;
    if (!rng_replay_path.empty()) {
        std::ifstream f(rng_replay_path, std::ios::binary);
        if (!f) { std::fprintf(stderr, "error: cannot open rng file: %s\n", rng_replay_path.c_str()); return 1; }
        f.seekg(0, std::ios::end);
        const std::size_t bytes = f.tellg();
        f.seekg(0, std::ios::beg);
        std::vector<float> vals(bytes / sizeof(float));
        f.read(reinterpret_cast<char*>(vals.data()), bytes);
        std::fprintf(stderr, "[rng-replay] loaded %zu uniform samples from %s\n",
                     vals.size(), rng_replay_path.c_str());
        replay_rng = std::make_unique<game_ggml::internal::InjectedRng>(std::move(vals));
    }

    for (std::size_t i = 0; i < chunks.size(); ++i) {
        auto & ch = chunks[i];
        std::fprintf(stderr, "  chunk %zu/%zu offset=%.3fs len=%.3fs\n",
                     i + 1, chunks.size(), ch.offset_seconds,
                     double(ch.waveform.size()) / slc_cfg.sample_rate);
        InferResult r;
        if (replay_rng) {
            r = model.internals().infer_with_rng(
                ch.waveform.data(), ch.waveform.size(), p, *replay_rng);
        } else {
            r = model.infer(ch.waveform.data(), ch.waveform.size(), p);
        }
        for (auto & n : r.notes) {
            n.offset_seconds += static_cast<float>(ch.offset_seconds);
            all_notes.push_back(n);
        }
    }
    std::fprintf(stderr, "total notes: %zu\n", all_notes.size());

    // Output.
    fs::path in_path(input);
    fs::path out_dir = output_dir.empty() ? in_path.parent_path() : fs::path(output_dir);
    fs::create_directories(out_dir);
    const std::string stem = in_path.stem().string();

    TextWriteOptions text_opts;
    text_opts.round_pitch = round_pitch;
    text_opts.use_names   = (pitch_format != "number");

    if (output_formats.count("mid")) {
        MidiWriteOptions mopts;
        mopts.tempo_bpm = tempo;
        auto p_out = out_dir / (stem + ".mid");
        write_midi_file(p_out.string(), all_notes, mopts);
        std::fprintf(stderr, "wrote %s\n", p_out.string().c_str());
    }
    if (output_formats.count("txt")) {
        auto p_out = out_dir / (stem + ".txt");
        write_text_file(p_out.string(), all_notes, TextFormat::Txt, text_opts);
        std::fprintf(stderr, "wrote %s\n", p_out.string().c_str());
    }
    if (output_formats.count("csv")) {
        auto p_out = out_dir / (stem + ".csv");
        write_text_file(p_out.string(), all_notes, TextFormat::Csv, text_opts);
        std::fprintf(stderr, "wrote %s\n", p_out.string().c_str());
    }
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) { print_usage(argv[0]); return 1; }
    std::string cmd = argv[1];
    try {
        if (cmd == "--version" || cmd == "-v") { print_version(); return 0; }
        if (cmd == "--help"    || cmd == "-h") { print_usage(argv[0]); return 0; }
        if (cmd == "inspect")                   return cmd_inspect(argc - 2, argv + 2);
        if (cmd == "extract")                   return cmd_extract(argc - 2, argv + 2);
        std::fprintf(stderr, "error: unknown command '%s'\n\n", cmd.c_str());
        print_usage(argv[0]);
        return 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
