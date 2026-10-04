#include "engine/community_models/dialogue_sidon/session.h"
#include "engine/community_models/dialogue_sidon/frontend.h"
#include "engine/community_models/dialogue_sidon/runtime.h"
#include "engine/community_models/dialogue_sidon/scheduler.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/sampling/torch_random.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace engine::community_models::dialogue_sidon {
namespace {

constexpr int kInputRate = 16000;
// The demo separates up to 120 s in one pass and stitches longer input from 120 s chunks
// that overlap by 10 s.
constexpr int64_t kChunkSamples = 120 * kInputRate;
constexpr int64_t kOverlapSamples = 10 * kInputRate;

struct Assets {
    assets::ResourceBundle resources;
};

// DIALOGUE_SIDON_DUMP_DIR=<dir> writes the first chunk's intermediates as raw float32 files
// (the names tests/dialogue_sidon/compare_parity.py reads); DIALOGUE_SIDON_NOISE=<file> replaces
// the seeded initial noise with raw float32 values, e.g. a Python reference's noise.
class Dumper {
public:
    Dumper() {
        if (const char * dir = std::getenv("DIALOGUE_SIDON_DUMP_DIR")) {
            dir_ = dir;
            std::filesystem::create_directories(dir_);
        }
    }
    bool enabled() const { return !dir_.empty(); }
    void operator()(const std::string & name, const std::vector<float> & values) const {
        if (!enabled()) return;
        std::ofstream out(dir_ / (name + ".f32"), std::ios::binary);
        out.write(reinterpret_cast<const char *>(values.data()), std::streamsize(values.size() * sizeof(float)));
    }

private:
    std::filesystem::path dir_;
};

std::vector<float> initial_noise(size_t count, uint64_t seed) {
    if (const char * path = std::getenv("DIALOGUE_SIDON_NOISE")) {
        std::ifstream in(path, std::ios::binary);
        std::vector<float> noise(count);
        if (!in.read(reinterpret_cast<char *>(noise.data()), std::streamsize(count * sizeof(float))))
            throw std::runtime_error(std::string("DIALOGUE_SIDON_NOISE has fewer than the needed values: ") + path);
        return noise;
    }
    // The demo samples torch.randn on CUDA; this reproduces that Philox stream for a seed.
    return sampling::generate_torch_cuda_randn(count, seed);
}

// Pearson correlation of 20 ms log-RMS envelopes. The demo correlates raw samples, but each
// chunk is resynthesized from its own noise, so waveforms in the overlap are not phase aligned
// and that score is close to random; who is speaking when survives resynthesis.
float channel_similarity(const float * a, const float * b, size_t count, int sample_rate) {
    const size_t hop = std::max<size_t>(1, size_t(sample_rate / 50));
    const auto envelope = [&](const float * x) {
        std::vector<double> out;
        for (size_t begin = 0; begin + hop <= count; begin += hop) {
            double energy = 0;
            for (size_t i = begin; i < begin + hop; ++i) energy += double(x[i]) * x[i];
            out.push_back(std::log(std::sqrt(energy / double(hop)) + 1.0e-4));
        }
        return out;
    };
    const auto ea = envelope(a), eb = envelope(b);
    if (ea.size() < 2) return 0.0f;
    double mean_a = 0, mean_b = 0;
    for (size_t i = 0; i < ea.size(); ++i) {
        mean_a += ea[i];
        mean_b += eb[i];
    }
    mean_a /= double(ea.size());
    mean_b /= double(eb.size());
    double dot = 0, norm_a = 0, norm_b = 0;
    for (size_t i = 0; i < ea.size(); ++i) {
        const double x = ea[i] - mean_a, y = eb[i] - mean_b;
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    const double denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    return denom > 1e-8 ? float(dot / denom) : 0.0f;
}

class Session final : public runtime::RuntimeSessionBase, public runtime::IOfflineVoiceTaskSession {
public:
    Session(const runtime::SessionOptions & options, std::shared_ptr<const assets::TensorSource> source,
            std::shared_ptr<const model_spec::ModelContract> contract)
        : RuntimeSessionBase(options), contract_(std::move(contract)), runtime_(std::move(source), execution_context()) {}

    std::string family() const override { return "dialogue_sidon"; }
    runtime::VoiceTaskKind task_kind() const override { return runtime::VoiceTaskKind::SourceSeparation; }
    runtime::RunMode run_mode() const override { return runtime::RunMode::Offline; }

    void prepare(const runtime::SessionPreparationRequest & request) override {
        runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, "DialogueSidon");
        if (!request.audio) throw std::runtime_error("DialogueSidon requires input audio");
        mark_prepared();
    }

    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        require_prepared("DialogueSidon run");
        runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, "DialogueSidon");
        if (!request.audio_input) throw std::runtime_error("DialogueSidon requires --audio");
        const int steps = runtime::parse_int_option(request.options, {"num_steps"}).value_or(30);
        if (steps < 1 || steps > 1000) throw std::runtime_error("DialogueSidon num_steps must be in [1, 1000]");
        const uint64_t seed = uint64_t(std::max(0, runtime::parse_int_option(request.options, {"seed"}).value_or(0)));
        const auto normalize_option = runtime::find_option(request.options, {"normalize"});
        const bool normalize = normalize_option ? runtime::parse_bool_option(*normalize_option, "normalize") : true;

        const auto started = std::chrono::steady_clock::now();
        const auto wav = to_mono_16k(*request.audio_input);
        const int out_rate = runtime_.config().sample_rate;
        const int64_t total = int64_t(wav.size());
        std::array<std::vector<float>, 2> stitched;
        if (total <= kChunkSamples) {
            stitched = separate_chunk(wav.data(), size_t(total), steps, seed, true);
        } else {
            // spaces/app.py separate(): 120 s chunks every 110 s; each chunk is cut or padded to the
            // input duration, its speaker order is matched to the previous chunk on the overlap
            // (by envelope, see channel_similarity), and the overlap is linearly crossfaded.
            int64_t previous_end = 0;
            int64_t index = 0;
            for (int64_t start = 0; start < total; start += kChunkSamples - kOverlapSamples, ++index) {
                const int64_t end = std::min(start + kChunkSamples, total);
                auto chunk = separate_chunk(wav.data() + start, size_t(end - start), steps, seed + uint64_t(index), index == 0);
                const int64_t target = std::max<int64_t>(1, int64_t(std::llround(double(end - start) * out_rate / kInputRate)));
                for (auto & channel : chunk) channel.resize(size_t(target), 0.0f);
                if (index == 0) {
                    stitched = std::move(chunk);
                    previous_end = end;
                    continue;
                }
                const int64_t overlap_in = std::max<int64_t>(0, previous_end - start);
                const int64_t overlap = std::max<int64_t>(0, std::min({int64_t(std::llround(double(overlap_in) * out_rate / kInputRate)),
                    int64_t(stitched[0].size()), target}));
                if (overlap > 0) {
                    const size_t tail = stitched[0].size() - size_t(overlap);
                    const auto similarity = [&](int a, int b) {
                        return channel_similarity(stitched[a].data() + tail, chunk[b].data(), size_t(overlap), out_rate);
                    };
                    const float direct = similarity(0, 0) + similarity(1, 1);
                    const float swapped = similarity(0, 1) + similarity(1, 0);
                    debug::timing_log_scalar("dialogue_sidon.overlap_direct", direct);
                    debug::timing_log_scalar("dialogue_sidon.overlap_swapped", swapped);
                    if (swapped > direct) std::swap(chunk[0], chunk[1]);
                    for (int s = 0; s < 2; ++s) {
                        for (int64_t i = 0; i < overlap; ++i) {
                            // torch.linspace(0, 1, overlap)
                            const float fade = overlap > 1 ? float(i) / float(overlap - 1) : 0.0f;
                            auto & value = stitched[s][tail + size_t(i)];
                            value = value * (1.0f - fade) + chunk[s][size_t(i)] * fade;
                        }
                        stitched[s].insert(stitched[s].end(), chunk[s].begin() + overlap, chunk[s].end());
                    }
                } else {
                    for (int s = 0; s < 2; ++s) stitched[s].insert(stitched[s].end(), chunk[s].begin(), chunk[s].end());
                }
                previous_end = end;
            }
        }
        if (normalize) {
            // The demo's audio outputs scale each speaker to a 0.9 peak.
            for (auto & channel : stitched) {
                float peak = 1.0e-6f;
                for (float value : channel) peak = std::max(peak, std::abs(value));
                for (float & value : channel) value = std::clamp(value / peak * 0.9f, -1.0f, 1.0f);
            }
        }

        runtime::TaskResult result;
        runtime::AudioBuffer stereo;
        stereo.sample_rate = out_rate;
        stereo.channels = 2;
        stereo.samples.resize(stitched[0].size() * 2);
        for (size_t i = 0; i < stitched[0].size(); ++i) {
            stereo.samples[2 * i] = stitched[0][i];
            stereo.samples[2 * i + 1] = stitched[1][i];
        }
        result.audio_output = std::move(stereo);
        for (int s = 0; s < 2; ++s) {
            runtime::NamedAudioBuffer named;
            named.id = "speaker" + std::to_string(s + 1);
            named.audio.sample_rate = out_rate;
            named.audio.channels = 1;
            named.audio.samples = std::move(stitched[s]);
            result.named_audio_outputs.push_back(std::move(named));
        }
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
        return result;
    }

private:
    // spaces/app.py _separate_chunk(): one pass over a 16 kHz chunk, returning both speakers at 24 kHz.
    std::array<std::vector<float>, 2> separate_chunk(const float * samples, size_t count, int steps, uint64_t seed, bool first) {
        const Dumper dump;
        const bool dumping = first && dump.enabled();
        const auto log = [&](const std::string & name, const std::vector<float> & values) {
            if (dumping) dump(name, values);
        };
        const auto padded = normalize_and_pad(samples, count);
        log("wav16k_padded", padded);
        modules::Wav2Vec2BertEncoderInput features;
        const auto frontend_ms = debug::measure_ms([&] {
            features = extract_features(padded, execution_context().config().threads);
        });
        log("input_features", features.values);
        log("attention_mask", std::vector<float>(features.attention_mask.begin(), features.attention_mask.end()));
        modules::Wav2Vec2BertEncoderOutput hidden;
        const auto encoder_ms = debug::measure_ms([&] { hidden = runtime_.encode(features); });
        log("features", hidden.values);
        DialogueSidonRuntime::Conditioning conditioning;
        const auto condition_ms = debug::measure_ms([&] { conditioning = runtime_.condition(hidden); });
        log("predicted", conditioning.predicted);
        log("conditioning", conditioning.values);

        const auto & config = runtime_.config();
        const int64_t frames = conditioning.frames;
        const int64_t joint = config.joint_latent();
        auto latents = initial_noise(size_t(frames * joint), seed);
        log("noise", latents);
        DpmSolverScheduler scheduler;
        scheduler.set_timesteps(steps);
        const auto sampler_ms = debug::measure_ms([&] {
            runtime_.begin_sampling(conditioning);
            for (size_t i = 0; i < scheduler.timesteps().size(); ++i) {
                const auto model_output = runtime_.predict(latents, scheduler.timesteps()[i]);
                latents = scheduler.step(model_output, latents);
                if (dumping) {
                    char name[32];
                    std::snprintf(name, sizeof(name), "step_%02zu_", i);
                    log(std::string(name) + "model_output", model_output);
                    log(std::string(name) + "latents", latents);
                }
            }
        });
        const auto & mean = runtime_.latent_mean();
        const auto & stddev = runtime_.latent_std();
        for (int64_t t = 0; t < frames; ++t)
            for (int64_t c = 0; c < joint; ++c) {
                auto & value = latents[size_t(t * joint + c)];
                value = value * stddev[size_t(c)] + mean[size_t(c)];
            }
        log("latents_final", latents);

        std::array<std::vector<float>, 2> speakers;
        const auto decoder_ms = debug::measure_ms([&] {
            for (int s = 0; s < 2; ++s) {
                std::vector<float> latent(size_t(config.latent_dim * frames));
                for (int64_t c = 0; c < config.latent_dim; ++c)
                    for (int64_t t = 0; t < frames; ++t)
                        latent[size_t(c * frames + t)] = latents[size_t(t * joint + s * config.latent_dim + c)];
                speakers[size_t(s)] = runtime_.decode(latent, frames);
                log("wav_speaker" + std::to_string(s), speakers[size_t(s)]);
            }
        });
        debug::timing_log_scalar("dialogue_sidon.frontend.ms", frontend_ms);
        debug::timing_log_scalar("dialogue_sidon.encoder.ms", encoder_ms);
        debug::timing_log_scalar("dialogue_sidon.condition.ms", condition_ms);
        debug::timing_log_scalar("dialogue_sidon.sampler.ms", sampler_ms);
        debug::timing_log_scalar("dialogue_sidon.decoder.ms", decoder_ms);
        debug::timing_log_scalar("dialogue_sidon.frames", double(frames));
        return speakers;
    }

    std::shared_ptr<const model_spec::ModelContract> contract_;
    DialogueSidonRuntime runtime_;
};

}  // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_dialogue_sidon_loader() {
    runtime::SpecBackedVoiceModelConfig<Assets> config;
    config.family = "dialogue_sidon";
    config.load_assets = [](const std::filesystem::path & path) {
        auto assets = std::make_shared<Assets>();
        assets->resources = model_spec::load_resource_bundle_for_family(path, "dialogue_sidon");
        return assets;
    };
    config.create_session = [](const runtime::TaskSpec & task, const runtime::SessionOptions & options,
                               std::shared_ptr<const Assets> assets,
                               std::shared_ptr<const model_spec::ModelContract> contract) {
        if (task.task != runtime::VoiceTaskKind::SourceSeparation || task.mode != runtime::RunMode::Offline)
            throw std::runtime_error("DialogueSidon supports offline two-speaker separation (--task sep)");
        runtime::validate_spec_backed_session_options(options, *contract, "dialogue_sidon", "DialogueSidon");
        return std::make_unique<Session>(options, assets->resources.open_tensor_source("weights"), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::community_models::dialogue_sidon
