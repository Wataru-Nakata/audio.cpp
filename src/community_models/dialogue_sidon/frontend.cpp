#include "engine/community_models/dialogue_sidon/frontend.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/kaldi_fbank.h"
#include "engine/framework/audio/resampling.h"

#include <algorithm>
#include <cmath>
#include <future>
#include <stdexcept>

namespace engine::community_models::dialogue_sidon {

std::vector<float> to_mono_16k(const runtime::AudioBuffer & audio) {
    if (audio.sample_rate <= 0 || audio.channels <= 0 || audio.samples.empty())
        throw std::runtime_error("DialogueSidon requires non-empty audio with a valid sample rate and channel count");
    auto mono = audio::mixdown_interleaved_to_mono_average(audio.samples, audio.channels);
    if (audio.sample_rate != 16000)
        mono = audio::resample_mono_torchaudio_sinc_hann(mono, audio.sample_rate, 16000);
    return mono;
}

std::vector<float> normalize_and_pad(const float * samples, size_t count) {
    float peak = 0.0f;
    for (size_t i = 0; i < count; ++i) peak = std::max(peak, std::abs(samples[i]));
    peak = std::max(peak, 1.0e-6f);
    std::vector<float> padded(count + 320, 0.0f);
    for (size_t i = 0; i < count; ++i) padded[i + 160] = 0.9f * samples[i] / peak;
    return padded;
}

modules::Wav2Vec2BertEncoderInput extract_features(const std::vector<float> & padded_audio, int threads) {
    audio::KaldiFbankOptions options;
    options.window_type = audio::KaldiFbankWindowType::Povey;
    options.lfr_m = 1;
    options.lfr_n = 1;
    if (padded_audio.size() < 400) throw std::runtime_error("DialogueSidon frontend needs at least one mel frame");
    const int frames = int((padded_audio.size() - 400) / 160 + 1);
    if (frames < 2) throw std::runtime_error("DialogueSidon frontend needs at least two mel frames");
    const int workers = std::clamp(threads, 1, std::max(1, frames / 128));
    audio::KaldiFbankFeatures features;
    if (workers == 1) {
        features = audio::extract_kaldi_fbank(padded_audio, options);
    } else {
        std::vector<std::future<audio::KaldiFbankFeatures>> pending;
        for (int worker = 0; worker < workers; ++worker) {
            const int begin = frames * worker / workers;
            const int end = frames * (worker + 1) / workers;
            pending.push_back(std::async(std::launch::async, [&, begin, end] {
                // Frames are independent; keep the last window's full 400 samples.
                std::vector<float> part(padded_audio.begin() + begin * 160,
                    padded_audio.begin() + (end - 1) * 160 + 400);
                return audio::extract_kaldi_fbank(part, options);
            }));
        }
        features.frames = frames;
        features.feature_dim = 80;
        features.values.reserve(size_t(frames) * 80);
        for (auto & task : pending) {
            auto part = task.get();
            features.values.insert(features.values.end(), part.values.begin(), part.values.end());
        }
    }
    if (features.frames != frames) throw std::runtime_error("DialogueSidon fbank frame count mismatch");
    for (int channel = 0; channel < 80; ++channel) {
        double mean = 0;
        for (int t = 0; t < frames; ++t) mean += features.values[t * 80 + channel];
        mean /= frames;
        double variance = 0;
        for (int t = 0; t < frames; ++t) {
            const double centered = features.values[t * 80 + channel] - mean;
            variance += centered * centered;
        }
        const double stddev = std::sqrt(variance / (frames - 1) + 1e-5);
        for (int t = 0; t < frames; ++t)
            features.values[t * 80 + channel] = float((features.values[t * 80 + channel] - mean) / stddev);
    }
    modules::Wav2Vec2BertEncoderInput output;
    output.frames = (frames + 1) / 2;
    output.dims = 160;
    output.values = std::move(features.values);
    output.values.resize(size_t(output.frames) * 160, 0.0f);
    output.attention_mask.assign(output.frames, 1);
    if (frames % 2) output.attention_mask.back() = 0;
    return output;
}

}  // namespace engine::community_models::dialogue_sidon
