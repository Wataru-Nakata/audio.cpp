#pragma once

#include "engine/framework/modules/speech_encoders/wav2vec2_bert_encoder.h"
#include "engine/framework/runtime/session.h"

#include <cstddef>
#include <vector>

namespace engine::community_models::dialogue_sidon {

// Mono mixdown at 16 kHz, as the demo feeds the encoder (channel mean, then torchaudio resample).
std::vector<float> to_mono_16k(const runtime::AudioBuffer & audio);

// Peak-normalizes one chunk to 0.9 and pads 160 zero samples on both sides.
std::vector<float> normalize_and_pad(const float * samples, size_t count);

// Kaldi fbank (Povey, no rescale), per-bin CMVN with sqrt(var + 1e-5), stride-2 stacking.
// An odd fbank frame count gets a zero frame whose stacked frame is masked out, as the
// demo's pad_to_multiple_of=2 batch padding does.
modules::Wav2Vec2BertEncoderInput extract_features(const std::vector<float> & padded_audio, int threads);

}  // namespace engine::community_models::dialogue_sidon
