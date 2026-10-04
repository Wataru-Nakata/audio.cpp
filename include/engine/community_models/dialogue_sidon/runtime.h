#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/speech_encoders/wav2vec2_bert_encoder.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::community_models::dialogue_sidon {

struct DialogueSidonConfig {
    int64_t encoder_hidden = 1024;
    int64_t encoder_layers = 13;
    int64_t latent_dim = 32;      // per speaker
    int64_t hidden_size = 768;    // diffusion head
    int64_t num_layers = 8;
    int64_t num_heads = 12;       // not stored in the weights; the export reshapes to (12, 64)
    int64_t ffn_size = 3072;
    int64_t frequency_size = 256;
    float rope_theta = 10000.0f;
    float layer_norm_eps = 1.0e-6f;
    int64_t decoder_channels = 1536;
    std::vector<int> decoder_strides;
    int64_t hop_length = 480;
    int sample_rate = 24000;

    int64_t joint_latent() const noexcept { return 2 * latent_dim; }
    int64_t cond_size() const noexcept { return encoder_hidden + joint_latent(); }
    // Transposed convs use padding ceil(stride / 2) without output_padding, so each stage
    // loses (stride mod 2) samples: 480 * T - 13 for strides 8, 5, 4, 3.
    int64_t decoded_samples(int64_t frames) const noexcept;
};

class DialogueSidonRuntime {
public:
    DialogueSidonRuntime(std::shared_ptr<const assets::TensorSource> source, core::ExecutionContext & execution);
    ~DialogueSidonRuntime();

    const DialogueSidonConfig & config() const noexcept;

    // w2v-BERT last_hidden_state [frames, 1024]. A masked final frame (odd fbank count) runs the
    // encoder with key/conv masking, which the export applies through its attention_mask input.
    modules::Wav2Vec2BertEncoderOutput encode(const modules::Wav2Vec2BertEncoderInput & input);

    // Speaker heads on the encoder features: predicted latents [frames, 64] (pred0 | pred1) and the
    // diffusion conditioning [frames, 1088] = normalize(predicted) | features.
    struct Conditioning {
        std::vector<float> predicted;
        std::vector<float> values;
        int64_t frames = 0;
    };
    Conditioning condition(const modules::Wav2Vec2BertEncoderOutput & features);

    // Binds the conditioning for the following predict() calls (one graph per frame count).
    void begin_sampling(const Conditioning & conditioning);
    // One diffusion-head evaluation: v-prediction [frames, 64] for normalized latents [frames, 64].
    std::vector<float> predict(const std::vector<float> & latents, int64_t timestep);

    // Latent normalization statistics (64 values, speaker 0 then speaker 1).
    const std::vector<float> & latent_mean() const noexcept;
    const std::vector<float> & latent_std() const noexcept;

    // One speaker's channel-major latent [32, frames] to 24 kHz audio.
    std::vector<float> decode(const std::vector<float> & latent, int64_t frames);

private:
    struct State;
    std::unique_ptr<State> state_;
};

}  // namespace engine::community_models::dialogue_sidon
