#include "engine/community_models/dialogue_sidon/scheduler.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace engine::community_models::dialogue_sidon {
namespace {

struct AlphaSigma {
    float alpha;
    float sigma;
    float lambda;
};

AlphaSigma alpha_sigma(float sigma) {
    const float alpha = 1.0f / std::sqrt(sigma * sigma + 1.0f);
    const float sigma_t = sigma * alpha;
    return {alpha, sigma_t, std::log(alpha) - std::log(sigma_t)};
}

}  // namespace

DpmSolverScheduler::DpmSolverScheduler(DpmSolverConfig config) {
    if (config.train_timesteps < 2) throw std::runtime_error("DialogueSidon scheduler needs at least two train steps");
    // torch.linspace(beta_start, beta_end, N, dtype=float32) and a float32 cumprod.
    float cumulative = 1.0f;
    alphas_cumprod_.reserve(size_t(config.train_timesteps));
    for (int64_t i = 0; i < config.train_timesteps; ++i) {
        const float beta = float(double(config.beta_start) +
            (double(config.beta_end) - double(config.beta_start)) * double(i) / double(config.train_timesteps - 1));
        cumulative *= 1.0f - beta;
        alphas_cumprod_.push_back(cumulative);
    }
}

void DpmSolverScheduler::set_timesteps(int64_t steps) {
    const int64_t train = int64_t(alphas_cumprod_.size());
    if (steps <= 0 || steps > train) throw std::runtime_error("DialogueSidon scheduler step count is out of range");
    // np.linspace(0, train - 1, steps + 1).round()[::-1][:-1]: numpy evaluates i * step in
    // float64 and rounds half to even, which differs from (train - 1) * i / steps at i = 15 of 30.
    const double spacing = double(train - 1) / double(steps);
    timesteps_.clear();
    sigmas_.clear();
    for (int64_t i = steps; i >= 1; --i) {
        const int64_t t = i == steps ? train - 1 : int64_t(std::nearbyint(double(i) * spacing));
        timesteps_.push_back(t);
        const float alpha = alphas_cumprod_[size_t(t)];
        sigmas_.push_back(std::sqrt((1.0f - alpha) / alpha));
    }
    sigmas_.push_back(0.0f);
    previous_x0_.clear();
    step_index_ = 0;
    lower_order_nums_ = 0;
}

std::vector<float> DpmSolverScheduler::step(const std::vector<float> & model_output, const std::vector<float> & sample) {
    if (step_index_ >= int64_t(timesteps_.size())) throw std::runtime_error("DialogueSidon scheduler ran past its last step");
    if (model_output.size() != sample.size()) throw std::runtime_error("DialogueSidon scheduler size mismatch");
    const size_t index = size_t(step_index_);
    const auto s0 = alpha_sigma(sigmas_[index]);
    // v-prediction to the data estimate (dpmsolver++ converts every model output to x0).
    std::vector<float> x0(sample.size());
    for (size_t i = 0; i < sample.size(); ++i) x0[i] = s0.alpha * sample[i] - s0.sigma * model_output[i];

    std::vector<float> out(sample.size());
    const bool last = step_index_ == int64_t(timesteps_.size()) - 1;
    if (last) {
        // final_sigmas_type "zero": sigma_t = 0, so the first-order update returns x0 exactly.
        out = x0;
    } else {
        const auto t = alpha_sigma(sigmas_[index + 1]);
        const float h = t.lambda - s0.lambda;
        const float sample_coeff = t.sigma / s0.sigma;
        const float x0_coeff = -t.alpha * (std::exp(-h) - 1.0f);
        if (lower_order_nums_ < 1) {
            for (size_t i = 0; i < sample.size(); ++i) out[i] = sample_coeff * sample[i] + x0_coeff * x0[i];
        } else {
            const auto s1 = alpha_sigma(sigmas_[index - 1]);
            const float r0 = (s0.lambda - s1.lambda) / h;
            for (size_t i = 0; i < sample.size(); ++i) {
                const float d1 = (1.0f / r0) * (x0[i] - previous_x0_[i]);
                out[i] = sample_coeff * sample[i] + x0_coeff * x0[i] + 0.5f * x0_coeff * d1;
            }
        }
    }
    previous_x0_ = std::move(x0);
    lower_order_nums_ = std::min<int64_t>(lower_order_nums_ + 1, 2);
    ++step_index_;
    return out;
}

}  // namespace engine::community_models::dialogue_sidon
