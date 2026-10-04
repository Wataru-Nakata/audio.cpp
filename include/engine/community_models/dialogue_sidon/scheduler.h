#pragma once

#include <cstdint>
#include <vector>

namespace engine::community_models::dialogue_sidon {

struct DpmSolverConfig {
    int64_t train_timesteps = 1000;
    float beta_start = 1.0e-4f;
    float beta_end = 0.02f;
};

// diffusers DPMSolverMultistepScheduler as the demo configures it: dpmsolver++, solver order 2,
// linspace spacing, v_prediction, final_sigmas_type "zero" (first-order last step), on a linear
// beta schedule. VibeVoice's scheduler is the same solver but only accepts cosine betas.
class DpmSolverScheduler {
public:
    explicit DpmSolverScheduler(DpmSolverConfig config = {});

    void set_timesteps(int64_t steps);
    const std::vector<int64_t> & timesteps() const noexcept { return timesteps_; }
    // Advances one step: model_output is the head's v-prediction for sample at timesteps()[step].
    std::vector<float> step(const std::vector<float> & model_output, const std::vector<float> & sample);

private:
    std::vector<float> alphas_cumprod_;
    std::vector<int64_t> timesteps_;
    std::vector<float> sigmas_;
    std::vector<float> previous_x0_;
    int64_t step_index_ = 0;
    int64_t lower_order_nums_ = 0;
};

}  // namespace engine::community_models::dialogue_sidon
