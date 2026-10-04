#pragma once

#include "engine/framework/runtime/model.h"

#include <memory>

namespace engine::community_models::dialogue_sidon {

std::shared_ptr<runtime::IVoiceModelLoader> make_dialogue_sidon_loader();

}  // namespace engine::community_models::dialogue_sidon
