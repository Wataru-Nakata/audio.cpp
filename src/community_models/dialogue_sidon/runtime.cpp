#include "engine/community_models/dialogue_sidon/runtime.h"

#include "engine/framework/core/attention_fallback.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>

namespace engine::community_models::dialogue_sidon {
namespace {

constexpr size_t kContextBytes = 16 * 1024 * 1024;
constexpr const char * kHead = "diffusion_head.";

struct ContextDeleter {
    void operator()(ggml_context * context) const { ggml_free(context); }
};

struct AllocatorDeleter {
    void operator()(ggml_gallocr_t allocator) const { ggml_gallocr_free(allocator); }
};

// One reusable ggml graph: inputs are re-uploaded before every compute because the
// allocator may reuse their buffers for intermediates.
class Graph {
public:
    Graph(core::ExecutionContext & execution, const char * label)
        : execution_(execution), label_(label), context_(ggml_init({kContextBytes, nullptr, true})) {
        if (!context_) throw std::runtime_error(std::string(label) + " context allocation failed");
    }
    Graph(const Graph &) = delete;
    Graph & operator=(const Graph &) = delete;
    ~Graph() {
        if (graph_) core::release_backend_graph_resources(execution_.backend(), graph_, true);
    }

    core::ModuleBuildContext build_context() { return {context_.get(), label_, execution_.backend_type()}; }

    core::TensorValue input(ggml_type type, std::initializer_list<int64_t> dims) {
        auto ctx = build_context();
        auto value = core::make_tensor(ctx, type, core::TensorShape::from_dims(dims));
        ggml_set_input(value.tensor);
        return value;
    }

    void finalize(const std::vector<ggml_tensor *> & outputs, size_t nodes) {
        graph_ = ggml_new_graph_custom(context_.get(), nodes, false);
        for (auto * output : outputs) {
            ggml_set_output(output);
            ggml_build_forward_expand(graph_, output);
        }
        core::validate_backend_graph_supported(execution_.backend(), graph_, label_);
        allocator_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend())));
        if (!ggml_gallocr_alloc_graph(allocator_.get(), graph_))
            throw std::runtime_error(std::string(label_) + " graph allocation failed");
        core::prepare_host_graph_plan(execution_, graph_, plan_);
    }

    void compute() {
        if (core::compute_graph(execution_, graph_, plan_, label_) != GGML_STATUS_SUCCESS)
            throw std::runtime_error(std::string(label_) + " compute failed");
        ggml_backend_synchronize(execution_.backend());
    }

private:
    core::ExecutionContext & execution_;
    const char * label_;
    std::unique_ptr<ggml_context, ContextDeleter> context_;
    std::unique_ptr<ggml_gallocr, AllocatorDeleter> allocator_;
    core::HostGraphPlan plan_;
    ggml_cgraph * graph_ = nullptr;
};

core::TensorValue contiguous(core::ModuleBuildContext & ctx, const core::TensorValue & value) {
    return core::ensure_backend_addressable_layout(ctx, value);
}

core::TensorValue wrap(ggml_tensor * tensor, const core::TensorShape & shape) {
    return core::wrap_tensor(tensor, shape, GGML_TYPE_F32);
}

// Slice `index` of `count` equal parts along the last axis of a [1, T, count * width] tensor.
core::TensorValue chunk(core::ModuleBuildContext & ctx, const core::TensorValue & value, int64_t index, int64_t width) {
    auto * view = ggml_view_3d(ctx.ggml, value.tensor, width, value.tensor->ne[1], value.tensor->ne[2],
        value.tensor->nb[1], value.tensor->nb[2], size_t(index * width) * ggml_element_size(value.tensor));
    return wrap(ggml_cont(ctx.ggml, view), core::TensorShape::from_dims({value.shape.dims[0], value.shape.dims[1], width}));
}

// DiT _modulate(norm(x), shift, scale) = norm(x) * (1 + scale) + shift with a non-affine LayerNorm.
core::TensorValue modulate(core::ModuleBuildContext & ctx, const core::TensorValue & x, const core::TensorValue & shift,
                           const core::TensorValue & scale, float eps) {
    auto normed = modules::LayerNormModule({x.shape.last_dim(), eps, false, false}).build(ctx, x, modules::NormWeights{});
    auto * scaled = ggml_mul(ctx.ggml, normed.tensor, ggml_scale_bias(ctx.ggml, scale.tensor, 1.0f, 1.0f));
    return wrap(ggml_add(ctx.ggml, scaled, shift.tensor), x.shape);
}

core::TensorValue gated_residual(core::ModuleBuildContext & ctx, const core::TensorValue & x, const core::TensorValue & gate,
                                 const core::TensorValue & h) {
    return wrap(ggml_add(ctx.ggml, x.tensor, ggml_mul(ctx.ggml, gate.tensor, h.tensor)), x.shape);
}

struct DiTBlockWeights {
    modules::LinearWeights adaln, q, k, v, out, fc1, fc2;
};

struct DiTWeights {
    modules::LinearWeights latent_proj, cond_proj, t_fc1, t_fc2, final_adaln, final_linear;
    std::vector<DiTBlockWeights> blocks;
};

struct DecoderWeights {
    std::map<std::string, core::TensorValue> tensors;
    const core::TensorValue & at(const std::string & name) const {
        auto it = tensors.find(name);
        if (it == tensors.end()) throw std::runtime_error("DialogueSidon decoder tensor missing: " + name);
        return it->second;
    }
};

// Mirrors the framework's Wav2Vec2-BERT loader so the plain and the masked encoder graphs can
// share one weight store (the loader itself is internal to the framework module).
std::shared_ptr<modules::Wav2Vec2BertEncoderWeights> load_encoder_weights(
    const assets::TensorSource & source, core::ExecutionContext & execution, const modules::Wav2Vec2BertEncoderConfig & config) {
    using assets::TensorStorageType;
    namespace binding = modules::binding;
    auto weights = std::make_shared<modules::Wav2Vec2BertEncoderWeights>();
    weights->config = config;
    weights->store = std::make_shared<core::BackendWeightStore>(
        execution.backend(), execution.backend_type(), "dialogue_sidon.encoder", kContextBytes);
    auto & store = *weights->store;
    const auto type = TensorStorageType::F32;
    const int64_t hidden = config.hidden_size;
    const int64_t inner = config.intermediate_size;
    weights->feature_norm = binding::norm_from_source(store, source, "feature_projection.layer_norm", config.feature_dim);
    weights->feature_projection = binding::linear_from_source(
        store, source, "feature_projection.projection", type, hidden, config.feature_dim, true);
    for (int64_t index = 0; index < config.num_hidden_layers; ++index) {
        const std::string prefix = "encoder.layers." + std::to_string(index) + ".";
        modules::Wav2Vec2BertLayerWeights layer;
        layer.ffn1_norm = binding::norm_from_source(store, source, prefix + "ffn1_layer_norm", hidden);
        layer.ffn1_in = binding::linear_from_source(store, source, prefix + "ffn1.intermediate_dense", type, inner, hidden, true);
        layer.ffn1_out = binding::linear_from_source(store, source, prefix + "ffn1.output_dense", type, hidden, inner, true);
        layer.self_attn_norm = binding::norm_from_source(store, source, prefix + "self_attn_layer_norm", hidden);
        for (auto [target, name] : {std::pair{&layer.self_attn.q, "q"}, std::pair{&layer.self_attn.k, "k"},
                                    std::pair{&layer.self_attn.v, "v"}, std::pair{&layer.self_attn.out, "out"}})
            *target = binding::linear_from_source(store, source, prefix + "self_attn.linear_" + name, type, hidden, hidden, true);
        layer.self_attn.distance_embedding = store.load_tensor(source, prefix + "self_attn.distance_embedding.weight", type,
            {config.relative_positions, hidden / config.num_attention_heads});
        layer.conv.layer_norm = binding::norm_from_source(store, source, prefix + "conv_module.layer_norm", hidden);
        layer.conv.pointwise_in = binding::conv1d_from_source(
            store, source, prefix + "conv_module.pointwise_conv1", type, 2 * hidden, hidden, 1, false);
        layer.conv.depthwise = binding::depthwise_conv1d_from_source(
            store, source, prefix + "conv_module.depthwise_conv", type, hidden, config.conv_kernel, false);
        layer.conv.depthwise_layer_norm = binding::norm_from_source(store, source, prefix + "conv_module.depthwise_layer_norm", hidden);
        layer.conv.pointwise_out = binding::conv1d_from_source(
            store, source, prefix + "conv_module.pointwise_conv2", type, hidden, hidden, 1, false);
        layer.ffn2_norm = binding::norm_from_source(store, source, prefix + "ffn2_layer_norm", hidden);
        layer.ffn2_in = binding::linear_from_source(store, source, prefix + "ffn2.intermediate_dense", type, inner, hidden, true);
        layer.ffn2_out = binding::linear_from_source(store, source, prefix + "ffn2.output_dense", type, hidden, inner, true);
        layer.final_norm = binding::norm_from_source(store, source, prefix + "final_layer_norm", hidden);
        weights->layers.push_back(std::move(layer));
    }
    weights->half_scale = store.make_f32(core::TensorShape::from_dims({hidden}), std::vector<float>(size_t(hidden), 0.5f));
    weights->relative_scale = store.make_f32(core::TensorShape::from_dims({config.relative_positions}),
        std::vector<float>(size_t(config.relative_positions),
            1.0f / std::sqrt(float(hidden / config.num_attention_heads))));
    weights->left_conv_pad = store.make_f32(core::TensorShape::from_dims({1, hidden, config.conv_kernel - 1}),
        std::vector<float>(size_t(hidden * (config.conv_kernel - 1)), 0.0f));
    store.upload();
    return weights;
}

}  // namespace

int64_t DialogueSidonConfig::decoded_samples(int64_t frames) const noexcept {
    int64_t length = frames;
    for (int stride : decoder_strides) length = length * stride - (stride % 2);
    return length;
}

struct DialogueSidonRuntime::State {
    core::ExecutionContext & execution;
    DialogueSidonConfig config;
    core::BackendWeightStore store;
    modules::LinearWeights head0, head1;
    core::TensorValue latent_mean_tensor, latent_std_tensor;
    std::vector<float> latent_mean, latent_std, time_freqs;
    DiTWeights dit;
    DecoderWeights decoder;
    std::unique_ptr<modules::Wav2Vec2BertEncoderComponent> encoder, masked_encoder;
    bool flash = true;

    struct ConditionGraph {
        Graph graph;
        int64_t frames;
        core::TensorValue features, predicted, cond;
        ConditionGraph(core::ExecutionContext & execution, int64_t t) : graph(execution, "dialogue_sidon.condition"), frames(t) {}
    };
    std::unique_ptr<ConditionGraph> condition_graph;

    struct DiTGraph {
        Graph graph;
        int64_t frames;
        core::TensorValue latents, cond, time, positions, output;
        DiTGraph(core::ExecutionContext & execution, int64_t t) : graph(execution, "dialogue_sidon.dit"), frames(t) {}
    };
    std::unique_ptr<DiTGraph> dit_graph;
    std::vector<float> bound_cond;
    std::vector<int32_t> positions;

    struct DecoderGraph {
        Graph graph;
        int64_t frames;
        core::TensorValue latent, audio;
        DecoderGraph(core::ExecutionContext & execution, int64_t t) : graph(execution, "dialogue_sidon.decoder"), frames(t) {}
    };
    std::unique_ptr<DecoderGraph> decoder_graph;

    State(std::shared_ptr<const assets::TensorSource> source, core::ExecutionContext & context)
        : execution(context), store(context.backend(), context.backend_type(), "dialogue_sidon.weights", kContextBytes) {
        using assets::TensorStorageType;
        namespace binding = modules::binding;
        const auto & src = *source;
        std::map<std::string, std::vector<int64_t>> shapes;
        for (const auto & tensor : src.tensors()) shapes[tensor.name] = tensor.shape;
        const auto shape_of = [&](const std::string & name) -> const std::vector<int64_t> & {
            auto it = shapes.find(name);
            if (it == shapes.end()) throw std::runtime_error("DialogueSidon weights are missing " + name);
            return it->second;
        };

        // Sizes come from the tensors; the head count only exists in the export graph.
        config.latent_dim = shape_of("output_linear1.weight")[0];
        config.encoder_hidden = shape_of("output_linear1.weight")[1];
        config.hidden_size = shape_of(std::string(kHead) + "latent_proj.weight")[0];
        config.ffn_size = shape_of(std::string(kHead) + "blocks.0.ffn.fc1.weight")[0];
        config.frequency_size = shape_of(std::string(kHead) + "t_embedder.fc1.weight")[1];
        config.num_heads = config.hidden_size / 64;
        config.num_layers = 0;
        while (shapes.count(std::string(kHead) + "blocks." + std::to_string(config.num_layers) + ".adaln.weight"))
            ++config.num_layers;
        config.encoder_layers = 0;
        while (shapes.count("encoder.layers." + std::to_string(config.encoder_layers) + ".final_layer_norm.weight"))
            ++config.encoder_layers;
        if (shape_of(std::string(kHead) + "latent_proj.weight")[1] != config.joint_latent() ||
            shape_of(std::string(kHead) + "cond_proj.weight")[1] != config.cond_size())
            throw std::runtime_error("DialogueSidon diffusion head does not match the speaker heads");
        config.decoder_channels = shape_of("decoder.input.weight")[0];
        if (shape_of("decoder.input.weight")[1] != config.latent_dim)
            throw std::runtime_error("DialogueSidon decoder input does not match the latent size");
        for (int stage = 0; shapes.count("decoder.stages." + std::to_string(stage) + ".upsample.weight"); ++stage)
            config.decoder_strides.push_back(int(shape_of("decoder.stages." + std::to_string(stage) + ".upsample.weight")[2] / 2));
        config.hop_length = 1;
        for (int stride : config.decoder_strides) config.hop_length *= stride;

        const auto type = TensorStorageType::F32;
        const int64_t H = config.hidden_size;
        head0 = binding::linear_from_source(store, src, "output_linear1", type, config.latent_dim, config.encoder_hidden, true);
        head1 = binding::linear_from_source(store, src, "output_linear2", type, config.latent_dim, config.encoder_hidden, true);
        latent_mean = src.require_f32("latent_norm.mean", {config.joint_latent()});
        latent_std = src.require_f32("latent_norm.std", {config.joint_latent()});
        latent_mean_tensor = store.make_f32(core::TensorShape::from_dims({config.joint_latent()}), latent_mean);
        latent_std_tensor = store.make_f32(core::TensorShape::from_dims({config.joint_latent()}), latent_std);
        time_freqs = src.require_f32(std::string(kHead) + "t_embedder.freqs", {config.frequency_size / 2});

        const auto head = [&](const std::string & name, int64_t out, int64_t in, bool bias) {
            return binding::linear_from_source(store, src, kHead + name, type, out, in, bias);
        };
        dit.latent_proj = head("latent_proj", H, config.joint_latent(), false);
        dit.cond_proj = head("cond_proj", H, config.cond_size(), false);
        dit.t_fc1 = head("t_embedder.fc1", H, config.frequency_size, false);
        dit.t_fc2 = head("t_embedder.fc2", H, H, false);
        dit.final_adaln = head("final_layer.adaln", 2 * H, H, true);
        dit.final_linear = head("final_layer.linear", config.joint_latent(), H, false);
        for (int64_t layer = 0; layer < config.num_layers; ++layer) {
            const std::string prefix = "blocks." + std::to_string(layer) + ".";
            dit.blocks.push_back({head(prefix + "adaln", 6 * H, H, true), head(prefix + "q_proj", H, H, true),
                head(prefix + "k_proj", H, H, true), head(prefix + "v_proj", H, H, true),
                head(prefix + "out_proj", H, H, true), head(prefix + "ffn.fc1", config.ffn_size, H, true),
                head(prefix + "ffn.fc2", H, config.ffn_size, true)});
        }
        for (const auto & [name, shape] : shapes)
            if (name.rfind("decoder.", 0) == 0)
                decoder.tensors.emplace(name, store.load_tensor(src, name, TensorStorageType::Native, shape));
        store.upload();

        modules::Wav2Vec2BertEncoderConfig encoder_config;
        encoder_config.hidden_size = config.encoder_hidden;
        encoder_config.num_hidden_layers = config.encoder_layers;
        encoder_config.output_hidden_layer = config.encoder_layers;
        encoder_config.relative_positions = shape_of("encoder.layers.0.self_attn.distance_embedding.weight")[0];
        encoder_config.apply_semantic_normalization = false;
        encoder_config.mask_padded_frames = false;
        encoder_config.project_relative_keys_first = true;
        encoder_config.pointwise_conv_as_linear = true;
        auto plain = load_encoder_weights(src, context, encoder_config);
        auto masked = std::make_shared<modules::Wav2Vec2BertEncoderWeights>(*plain);
        masked->config.mask_padded_frames = true;
        encoder = std::make_unique<modules::Wav2Vec2BertEncoderComponent>(plain, context, kContextBytes);
        masked_encoder = std::make_unique<modules::Wav2Vec2BertEncoderComponent>(masked, context, kContextBytes);
        source->release_storage();

        flash = core::resolve_flash_attention(context.backend(), H / config.num_heads, core::AttentionPreference::Auto);
    }

    void build_condition_graph(int64_t frames) {
        condition_graph.reset();
        condition_graph = std::make_unique<ConditionGraph>(execution, frames);
        auto & g = *condition_graph;
        g.features = g.graph.input(GGML_TYPE_F32, {1, frames, config.encoder_hidden});
        auto ctx = g.graph.build_context();
        const modules::LinearModule speaker({config.encoder_hidden, config.latent_dim, true, GGML_PREC_F32});
        auto pred0 = speaker.build(ctx, g.features, head0);
        auto pred1 = speaker.build(ctx, g.features, head1);
        g.predicted = modules::ConcatModule({2}).build(ctx, pred0, pred1);
        auto normalized = wrap(ggml_div(ctx.ggml, ggml_sub(ctx.ggml, g.predicted.tensor, latent_mean_tensor.tensor),
            latent_std_tensor.tensor), g.predicted.shape);
        g.cond = modules::ConcatModule({2}).build(ctx, normalized, g.features);
        g.graph.finalize({g.predicted.tensor, g.cond.tensor}, 1024);
    }

    void build_dit_graph(int64_t frames) {
        dit_graph.reset();
        dit_graph = std::make_unique<DiTGraph>(execution, frames);
        auto & g = *dit_graph;
        const int64_t H = config.hidden_size;
        const int64_t heads = config.num_heads;
        const int64_t head_dim = H / heads;
        const float eps = config.layer_norm_eps;
        g.latents = g.graph.input(GGML_TYPE_F32, {1, frames, config.joint_latent()});
        g.cond = g.graph.input(GGML_TYPE_F32, {1, frames, config.cond_size()});
        g.time = g.graph.input(GGML_TYPE_F32, {1, 1, config.frequency_size});
        g.positions = g.graph.input(GGML_TYPE_I32, {frames});
        auto ctx = g.graph.build_context();
        const auto linear = [&](const core::TensorValue & x, const modules::LinearWeights & w, int64_t in, int64_t out) {
            return modules::LinearModule({in, out, w.bias.has_value(), GGML_PREC_F32}).build(ctx, x, w);
        };
        auto x = linear(g.latents, dit.latent_proj, config.joint_latent(), H);
        auto t = linear(g.time, dit.t_fc1, config.frequency_size, H);
        t = linear(modules::SiluModule{}.build(ctx, t), dit.t_fc2, H, H);
        auto c = linear(g.cond, dit.cond_proj, config.cond_size(), H);
        c = wrap(ggml_add(ctx.ggml, c.tensor, t.tensor), c.shape);
        const auto c_act = modules::SiluModule{}.build(ctx, c);

        const modules::RoPEModule rope({head_dim, GGML_ROPE_TYPE_NEOX, config.rope_theta});
        modules::ScaledDotProductAttentionConfig attention;
        attention.head_dim = head_dim;
        attention.lowering = flash ? modules::ScaledDotProductAttentionLowering::Flash
                                   : modules::ScaledDotProductAttentionLowering::Explicit;
        attention.precision = GGML_PREC_F32;
        attention.causality = modules::AttentionCausality::NonCausal;
        const auto heads_of = [&](const core::TensorValue & value, bool rotate) {
            auto split = core::reshape_tensor(ctx, value, core::TensorShape::from_dims({1, frames, heads, head_dim}));
            if (rotate) split = rope.build(ctx, split, g.positions);
            return contiguous(ctx, modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, split));
        };
        for (const auto & block : dit.blocks) {
            auto modulation = linear(c_act, block.adaln, H, 6 * H);
            auto h = modulate(ctx, x, chunk(ctx, modulation, 0, H), chunk(ctx, modulation, 1, H), eps);
            auto q = heads_of(linear(h, block.q, H, H), true);
            auto k = heads_of(linear(h, block.k, H, H), true);
            auto v = heads_of(linear(h, block.v, H, H), false);
            auto context = modules::ScaledDotProductAttentionModule(attention).build(ctx, q, k, v);
            context = core::reshape_tensor(ctx, contiguous(ctx, context), core::TensorShape::from_dims({1, frames, H}));
            x = gated_residual(ctx, x, chunk(ctx, modulation, 2, H), linear(context, block.out, H, H));
            h = modulate(ctx, x, chunk(ctx, modulation, 3, H), chunk(ctx, modulation, 4, H), eps);
            h = linear(modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx,
                linear(h, block.fc1, H, config.ffn_size)), block.fc2, config.ffn_size, H);
            x = gated_residual(ctx, x, chunk(ctx, modulation, 5, H), h);
        }
        auto modulation = linear(c_act, dit.final_adaln, H, 2 * H);
        x = modulate(ctx, x, chunk(ctx, modulation, 0, H), chunk(ctx, modulation, 1, H), eps);
        g.output = linear(x, dit.final_linear, H, config.joint_latent());
        g.graph.finalize({g.output.tensor}, 8192);
        positions.resize(size_t(frames));
        for (int64_t i = 0; i < frames; ++i) positions[size_t(i)] = int32_t(i);
    }

    void build_decoder_graph(int64_t frames) {
        decoder_graph.reset();
        decoder_graph = std::make_unique<DecoderGraph>(execution, frames);
        auto & g = *decoder_graph;
        g.latent = g.graph.input(GGML_TYPE_F32, {1, config.latent_dim, frames});
        auto ctx = g.graph.build_context();
        const auto conv = [&](core::TensorValue x, const std::string & name, int dilation) {
            const auto & w = decoder.at(name + ".weight");
            const auto & dims = w.shape.dims;
            if (dims[2] == 1) {
                auto time_major = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
                auto matrix = core::reshape_tensor(ctx, w, core::TensorShape::from_dims({dims[0], dims[1]}));
                auto projected = modules::LinearModule({dims[1], dims[0], true})
                    .build(ctx, time_major, {matrix, decoder.at(name + ".bias")});
                return modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, projected);
            }
            return modules::Conv1dModule({dims[1], dims[0], dims[2], 1, int(dims[2] / 2) * dilation, dilation, true})
                .build(ctx, x, {w, decoder.at(name + ".bias")});
        };
        const auto snake = [&](core::TensorValue x, const std::string & name) {
            return modules::Snake1dModule({x.shape.dims[1]}).build(ctx, x,
                {decoder.at(name + ".alpha"), decoder.at(name + ".inv_alpha")});
        };
        auto x = conv(g.latent, "decoder.input", 1);
        constexpr int dilations[] = {1, 3, 9};
        for (size_t stage = 0; stage < config.decoder_strides.size(); ++stage) {
            const auto prefix = "decoder.stages." + std::to_string(stage) + ".";
            const int stride = config.decoder_strides[stage];
            x = snake(x, prefix + "activation");
            const auto & w = decoder.at(prefix + "upsample.weight");
            const auto & dims = w.shape.dims;
            const int padding = (stride + 1) / 2;
            modules::ConvTranspose1dConfig upsample{dims[0], dims[1], dims[2], stride, padding, 1, true};
            const bool crop_padding = !modules::is_conv_transpose1d_col2im_fast_path_eligible(ctx, upsample);
            if (crop_padding) upsample.padding = 0;
            x = modules::ConvTranspose1dModule(upsample).build(ctx, x, {w, decoder.at(prefix + "upsample.bias")});
            if (crop_padding)
                x = modules::SliceModule({2, padding, x.shape.dims[2] - 2 * padding}).build(ctx, x);
            for (int unit = 0; unit < 3; ++unit) {
                const auto block = prefix + "residuals." + std::to_string(unit) + ".";
                auto residual = conv(snake(x, block + "activation1"), block + "conv1", dilations[unit]);
                residual = conv(snake(residual, block + "activation2"), block + "conv2", 1);
                x = modules::AddModule().build(ctx, x, residual);
            }
        }
        g.audio = modules::TanhModule().build(ctx, conv(snake(x, "decoder.output_activation"), "decoder.output", 1));
        g.graph.finalize({g.audio.tensor}, 8192);
    }
};

DialogueSidonRuntime::DialogueSidonRuntime(std::shared_ptr<const assets::TensorSource> source, core::ExecutionContext & execution)
    : state_(std::make_unique<State>(std::move(source), execution)) {}
DialogueSidonRuntime::~DialogueSidonRuntime() = default;

const DialogueSidonConfig & DialogueSidonRuntime::config() const noexcept { return state_->config; }
const std::vector<float> & DialogueSidonRuntime::latent_mean() const noexcept { return state_->latent_mean; }
const std::vector<float> & DialogueSidonRuntime::latent_std() const noexcept { return state_->latent_std; }

modules::Wav2Vec2BertEncoderOutput DialogueSidonRuntime::encode(const modules::Wav2Vec2BertEncoderInput & input) {
    const bool masked = std::find(input.attention_mask.begin(), input.attention_mask.end(), 0) != input.attention_mask.end();
    auto & encoder = masked ? *state_->masked_encoder : *state_->encoder;
    // Only one encoder graph stays resident.
    (masked ? *state_->encoder : *state_->masked_encoder).release_runtime_graph();
    encoder.prepare(input.frames);
    return encoder.encode(input);
}

DialogueSidonRuntime::Conditioning DialogueSidonRuntime::condition(const modules::Wav2Vec2BertEncoderOutput & features) {
    auto & s = *state_;
    if (features.dims != s.config.encoder_hidden) throw std::runtime_error("DialogueSidon encoder feature size mismatch");
    if (!s.condition_graph || s.condition_graph->frames != features.frames) s.build_condition_graph(features.frames);
    auto & g = *s.condition_graph;
    core::write_tensor_f32(g.features, features.values);
    g.graph.compute();
    Conditioning out;
    out.frames = features.frames;
    out.predicted = core::read_tensor_f32(g.predicted.tensor);
    out.values = core::read_tensor_f32(g.cond.tensor);
    return out;
}

void DialogueSidonRuntime::begin_sampling(const Conditioning & conditioning) {
    auto & s = *state_;
    if (conditioning.values.size() != size_t(conditioning.frames * s.config.cond_size()))
        throw std::runtime_error("DialogueSidon conditioning shape mismatch");
    if (!s.dit_graph || s.dit_graph->frames != conditioning.frames) s.build_dit_graph(conditioning.frames);
    s.bound_cond = conditioning.values;
}

std::vector<float> DialogueSidonRuntime::predict(const std::vector<float> & latents, int64_t timestep) {
    auto & s = *state_;
    if (!s.dit_graph) throw std::runtime_error("DialogueSidon predict() called before begin_sampling()");
    auto & g = *s.dit_graph;
    if (latents.size() != size_t(g.frames * s.config.joint_latent()))
        throw std::runtime_error("DialogueSidon latent shape mismatch");
    // TimestepEmbedder.timestep_embedding: cat(cos(t * f), sin(t * f)) in float32.
    const size_t half = s.time_freqs.size();
    std::vector<float> time(2 * half);
    for (size_t i = 0; i < half; ++i) {
        const float arg = float(timestep) * s.time_freqs[i];
        time[i] = std::cos(arg);
        time[half + i] = std::sin(arg);
    }
    core::write_tensor_f32(g.latents, latents);
    core::write_tensor_f32(g.cond, s.bound_cond);
    core::write_tensor_f32(g.time, time);
    core::write_tensor_i32(g.positions, s.positions);
    g.graph.compute();
    return core::read_tensor_f32(g.output.tensor);
}

std::vector<float> DialogueSidonRuntime::decode(const std::vector<float> & latent, int64_t frames) {
    auto & s = *state_;
    const int64_t channels = s.config.latent_dim;
    if (latent.size() != size_t(frames * channels)) throw std::runtime_error("DialogueSidon decoder latent shape mismatch");
    if (frames > 2048) {
        // The decoder sees about ten latent frames on either side, so tiles with a 16-frame
        // halo reproduce the full-length output while bounding the upsampled activations.
        constexpr int64_t tile_frames = 1024;
        constexpr int64_t halo = 16;
        const int64_t hop = s.config.hop_length;
        std::vector<float> output;
        output.reserve(size_t(s.config.decoded_samples(frames)));
        for (int64_t begin = 0; begin < frames; begin += tile_frames) {
            const int64_t end = std::min(begin + tile_frames, frames);
            const int64_t left = std::max<int64_t>(0, begin - halo);
            const int64_t right = std::min(frames, end + halo);
            std::vector<float> tile(size_t((right - left) * channels));
            for (int64_t c = 0; c < channels; ++c)
                std::copy(latent.begin() + c * frames + left, latent.begin() + c * frames + right,
                          tile.begin() + c * (right - left));
            auto decoded = decode(tile, right - left);
            const auto first = decoded.begin() + (begin - left) * hop;
            const auto last = end == frames ? decoded.end() : first + (end - begin) * hop;
            output.insert(output.end(), first, last);
        }
        return output;
    }
    if (!s.decoder_graph || s.decoder_graph->frames != frames) {
        s.decoder_graph.reset();
        s.build_decoder_graph(frames);
    }
    auto & g = *s.decoder_graph;
    core::write_tensor_f32(g.latent, latent);
    g.graph.compute();
    return core::read_tensor_f32(g.audio.tensor);
}

}  // namespace engine::community_models::dialogue_sidon
