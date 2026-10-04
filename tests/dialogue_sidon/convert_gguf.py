#!/usr/bin/env python3
"""Convert DialogueSidon weights into audio.cpp tensor names.

Two sources are accepted:

* ``--checkpoint``: the Lightning ``.ckpt`` written by
  ``DialogueSidonDiffusionLightningModule`` (sarulab-speech/Sidon).
* ``--export-dir``: the released ``torch.export`` bundle from
  ``export_diffusion_dialogue.py`` (``ssl_encoder.pt2``, ``diffusion_head.pt2``,
  ``vae_decoder.pt2`` and ``metadata.json``), as published in
  sarulab-speech/DialogueSidon.

The converter folds the encoder's LoRA adapters and the decoder's weight norm,
drops training-only modules, and fails on any tensor it does not recognize.
It writes ``model.safetensors``, ``config.json`` and ``conversion_report.json``
to ``--output-dir``; ``--gguf`` additionally packages a GGUF with
``audiocpp_gguf`` once a ``dialogue_sidon`` model spec exists.
"""

import argparse
from collections import defaultdict
import io
import json
import math
from pathlib import Path
import re
import subprocess
import zipfile

import torch
from safetensors.torch import save_file

STUDENT = "student_ssl_model."
HEAD = "diffusion_head."
DECODER = "vae.decoder.model."

# Modules the Lightning checkpoint carries that inference never runs.
DROPPED_PREFIXES = {
    "vae.ssl_model.": "teacher w2v-BERT that builds the training targets",
    "vae.bottleneck.": "SSL-VAE encoder head; only produces training targets",
    "vae.discriminator.": "GAN discriminator (training only)",
    "vae.regression_loss.": "mel-loss buffers (training only)",
    "pit.": "permutation-invariant training metric",
    "ssl_model_criterion.": "training loss",
}
DROPPED_ENCODER = {
    "masked_spec_embed": "SpecAugment mask embedding (training only)",
}

ENCODER_PATTERNS = [re.compile(pattern) for pattern in (
    r"feature_projection\.(layer_norm|projection)\.(weight|bias)",
    r"encoder\.layers\.\d+\.(ffn[12]_layer_norm|self_attn_layer_norm|final_layer_norm)\.(weight|bias)",
    r"encoder\.layers\.\d+\.conv_module\.(layer_norm|depthwise_layer_norm)\.(weight|bias)",
    r"encoder\.layers\.\d+\.ffn[12]\.(intermediate_dense|output_dense)\.(weight|bias)",
    r"encoder\.layers\.\d+\.self_attn\.linear_(q|k|v|out)\.(weight|bias)",
    r"encoder\.layers\.\d+\.self_attn\.distance_embedding\.weight",
    r"encoder\.layers\.\d+\.conv_module\.(pointwise_conv1|depthwise_conv|pointwise_conv2)\.weight",
)]
ENCODER_TENSORS_PER_LAYER = 32

HEAD_RENAMES = [(re.compile(pattern), replacement) for pattern, replacement in (
    (r"t_embedder\.mlp\.0\.weight", "t_embedder.fc1.weight"),
    (r"t_embedder\.mlp\.2\.weight", "t_embedder.fc2.weight"),
    (r"(latent_proj|cond_proj)\.weight", r"\1.weight"),
    (r"blocks\.(\d+)\.(q_proj|k_proj|v_proj|out_proj)\.(weight|bias)", r"blocks.\1.\2.\3"),
    (r"blocks\.(\d+)\.mlp\.0\.(weight|bias)", r"blocks.\1.ffn.fc1.\2"),
    (r"blocks\.(\d+)\.mlp\.2\.(weight|bias)", r"blocks.\1.ffn.fc2.\2"),
    (r"blocks\.(\d+)\.adaLN_modulation\.1\.(weight|bias)", r"blocks.\1.adaln.\2"),
    (r"final_layer\.adaLN_modulation\.1\.(weight|bias)", r"final_layer.adaln.\1"),
    (r"final_layer\.linear\.weight", "final_layer.linear.weight"),
)]

# Pieces of the PyTorch inference path that audio.cpp has no ready module for.
UNSUPPORTED = [
    {
        "component": "DPMSolverMultistepScheduler (dpmsolver++, linspace, linear betas 1e-4..0.02, v_prediction)",
        "status": "missing",
        "note": "VibeVoiceDPMSolverScheduler implements the same solver but rejects non-cosine beta schedules; "
                "generalize it or add a model-local scheduler.",
    },
    {
        "component": "DiT timestep embedder (cos|sin, 256 -> hidden, bias-free MLP)",
        "status": "partial",
        "note": "TimestepEmbeddingModule requires fc biases and an RMS weight; build it model-locally. "
                "The converter emits diffusion_head.t_embedder.freqs for it.",
    },
    {
        "component": "SSL-VAE snake decoder (latent_dim -> 1536 channels, strides 8/5/4/3, 24 kHz)",
        "status": "partial",
        "note": "Same blocks and tensor names as the Sidon v0.1 decoder, but SidonSnakeVocoderGraph hard-codes "
                "1024 input channels and five strides; parameterize it from config.json.",
    },
]


def load_lightning(path):
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    state = dict(checkpoint["state_dict"])
    hparams = checkpoint.get("hyper_parameters", {})
    cfg = hparams.get("cfg", hparams) or {}

    def get(node, key, default):
        try:
            value = node.get(key, default)
        except AttributeError:
            return default
        return default if value is None else value

    head_cfg = get(cfg, "diffusion_head", {})
    diffusion_cfg = get(cfg, "diffusion", {})
    norm_cfg = get(cfg, "latent_normalization", {})
    if not get(cfg, "lora", True) and any(".lora_" in key for key in state):
        raise ValueError("Checkpoint has LoRA tensors although cfg.lora is false")
    meta = {
        "source": str(path),
        "sample_rate": int(get(cfg, "sample_rate", 24000)),
        "num_heads": get(head_cfg, "num_heads", None),
        "use_rope": bool(get(head_cfg, "use_positional", False)),
        "latent_norm_enabled": bool(get(norm_cfg, "enabled", True)),
        "scheduler": {
            "num_train_timesteps": int(get(diffusion_cfg, "num_train_timesteps", 1000)),
            "prediction_type": str(get(diffusion_cfg, "prediction_type", "epsilon")),
            "beta_schedule": "linear",
            "beta_start": 1.0e-4,
            "beta_end": 0.02,
            "num_inference_steps": int(get(diffusion_cfg, "num_inference_steps", 1000)),
            "algorithm_type": str(get(diffusion_cfg, "inference_algorithm_type", "dpmsolver++")),
            "timestep_spacing": str(get(diffusion_cfg, "inference_timestep_spacing", "linspace")),
        },
    }
    return state, meta


def read_export_weights(path):
    """Load a .pt2 archive's weights on CPU without rebuilding its graph.

    The released bundle (torch 2.8) stores a pickled ``data/weights/model.pt``
    with CUDA storages, which ``torch.export.load`` cannot place on a CPU-only host.
    """
    with zipfile.ZipFile(path) as archive:
        legacy = [name for name in archive.namelist() if name.endswith("data/weights/model.pt")]
        if legacy:
            return torch.load(io.BytesIO(archive.read(legacy[0])), map_location="cpu", weights_only=True)
    return {name: value.cpu() for name, value in torch.export.load(str(path)).state_dict.items()}


def load_export(directory):
    state = {}
    for filename, prefixes in (
        ("ssl_encoder.pt2", {"ssl_model.": STUDENT, "linear1.": "output_linear1.", "linear2.": "output_linear2."}),
        ("diffusion_head.pt2", {"": HEAD}),
        ("vae_decoder.pt2", {"decoder.": "vae.decoder."}),
    ):
        for name, value in read_export_weights(directory / filename).items():
            for source, target in prefixes.items():
                if name.startswith(source):
                    state[target + name[len(source):]] = value
                    break
            else:
                raise ValueError(f"{filename}: unexpected tensor {name}")
    with open(directory / "metadata.json") as handle:
        metadata = json.load(handle)
    state["latent_norm_mean"] = torch.tensor(metadata["latent_norm_mean"], dtype=torch.float32)
    state["latent_norm_std"] = torch.tensor(metadata["latent_norm_std"], dtype=torch.float32)
    state["latent_norm_initialized"] = torch.tensor(bool(metadata["latent_norm_initialized"]))
    ddpm = metadata.get("ddpm_config", {})
    meta = {
        "source": str(directory),
        "sample_rate": int(metadata.get("sample_rate", 24000)),
        # The released head is 768 hidden / 12 heads; the count only appears in the
        # exported graph (view(1, T, 12, 64)), not in the weights.
        "num_heads": 12,
        # Every released DialogueSidon config trains the head with RoPE.
        "use_rope": True,
        "latent_norm_enabled": True,
        "scheduler": {
            "num_train_timesteps": int(ddpm.get("num_train_timesteps", 1000)),
            "prediction_type": str(ddpm.get("prediction_type", "v_prediction")),
            "beta_schedule": str(ddpm.get("beta_schedule", "linear")),
            "beta_start": float(ddpm.get("beta_start", 1.0e-4)),
            "beta_end": float(ddpm.get("beta_end", 0.02)),
            "num_inference_steps": 30,
            "algorithm_type": "dpmsolver++",
            "timestep_spacing": "linspace",
        },
    }
    return state, meta


class Converter:
    def __init__(self, state, lora_alpha):
        self.state = {name: value for name, value in state.items() if torch.is_tensor(value)}
        self.remaining = set(self.state)
        self.lora_alpha = lora_alpha
        self.tensors = {}
        self.mapping = defaultdict(list)
        self.dropped = defaultdict(list)
        self.derived = {}

    def take(self, name):
        self.remaining.discard(name)
        return self.state[name].detach().float()

    def emit(self, target, value, *sources):
        if target in self.tensors:
            raise ValueError(f"Duplicate output tensor {target}")
        self.tensors[target] = value.contiguous()
        for source in sources:
            self.mapping[source].append(target)

    def drop_training_only(self):
        for name in sorted(self.remaining):
            for prefix, reason in DROPPED_PREFIXES.items():
                if name.startswith(prefix):
                    self.dropped[reason].append(name)
                    self.remaining.discard(name)
                    break

    def encoder(self):
        names = sorted(name for name in self.remaining if name.startswith(STUDENT))
        if not names:
            raise ValueError("No student w2v-BERT tensors found")
        modules = defaultdict(dict)
        for name in names:
            local = name[len(STUDENT):]
            if local.startswith("base_model.model."):
                local = local[len("base_model.model."):]
            if local in DROPPED_ENCODER:
                self.dropped[DROPPED_ENCODER[local]].append(name)
                self.remaining.discard(name)
                continue
            lora = re.fullmatch(r"(.+)\.lora_([AB])\.([^.]+)\.weight", local)
            if lora:
                modules[lora.group(1)][f"lora_{lora.group(2)}.{lora.group(3)}"] = name
                continue
            if re.search(r"\.lora_(embedding|magnitude)", local):
                raise ValueError(f"Unsupported adapter tensor {name}")
            local = local.replace(".base_layer.", ".")
            module, _, leaf = local.rpartition(".")
            modules[module][leaf] = name

        lora_rank = None
        merged = 0
        for module, parts in sorted(modules.items()):
            adapters = sorted({key.split(".", 1)[1] for key in parts if key.startswith("lora_")})
            for leaf, name in parts.items():
                if leaf.startswith("lora_"):
                    continue
                target = f"{module}.{leaf}"
                if not any(pattern.fullmatch(target) for pattern in ENCODER_PATTERNS):
                    raise ValueError(f"Unrecognized encoder tensor {name} -> {target}")
                value = self.take(name)
                sources = [name]
                if leaf == "weight":
                    for adapter in adapters:
                        a_name = parts[f"lora_A.{adapter}"]
                        b_name = parts[f"lora_B.{adapter}"]
                        lora_a, lora_b = self.take(a_name), self.take(b_name)
                        rank = lora_a.shape[0]
                        lora_rank = lora_rank or rank
                        value = value + (self.lora_alpha / rank) * (lora_b @ lora_a)
                        sources += [a_name, b_name]
                        merged += 1
                self.emit(target, value, *sources)

        layers = defaultdict(int)
        for name in self.tensors:
            match = re.match(r"encoder\.layers\.(\d+)\.", name)
            if match:
                layers[int(match.group(1))] += 1
        if sorted(layers) != list(range(len(layers))):
            raise ValueError(f"Encoder layers are not contiguous: {sorted(layers)}")
        incomplete = {layer: count for layer, count in layers.items() if count != ENCODER_TENSORS_PER_LAYER}
        if incomplete:
            raise ValueError(f"Encoder layers with missing tensors: {incomplete}")
        hidden = self.tensors["feature_projection.projection.weight"].shape[0]
        return {
            "num_hidden_layers": len(layers),
            "hidden_size": int(hidden),
            "feature_dim": int(self.tensors["feature_projection.projection.weight"].shape[1]),
            "intermediate_size": int(self.tensors["encoder.layers.0.ffn1.intermediate_dense.weight"].shape[0]),
            "relative_positions": int(self.tensors["encoder.layers.0.self_attn.distance_embedding.weight"].shape[0]),
            "conv_kernel": int(self.tensors["encoder.layers.0.conv_module.depthwise_conv.weight"].shape[-1]),
            "lora_rank": lora_rank,
            "lora_alpha": self.lora_alpha if merged else None,
            "lora_merged_modules": merged,
        }

    def speaker_heads(self):
        for index in (1, 2):
            for leaf in ("weight", "bias"):
                name = f"output_linear{index}.{leaf}"
                if name not in self.state:
                    raise ValueError(f"Missing {name}")
                self.emit(name, self.take(name), name)
        return int(self.tensors["output_linear1.weight"].shape[0])

    def diffusion_head(self, num_heads, use_rope):
        names = sorted(name for name in self.remaining if name.startswith(HEAD))
        if not names:
            raise ValueError("No diffusion_head tensors found; the no-diffusion-head variant is not supported")
        for name in names:
            local = name[len(HEAD):]
            for pattern, replacement in HEAD_RENAMES:
                if pattern.fullmatch(local):
                    self.emit(HEAD + pattern.sub(replacement, local), self.take(name), name)
                    break
            else:
                raise ValueError(f"Unrecognized diffusion head tensor {name}")
        latent = self.tensors[HEAD + "latent_proj.weight"]
        hidden, latent_size = latent.shape
        frequency_size = self.tensors[HEAD + "t_embedder.fc1.weight"].shape[1]
        half = frequency_size // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
        self.emit(HEAD + "t_embedder.freqs", freqs)
        self.derived[HEAD + "t_embedder.freqs"] = "exp(-ln(10000) * i / 128), i < 128"
        num_layers = len({re.match(r"blocks\.(\d+)\.", name[len(HEAD):]).group(1)
                          for name in self.tensors if name.startswith(HEAD + "blocks.")})
        num_heads = int(num_heads or hidden // 64)
        if hidden % num_heads or (hidden // num_heads) % 2:
            raise ValueError(f"Invalid head count {num_heads} for hidden size {hidden}")
        return {
            "hidden_size": int(hidden),
            "latent_size": int(latent_size),
            "cond_size": int(self.tensors[HEAD + "cond_proj.weight"].shape[1]),
            "num_layers": num_layers,
            "num_heads": num_heads,
            "ffn_size": int(self.tensors[HEAD + "blocks.0.ffn.fc1.weight"].shape[0]),
            "frequency_embedding_size": int(frequency_size),
            "use_rope": use_rope,
            "rope_theta": 10000.0,
            "layer_norm_eps": 1.0e-6,
        }

    def decoder(self):
        names = sorted(name for name in self.remaining if name.startswith(DECODER))
        if not names:
            raise ValueError("No SSL-VAE decoder tensors found")
        grouped = defaultdict(dict)
        for name in names:
            local = name[len(DECODER):]
            local = local.replace(".parametrizations.weight.original0", ".weight_g")
            local = local.replace(".parametrizations.weight.original1", ".weight_v")
            module, _, leaf = local.rpartition(".")
            grouped[module][leaf] = name
        top = sorted({int(module.split(".")[0]) for module in grouped})
        last = max(top)
        stages = [index for index in top if f"{index}.block.1" in grouped]
        if stages != list(range(1, len(stages) + 1)) or last != len(stages) + 2:
            raise ValueError(f"Unexpected decoder layout: stages {stages}, last module {last}")

        def target_for(module):
            parts = module.split(".")
            index = int(parts[0])
            if module == "0":
                return "input"
            if module == str(last):
                return "output"
            if module == str(last - 1):
                return "output_activation"
            stage = f"stages.{index - 1}."
            if parts[1:] == ["block", "0"]:
                return stage + "activation"
            if parts[1:] == ["block", "1"]:
                return stage + "upsample"
            if len(parts) == 5 and parts[1] == "block" and parts[2] in "234" and parts[3] == "block":
                return stage + f"residuals.{int(parts[2]) - 2}." + (
                    "activation1", "conv1", "activation2", "conv2")[int(parts[4])]
            raise ValueError(f"Unrecognized decoder module {module}")

        strides = []
        for module, leaves in sorted(grouped.items(), key=lambda item: [int(p) if p.isdigit() else p
                                                                         for p in item[0].split(".")]):
            target = "decoder." + target_for(module)
            if set(leaves) == {"alpha"}:
                alpha = self.take(leaves["alpha"])
                self.emit(target + ".alpha", alpha, leaves["alpha"])
                self.emit(target + ".inv_alpha", 1.0 / (alpha + 1e-9))
                self.derived[target + ".inv_alpha"] = "1 / (alpha + 1e-9), as in dac.nn.layers.snake"
            elif set(leaves) == {"weight_g", "weight_v", "bias"}:
                g, v = self.take(leaves["weight_g"]), self.take(leaves["weight_v"])
                norm = v.flatten(1).norm(dim=1).view(-1, *([1] * (v.dim() - 1)))
                self.emit(target + ".weight", g * v / norm, leaves["weight_g"], leaves["weight_v"])
                self.emit(target + ".bias", self.take(leaves["bias"]), leaves["bias"])
                if target.endswith(".upsample"):
                    strides.append(int(v.shape[-1]) // 2)
            elif set(leaves) == {"weight", "bias"}:
                weight = self.take(leaves["weight"])
                self.emit(target + ".weight", weight, leaves["weight"])
                self.emit(target + ".bias", self.take(leaves["bias"]), leaves["bias"])
                if target.endswith(".upsample"):
                    strides.append(int(weight.shape[-1]) // 2)
            else:
                raise ValueError(f"Unexpected decoder parameters {module}: {sorted(leaves)}")
        input_weight = self.tensors["decoder.input.weight"]
        return {
            "latent_dim": int(input_weight.shape[1]),
            "channels": int(input_weight.shape[0]),
            "strides": strides,
            "dilations": [1, 3, 9],
            "hop_length": int(math.prod(strides)),
        }

    def latent_norm(self, enabled, latent_size):
        mean = self.take("latent_norm_mean").reshape(-1)
        std = self.take("latent_norm_std").reshape(-1)
        initialized = bool(self.state["latent_norm_initialized"].item())
        self.remaining.discard("latent_norm_initialized")
        if mean.numel() != latent_size or std.numel() != latent_size:
            raise ValueError(f"Latent norm stats have {mean.numel()} values, expected {latent_size}")
        active = enabled and initialized
        if not active:
            mean, std = torch.zeros_like(mean), torch.ones_like(std)
        self.emit("latent_norm.mean", mean, "latent_norm_mean")
        self.emit("latent_norm.std", std, "latent_norm_std")
        self.mapping["latent_norm_initialized"].append("config.json:latent_norm.active")
        return {"active": active, "eps": 1.0e-6}


def convert(state, meta, lora_alpha):
    converter = Converter(state, lora_alpha)
    converter.drop_training_only()
    encoder = converter.encoder()
    latent_dim = converter.speaker_heads()
    head = converter.diffusion_head(meta["num_heads"], meta["use_rope"])
    decoder = converter.decoder()
    latent_norm = converter.latent_norm(meta["latent_norm_enabled"], head["latent_size"])
    if converter.remaining:
        raise ValueError(f"Unmapped tensors: {sorted(converter.remaining)}")
    if decoder["latent_dim"] != latent_dim or head["latent_size"] != 2 * latent_dim:
        raise ValueError("Speaker heads, diffusion head and decoder disagree on latent size")
    if head["cond_size"] != encoder["hidden_size"] + head["latent_size"]:
        raise ValueError("Diffusion head conditioning does not match encoder + latent size")
    config = {
        "family": "dialogue_sidon",
        "source": meta["source"],
        "sample_rate": meta["sample_rate"],
        "input_sample_rate": 16000,
        "num_speakers": 2,
        "latent_dim": latent_dim,
        "encoder": encoder,
        "diffusion_head": head,
        "decoder": decoder,
        "latent_norm": latent_norm,
        "scheduler": meta["scheduler"],
    }
    report = {
        "source_tensors": len(converter.state),
        "output_tensors": len(converter.tensors),
        "mapping": dict(sorted(converter.mapping.items())),
        "derived": converter.derived,
        "dropped": {reason: names for reason, names in converter.dropped.items()},
        "unsupported": UNSUPPORTED,
    }
    return converter.tensors, config, report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="DialogueSidonDiffusionLightningModule .ckpt")
    source.add_argument("--export-dir", type=Path, help="Directory with the released .pt2 files and metadata.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lora-alpha", type=float, default=16.0,
                        help="LoRA alpha used in training (scale = alpha / rank); Sidon uses 16")
    parser.add_argument("--num-heads", type=int, help="Override the diffusion head's attention head count")
    parser.add_argument("--gguf", type=Path, help="Also package a GGUF with audiocpp_gguf")
    parser.add_argument("--model-spec", type=Path, help="dialogue_sidon model spec for --gguf")
    parser.add_argument("--type", default="f32", help="GGUF tensor storage type")
    parser.add_argument("--audiocpp-gguf", type=Path,
                        default=Path(__file__).resolve().parents[2] / "build/debug/bin/audiocpp_gguf")
    args = parser.parse_args()

    state, meta = load_lightning(args.checkpoint) if args.checkpoint else load_export(args.export_dir)
    if args.num_heads:
        meta["num_heads"] = args.num_heads
    tensors, config, report = convert(state, meta, args.lora_alpha)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights = args.output_dir / "model.safetensors"
    save_file(tensors, str(weights), metadata={"source": "sarulab-speech/Sidon DialogueSidon"})
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.output_dir / "conversion_report.json").write_text(json.dumps(report, indent=2) + "\n")
    dropped = sum(len(names) for names in report["dropped"].values())
    print(f"Converted {report['source_tensors']} source tensors: {report['output_tensors']} written, "
          f"{dropped} training-only dropped, {len(report['derived'])} derived")
    for item in UNSUPPORTED:
        print(f"  needs runtime work: {item['component']}")

    if args.gguf:
        command = [str(args.audiocpp_gguf), "--input", f"weights={weights}", "--output", str(args.gguf),
                   "--type", args.type, "--no-sidecars", "--overwrite"]
        if args.model_spec:
            command += ["--family", "dialogue_sidon", "--model-spec", str(args.model_spec)]
        else:
            print("No --model-spec given; writing a tensor archive audio.cpp will not load as a model")
            command += ["--allow-missing-model-spec"]
        subprocess.run(command, check=True)
        print(f"Wrote {args.gguf}")


if __name__ == "__main__":
    main()
