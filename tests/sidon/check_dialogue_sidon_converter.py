#!/usr/bin/env python3
"""Self-check convert_dialogue_sidon.py on a small synthetic DialogueSidon checkpoint.

Builds the real upstream modules (HF w2v-BERT with PEFT LoRA, the DiT head from
a Sidon checkout, DAC's weight-normed decoder) at reduced width, saves them in
the Lightning and torch.export layouts, converts both, and checks that
converted tensors reproduce the PyTorch modules' outputs.
"""

import argparse
import ast
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent


def load_definitions(path, names, namespace):
    tree = ast.parse(Path(path).read_text())
    body = [node for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if len(body) != len(names):
        raise RuntimeError(f"{path}: missing {set(names) - {node.name for node in body}}")
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def build_modules(sidon_dir):
    # Locate descript-audio-codec without importing it; its package init pulls in audiotools.
    dac_dir = Path(importlib.util.find_spec("dac").submodule_search_locations[0])
    spec = importlib.util.spec_from_file_location("dac_layers", dac_dir / "nn/layers.py")
    layers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(layers)
    dac_ns = load_definitions(dac_dir / "model/dac.py",
                              ["ResidualUnit", "DecoderBlock", "Decoder"],
                              {"nn": nn, "math": math, "torch": torch, **vars(layers)})
    head_ns = load_definitions(
        Path(sidon_dir) / "src/sidon/model/dialogue_sidion/lightning_module.py",
        ["_modulate", "TimestepEmbedder", "DiTBlock", "DiTFinalLayer", "DiffusionTransformerHead"],
        {"nn": nn, "math": math, "torch": torch, "F": F, "transformers": transformers})
    return dac_ns["Decoder"], head_ns["DiffusionTransformerHead"]


def w2v_config(layers):
    return transformers.Wav2Vec2BertConfig(
        hidden_size=128, intermediate_size=256, num_attention_heads=4,
        num_hidden_layers=layers, layerdrop=0.0, hidden_dropout=0.0,
        attention_dropout=0.0, activation_dropout=0.0, feat_proj_dropout=0.0)


class SSLEncoderExport(nn.Module):
    def __init__(self, ssl, linear1, linear2):
        super().__init__()
        self.ssl_model, self.linear1, self.linear2 = ssl, linear1, linear2

    def forward(self, input_features, attention_mask):
        features = self.ssl_model(input_features=input_features, attention_mask=attention_mask).last_hidden_state
        return features, self.linear1(features), self.linear2(features)


class VaeDecoderExport(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, latents):
        return self.decoder(latents)


def snake(x, alpha, inv_alpha):
    return x + inv_alpha * torch.sin(alpha * x).pow(2)


def run_converted_decoder(tensors, config, latents):
    """Reference forward over converted names, mirroring SidonSnakeVocoderGraph."""
    def conv(x, name, dilation=1):
        weight = tensors[name + ".weight"]
        return F.conv1d(x, weight, tensors[name + ".bias"],
                        padding=(weight.shape[-1] // 2) * dilation, dilation=dilation)

    def act(x, name):
        return snake(x, tensors[name + ".alpha"], tensors[name + ".inv_alpha"])

    x = conv(latents, "decoder.input")
    for stage, stride in enumerate(config["strides"]):
        prefix = f"decoder.stages.{stage}."
        x = act(x, prefix + "activation")
        x = F.conv_transpose1d(x, tensors[prefix + "upsample.weight"], tensors[prefix + "upsample.bias"],
                               stride=stride, padding=math.ceil(stride / 2))
        for unit, dilation in enumerate(config["dilations"]):
            block = prefix + f"residuals.{unit}."
            y = conv(act(x, block + "activation1"), block + "conv1", dilation)
            y = conv(act(y, block + "activation2"), block + "conv2")
            pad = (x.shape[-1] - y.shape[-1]) // 2
            x = (x[..., pad:x.shape[-1] - pad] if pad else x) + y
    return torch.tanh(conv(act(x, "decoder.output_activation"), "decoder.output"))


def convert(args):
    subprocess.run([sys.executable, str(HERE / "convert_dialogue_sidon.py"), *args], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidon-dir", type=Path, required=True, help="sarulab-speech/Sidon checkout")
    args = parser.parse_args()
    torch.manual_seed(0)
    Decoder, DiffusionTransformerHead = build_modules(args.sidon_dir)

    student = get_peft_model(transformers.Wav2Vec2BertModel(w2v_config(13)), LoraConfig(
        r=64, lora_alpha=16, lora_dropout=0.1,
        target_modules=["output_dense", "intermediate_dense", "linear_q", "linear_k", "linear_v"]))
    for name, parameter in student.named_parameters():
        if "lora_B" in name:
            nn.init.normal_(parameter, std=0.02)
    latent_dim = 16
    linear1, linear2 = nn.Linear(128, latent_dim), nn.Linear(128, latent_dim)
    head = DiffusionTransformerHead(latent_size=2 * latent_dim, cond_size=128 + 2 * latent_dim,
                                    hidden_size=128, num_layers=2, num_heads=2, use_positional=True)
    for parameter in head.parameters():
        nn.init.normal_(parameter, std=0.05)
    decoder = Decoder(input_channel=latent_dim, channels=64, rates=[8, 5, 4, 3], d_out=1)
    for name, parameter in decoder.named_parameters():
        if name.endswith("alpha"):
            parameter.data.uniform_(0.5, 1.5)
    teacher = transformers.Wav2Vec2BertModel(w2v_config(2))
    bottleneck = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2 * latent_dim))
    mean, std = torch.randn(1, 1, 2 * latent_dim), torch.rand(1, 1, 2 * latent_dim) + 0.5
    student.eval(), head.eval(), decoder.eval()

    state = {}
    for prefix, module in (("student_ssl_model.", student), ("output_linear1.", linear1),
                           ("output_linear2.", linear2), ("diffusion_head.", head),
                           ("vae.decoder.", decoder), ("vae.ssl_model.", teacher),
                           ("vae.bottleneck.encoder.", bottleneck)):
        state.update({prefix + name: value for name, value in module.state_dict().items()})
    state["vae.discriminator.discriminator.discriminators.0.convs.0.weight_v"] = torch.randn(4, 1, 3)
    state["latent_norm_mean"], state["latent_norm_std"] = mean, std
    state["latent_norm_initialized"] = torch.tensor(True)
    cfg = {"lora": True, "sample_rate": 24000,
           "diffusion_head": {"hidden_size": 128, "head_layers": 2, "num_heads": 2, "use_positional": True},
           "diffusion": {"num_train_timesteps": 1000, "num_inference_steps": 30, "prediction_type": "v_prediction"},
           "latent_normalization": {"enabled": True}}

    features = torch.randn(1, 50, 160)
    mask = torch.ones(1, 50, dtype=torch.long)
    latents = torch.randn(1, 2 * latent_dim, 20)
    with torch.no_grad():
        expected_hidden = student(input_features=features, attention_mask=mask).last_hidden_state
        expected_audio = decoder(latents[:, :latent_dim])
        noisy = torch.randn(1, 20, 2 * latent_dim)
        conditioning = torch.randn(1, 20, 128 + 2 * latent_dim)
        steps = torch.tensor([321.0])
        expected_velocity = head(noisy, steps, conditioning)

    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        torch.save({"state_dict": state, "hyper_parameters": {"cfg": cfg}}, directory / "model.ckpt")
        export_dir = directory / "export"
        export_dir.mkdir()
        exported = True
        try:
            torch.export.save(torch.export.export(SSLEncoderExport(student, linear1, linear2),
                                                  (features, mask), strict=False), export_dir / "ssl_encoder.pt2")
            torch.export.save(torch.export.export(head, (noisy, steps.long(), conditioning), strict=False),
                              export_dir / "diffusion_head.pt2")
            torch.export.save(torch.export.export(VaeDecoderExport(decoder), (latents[:, :latent_dim],),
                                                  strict=False), export_dir / "vae_decoder.pt2")
            (export_dir / "metadata.json").write_text(json.dumps({
                "latent_dim": latent_dim, "sample_rate": 24000, "latent_norm_initialized": True,
                "latent_norm_mean": mean.squeeze().tolist(), "latent_norm_std": std.squeeze().tolist(),
                "ddpm_config": {"num_train_timesteps": 1000, "prediction_type": "v_prediction",
                                "beta_schedule": "linear", "beta_start": 1e-4, "beta_end": 0.02}}))
        except Exception as error:  # torch.export coverage varies across torch/peft versions
            exported = False
            print(f"skipping torch.export layout: {error!r}")

        sources = [("lightning", ["--checkpoint", str(directory / "model.ckpt")])]
        if exported:
            sources.append(("export", ["--export-dir", str(export_dir), "--num-heads", "2"]))
        for label, source_args in sources:
            out = directory / f"out-{label}"
            convert([*source_args, "--output-dir", str(out)])
            tensors = load_file(str(out / "model.safetensors"))
            config = json.loads((out / "config.json").read_text())
            report = json.loads((out / "conversion_report.json").read_text())

            covered = set(report["mapping"]) | {name for names in report["dropped"].values() for name in names}
            if label == "lightning":
                missing = set(state) - covered
                assert not missing, f"unaccounted source tensors: {sorted(missing)[:5]}"

            plain = transformers.Wav2Vec2BertModel(w2v_config(13)).eval()
            encoder_state = {name: value for name, value in tensors.items()
                             if name.startswith(("feature_projection.", "encoder."))}
            result = plain.load_state_dict(encoder_state, strict=False)
            assert not result.unexpected_keys and set(result.missing_keys) <= {"masked_spec_embed"}, result
            rebuilt = DiffusionTransformerHead(latent_size=2 * latent_dim, cond_size=128 + 2 * latent_dim,
                                               hidden_size=128, num_layers=config["diffusion_head"]["num_layers"],
                                               num_heads=config["diffusion_head"]["num_heads"],
                                               use_positional=config["diffusion_head"]["use_rope"]).eval()
            renames = {"t_embedder.fc1.": "t_embedder.mlp.0.", "t_embedder.fc2.": "t_embedder.mlp.2.",
                       ".ffn.fc1.": ".mlp.0.", ".ffn.fc2.": ".mlp.2.", ".adaln.": ".adaLN_modulation.1."}
            head_state = {}
            for name, value in tensors.items():
                if name.startswith("diffusion_head.") and not name.endswith(".freqs"):
                    local = name[len("diffusion_head."):]
                    for new, old in renames.items():
                        local = local.replace(new, old)
                    head_state[local] = value
            rebuilt.load_state_dict(head_state, strict=True)
            with torch.no_grad():
                hidden = plain(input_features=features, attention_mask=mask).last_hidden_state
                velocity = rebuilt(noisy, steps, conditioning)
                audio = run_converted_decoder(tensors, config["decoder"], latents[:, :latent_dim])
            checks = {
                "encoder (LoRA folded)": (hidden, expected_hidden),
                "speaker head 1": (hidden @ tensors["output_linear1.weight"].T + tensors["output_linear1.bias"],
                                   linear1(expected_hidden)),
                "diffusion head": (velocity, expected_velocity),
                "decoder (weight norm folded)": (audio, expected_audio),
                "latent norm mean": (tensors["latent_norm.mean"], mean.reshape(-1)),
            }
            for name, (actual, reference) in checks.items():
                error = (actual - reference).abs().max().item()
                assert actual.shape == reference.shape and error < 1e-4, f"{label} {name}: max error {error}"
                print(f"{label:9s} {name:30s} max abs error {error:.2e}")
            assert config["decoder"]["strides"] == [8, 5, 4, 3] and config["decoder"]["hop_length"] == 480
            assert config["encoder"]["num_hidden_layers"] == 13
            assert config["diffusion_head"]["num_heads"] == 2
            print(f"{label}: {report['source_tensors']} source tensors, {report['output_tensors']} written")
    print("OK")


if __name__ == "__main__":
    main()
