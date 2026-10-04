#!/usr/bin/env python3
"""Run the DialogueSidon demo path (HF Space app.py) on the released torch.export
components and dump every stage as raw float32 for compare_parity.py.

The stage names match what audiocpp_cli writes with DIALOGUE_SIDON_DUMP_DIR set:
wav16k_padded, input_features, attention_mask, features, predicted, conditioning,
noise, step_XX_model_output, step_XX_latents, latents_final, wav_speaker0/1.

Pass --noise <dump>/noise.f32 to start from audio.cpp's initial latent; otherwise
the noise is torch.randn on the device after torch.manual_seed(--seed), which
audio.cpp reproduces on CUDA hosts with the same seed.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch
import torchaudio
from diffusers import DPMSolverMultistepScheduler


def fbank_features(wav):
    feat = torchaudio.compliance.kaldi.fbank(
        wav.unsqueeze(0), sample_frequency=16000, num_mel_bins=80, frame_length=25, frame_shift=10,
        dither=0.0, preemphasis_coefficient=0.97, remove_dc_offset=True, window_type="povey",
        use_energy=False, energy_floor=1.192092955078125e-07)
    feat = (feat - feat.mean(0, keepdim=True)) / torch.sqrt(feat.var(0, keepdim=True) + 1e-5)
    frames = feat.shape[0] + feat.shape[0] % 2
    padded = torch.zeros(1, frames, 80, device=wav.device)
    mask = torch.zeros(1, frames, dtype=torch.int64, device=wav.device)
    padded[0, :feat.shape[0]] = feat
    mask[0, :feat.shape[0]] = 1
    return padded.reshape(1, frames // 2, 160), mask[:, 1::2]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True, help="sarulab-speech/DialogueSidon snapshot")
    parser.add_argument("--audio", type=Path, required=True, help="mixture, at most 120 s")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise", type=Path, help="raw float32 initial latent [frames * 64]")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)

    def dump(name, value):
        value.detach().float().cpu().contiguous().numpy().tofile(args.output_dir / f"{name}.f32")

    meta = json.loads((args.model_dir / "metadata.json").read_text())
    # The released archives keep CUDA storages; load them on a CUDA host.
    encoder = torch.export.load(str(args.model_dir / "ssl_encoder.pt2")).module().to(device)
    head = torch.export.load(str(args.model_dir / "diffusion_head.pt2")).module().to(device)
    decoder = torch.export.load(str(args.model_dir / "vae_decoder.pt2")).module().to(device)
    mean = torch.tensor(meta["latent_norm_mean"], device=device).view(1, 1, -1)
    std = torch.tensor(meta["latent_norm_std"], device=device).view(1, 1, -1)
    scheduler = DPMSolverMultistepScheduler.from_config(
        meta["ddpm_config"], algorithm_type="dpmsolver++", timestep_spacing="linspace")
    latent_dim = meta["latent_dim"]

    audio, sample_rate = sf.read(args.audio, dtype="float32", always_2d=True)
    wav = torch.from_numpy(audio).mean(dim=1).view(1, -1)
    if sample_rate != 16000:
        wav = torchaudio.functional.resample(wav, sample_rate, 16000)
    if wav.shape[-1] > 120 * 16000:
        raise ValueError("The reference covers the single-pass path (<= 120 s)")
    wav = wav.to(device)
    started = time.perf_counter()
    with torch.inference_mode():
        wav = torch.nn.functional.pad(0.9 * wav / wav.abs().max().clamp_min(1e-6), (160, 160))
        dump("wav16k_padded", wav)
        input_features, attention_mask = fbank_features(wav.view(-1))
        dump("input_features", input_features)
        dump("attention_mask", attention_mask)
        features, pred0, pred1 = encoder(input_features, attention_mask)
        predicted = torch.cat([pred0, pred1], dim=-1)
        conditioning = torch.cat([(predicted - mean) / std, features], dim=-1)
        for name, value in (("features", features), ("predicted", predicted), ("conditioning", conditioning)):
            dump(name, value)
        frames = conditioning.shape[1]
        if args.noise:
            latents = torch.from_numpy(np.fromfile(args.noise, dtype=np.float32)).view(1, frames, 2 * latent_dim).to(device)
        else:
            torch.manual_seed(args.seed)
            latents = torch.randn((1, frames, 2 * latent_dim), device=device)
        dump("noise", latents)
        scheduler.set_timesteps(args.steps, device=device)
        for index, t in enumerate(scheduler.timesteps):
            timestep = torch.full((1,), int(t.item()), device=device, dtype=torch.long)
            output = head(latents, timestep, conditioning)
            latents = scheduler.step(output, t, latents).prev_sample
            dump(f"step_{index:02d}_model_output", output)
            dump(f"step_{index:02d}_latents", latents)
        latents = latents * std + mean
        dump("latents_final", latents)
        for speaker in range(2):
            part = latents[:, :, speaker * latent_dim:(speaker + 1) * latent_dim].transpose(1, 2)
            dump(f"wav_speaker{speaker}", decoder(part).view(-1))
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(json.dumps({"frames": frames, "wall_ms": (time.perf_counter() - started) * 1000,
                      "timesteps": [int(t) for t in scheduler.timesteps]}))


if __name__ == "__main__":
    main()
