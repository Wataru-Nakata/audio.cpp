# DialogueSidon

DialogueSidon separates and restores a two-speaker conversation. The input is a
mixture. The output is one clean 24 kHz track per speaker.

It is a sibling of [Sidon](../models/sidon.md), not a variant. A w2v-BERT 2.0
encoder conditions an 8-layer diffusion transformer. The transformer samples
SSL-VAE latents for both speakers with DPM-Solver++. A snake decoder turns each
speaker's latents into audio.

## Usage

```bash
audiocpp_cli --task sep --family dialogue_sidon \
  --model /path/to/DialogueSidon-GGUF/dialogue-sidon-f32.gguf \
  --audio dialogue.wav --backend cuda --threads 8 --log \
  --out separated.wav --out-dir separated/
```

`--out` writes a stereo file with speaker 1 on the left and speaker 2 on the
right. `--out-dir` writes `speaker1.wav` and `speaker2.wav`.

Request options (`--request-option key=value`):

| Option | Default | Meaning |
|---|---|---|
| `num_steps` | 30 | DPM-Solver++ steps |
| `seed` | 0 | Seed for the initial noise. Chunk *i* of a long input uses `seed + i`. |
| `normalize` | true | Scale each speaker to a 0.9 peak, as the demo does |

The pipeline follows the released demo (`spaces/app.py` in
[sarulab-speech/DialogueSidon](https://huggingface.co/sarulab-speech/DialogueSidon)):

- Stereo input is mixed to mono and resampled to 16 kHz with torchaudio's
  sinc-Hann kernel. The audio is peak-normalized to 0.9 and padded by 160
  samples on each side.
- The frontend computes 80-bin Kaldi fbank features with per-utterance CMVN and
  stacks frame pairs. An odd frame count adds a masked zero frame. The encoder
  then runs with the export's key and convolution masking.
- Two speaker heads predict each speaker's latents. Their normalized values,
  concatenated with the encoder features, condition the diffusion transformer.
- DPM-Solver++ uses order 2, linear betas, linspace timesteps and
  v-prediction. The timesteps are rounded as numpy does, so step 15 is 499.
- The initial noise is generated with `sampling::generate_torch_cuda_randn`. It
  is the same stream torch draws on CUDA for the same seed.
- The decoder produces 480 × T − 13 samples for T latent frames. Long latents
  are decoded in tiles of 1024 frames with a 16-frame halo.

Input longer than 120 s is processed in 120 s chunks that overlap by 10 s, as
the demo does. The overlap is crossfaded linearly. This is offline processing,
not streaming.

Speaker order in each chunk is matched to the previous chunk on the overlap.
This differs from the demo on purpose. The demo correlates raw waveforms, but
each chunk is resynthesized from its own noise. The overlapping waveforms
therefore don't line up in phase, and in testing the demo's check swapped the
speakers in the second chunk of a 130 s mix. audio.cpp compares 20 ms loudness
envelopes instead, which follow who is talking when. On the same mix that gave
a score of 1.91 for the correct order against −0.16 for the swapped one.

## Conversion

Download the release from
[sarulab-speech/DialogueSidon](https://huggingface.co/sarulab-speech/DialogueSidon).
You need `ssl_encoder.pt2`, `diffusion_head.pt2`, `vae_decoder.pt2` and
`metadata.json`. Build `audiocpp_gguf`, then run:

```bash
python tests/dialogue_sidon/convert_gguf.py \
  --export-dir /path/to/DialogueSidon \
  --output-dir /path/to/ds_converted \
  --gguf /path/to/DialogueSidon-GGUF/dialogue-sidon-f32.gguf \
  --model-spec model_specs/dialogue_sidon.json \
  --audiocpp-gguf build/rel/bin/audiocpp_gguf
```

The converter merges the encoder's LoRA adapters and folds the decoder's weight
norm. It drops training-only modules and fails on any tensor it does not
recognize. It also accepts a Lightning `--checkpoint` from
sarulab-speech/Sidon.

## Parity

`DIALOGUE_SIDON_DUMP_DIR=<dir>` makes the CLI dump the first chunk's stages as
raw float32. `DIALOGUE_SIDON_NOISE=<file>` replaces the initial noise with a
file. To compare against the released `torch.export` components:

```bash
DIALOGUE_SIDON_DUMP_DIR=cpp audiocpp_cli --task sep --family dialogue_sidon ...
python tests/dialogue_sidon/python_reference.py --model-dir /path/to/DialogueSidon \
  --audio dialogue.wav --noise cpp/noise.f32 --output-dir ref
python tests/dialogue_sidon/compare_parity.py ref cpp
```

The released archives store CUDA tensors, so the reference needs a CUDA host.

## Limitations

- F32 GGUF only. No quantized package yet.
- Separation is limited to two speakers.
- Runtime and memory grow with chunk length. A 120 s chunk is about 6000 latent
  frames, and the transformer attends over all of them.
- The weights are CC-BY-NC-4.0, so non-commercial use only.

## Upstream

- [Model checkpoint and demo](https://huggingface.co/sarulab-speech/DialogueSidon)
- [Source](https://github.com/sarulab-speech/Sidon)
