#!/usr/bin/env python3
"""Compare audio.cpp DialogueSidon stage dumps against python_reference.py.

    DIALOGUE_SIDON_DUMP_DIR=cpp audiocpp_cli --task sep --family dialogue_sidon ...
    python tests/dialogue_sidon/python_reference.py ... --noise cpp/noise.f32 --output-dir ref
    python tests/dialogue_sidon/compare_parity.py ref cpp

Prints max_abs and relative L2 error for every stage both directories contain.
Pass --noise so both sides start from the same latent; otherwise only the stages
up to the conditioning are comparable.
"""

import argparse
import math
from pathlib import Path
import struct

STAGES = ["wav16k_padded", "input_features", "attention_mask", "features", "predicted", "conditioning", "noise"]
FINAL = ["latents_final", "wav_speaker0", "wav_speaker1"]


def load(directory, name):
    path = directory / f"{name}.f32"
    if not path.exists():
        return None
    raw = path.read_bytes()
    return struct.unpack(f"<{len(raw) // 4}f", raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()

    steps = sorted(path.stem for path in args.reference.glob("step_*.f32"))
    for name in STAGES + steps + FINAL:
        ref, got = load(args.reference, name), load(args.candidate, name)
        if ref is None or got is None:
            continue
        if len(ref) != len(got):
            print(f"{name:24s} size mismatch: reference {len(ref)}, candidate {len(got)}")
            continue
        err = [a - b for a, b in zip(ref, got)]
        norm = math.sqrt(sum(a * a for a in ref))
        rel = math.sqrt(sum(e * e for e in err)) / max(norm, 1e-12)
        print(f"{name:24s} n={len(ref):8d} max_abs={max(map(abs, err)):.3e} rel_l2={rel:.3e}")


if __name__ == "__main__":
    main()
