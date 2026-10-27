"""Export a trained policy as a genome the arena accepts, and prove the kernel agrees.

A genome is 515 f32 values, little-endian, nothing else (docs/DATA.md), stored as
<genome_id>.bin where the id hashes the bytes with the kernel sha. The trained
network is stochastic; the genome plays its mode: an action bit is pressed when its
output is above zero, which is the kernel's own rule.

    python -m species.ppo.export CHECKPOINT OUT_DIR [--kernel PATH]

A checkpoint is a .npy or .bin of 515 f32 values, or a torch .pt holding {"params": ...}.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from arena.genome import genome_id  # noqa: E402
from arena.kernel import Kernel, find_kernel  # noqa: E402

from . import policy as P  # noqa: E402


def load_params(path: Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".pt":
        import torch

        obj = torch.load(path, map_location="cpu", weights_only=True)
        arr = obj["params"] if isinstance(obj, dict) else obj
        params = np.asarray(arr.detach().cpu().numpy() if hasattr(arr, "detach") else arr, dtype=np.float32)
    elif path.suffix == ".npy":
        params = np.load(path).astype(np.float32)
    else:
        params = np.fromfile(path, dtype="<f4")
    if params.shape != (P.GENOME_LEN,):
        raise ValueError(f"a genome has {P.GENOME_LEN} parameters, {path} has {params.shape}")
    if not np.isfinite(params).all():
        raise ValueError("the parameters contain NaN or infinity")
    return params


def export_genome(params: np.ndarray, out_dir: Path, kernel_path: Path) -> Path:
    params = np.ascontiguousarray(params, dtype="<f4")
    if params.shape != (P.GENOME_LEN,):
        raise ValueError(f"a genome has {P.GENOME_LEN} parameters, got {params.shape}")
    sha = Kernel(kernel_path).kernel_sha
    out = Path(out_dir) / f"{genome_id(params, sha)}.bin"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(params.tobytes())
    return out


def verify_against_kernel(genome_bin: Path, kernel_path: Path, n: int = 1000, seed: int = 0) -> int:
    """Walk random play through the kernel and, at `n` states (both sides alternately), check that
    its policy_action equals the mode of our policy on the mirrored observation, computed with the
    kernel's own arithmetic. Returns the number of states checked; raises on the first difference."""
    params = load_params(genome_bin)
    net = P.unpack(params)
    k = Kernel(kernel_path)
    k.load_genomes(params[None, :])
    ex, st = k._exports, k._store
    from .rollout import MatchRunner  # for the zero-copy state view

    runner = MatchRunner(kernel_path)
    runner.k.load_genomes(params[None, :])
    state = runner.state
    rng = np.random.default_rng(seed)
    checked, match = 0, 0
    while checked < n:
        runner._reset(runner.store, int(rng.integers(1 << 32)))
        feed = P.ObsFeed()
        match += 1
        while checked < n and state[P.LIVES_A] > 0 and state[P.LIVES_B] > 0 and state[P.TICK] < 3000:
            for side in (0, 1):  # both sides every tick: the feed's stale rule needs both refreshed
                want = P.bits_to_action(P.mode_bits(P.logits_exact(feed.get(state, side), net)))
                got = runner._policy(runner.store, side, 0)
                if got != want:
                    raise AssertionError(f"state {checked} (match {match}, tick {state[P.TICK]:.0f}, side {side}): "
                                         f"kernel presses {got:03b}, the policy's mode is {want:03b}")
                checked += 1
            runner._set(runner.store, int(rng.integers(8)), int(rng.integers(8)))
            feed.after_step(runner._step(runner.store))
    return checked


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Export a trained policy as a genome .bin and verify it.")
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--kernel", type=Path, default=find_kernel())
    ap.add_argument("--states", type=int, default=1000)
    args = ap.parse_args(argv)
    if args.kernel is None:
        print("no kernel: set ARENA_KERNEL or pass --kernel")
        return 2
    out = export_genome(load_params(args.checkpoint), args.out_dir, args.kernel)
    n = verify_against_kernel(out, args.kernel, args.states)
    print(f"{out}: kernel policy_action equals the policy's mode on {n} states")
    return 0


if __name__ == "__main__":
    sys.exit(main())
