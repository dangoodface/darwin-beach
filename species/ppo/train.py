"""PPO for the kernel's 515-parameter network. Rollouts on the CPU, updates on the GPU.

    systemd-run --user --scope -p MemoryMax=6G -p CPUQuota=200% \
      ~/.cache/arena-preflight/.venv/bin/python -m species.ppo.train --kernel K.wasm --out DIR

Each iteration plays `--episodes` full matches against the opponent (the coach, or an elite
genome), then updates the policy with the clipped PPO objective, GAE and an entropy bonus.
Every `--eval-every` updates the deterministic policy (the one that will be exported) plays
fixed matches against the coach on both sides, and the best checkpoint by points margin is kept.
Reward is the kernel's: +1 for a point won, -1 for a point lost. Nothing is shaped.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # the CPU quota is two threads' worth; do not fan out

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from . import policy as P  # noqa: E402
from .rollout import COACH, Episode, RolloutPool  # noqa: E402

torch.set_num_threads(1)


class Actor(nn.Module):
    """The kernel's network, as torch layers: Linear weights are (out, in) row-major, which is
    the genome's W1[32][12] and W2[3][32] layout, so `flat()` is the genome."""

    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Linear(P.N_IN, P.N_HID)
        self.l2 = nn.Linear(P.N_HID, P.N_OUT)
        for lin, gain in ((self.l1, 1.0), (self.l2, 0.01)):  # small output layer: start near 50% presses
            nn.init.orthogonal_(lin.weight, gain)
            nn.init.zeros_(lin.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.l1(x)
        return self.l2(h / (1.0 + h.abs()))

    def flat(self) -> np.ndarray:
        with torch.no_grad():
            return P.pack(self.l1.weight.cpu().numpy(), self.l1.bias.cpu().numpy(),
                          self.l2.weight.cpu().numpy(), self.l2.bias.cpu().numpy())

    def load_flat(self, params: np.ndarray) -> None:
        w1, b1, w2, b2 = P.unpack(params)
        with torch.no_grad():
            for t, a in ((self.l1.weight, w1), (self.l1.bias, b1), (self.l2.weight, w2), (self.l2.bias, b2)):
                t.copy_(torch.as_tensor(a, dtype=torch.float32))


def make_critic() -> nn.Module:
    return nn.Sequential(nn.Linear(P.N_IN, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))


def gae(rew: np.ndarray, val: np.ndarray, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Advantages and returns for one finished match (terminal value 0)."""
    adv = np.zeros_like(rew)
    last = 0.0
    for t in range(len(rew) - 1, -1, -1):
        nxt = val[t + 1] if t + 1 < len(rew) else 0.0
        delta = rew[t] + gamma * nxt - val[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + val


def evaluate(pool: RolloutPool, params: np.ndarray, opponent, matches: int, max_ticks: int = 3000) -> dict:
    """The deterministic policy (what gets exported) against `opponent`, both sides, fixed seeds."""
    jobs = [(params, opponent, 7000 + i, i % 2, max_ticks, i, True) for i in range(matches)]
    eps = pool.run(jobs)
    pf = sum(e.points_for for e in eps)
    pa = sum(e.points_against for e in eps)
    return {"win_rate": float(np.mean([e.points_for > e.points_against for e in eps])),
            "points_for": pf / matches, "points_against": pa / matches, "margin": (pf - pa) / matches,
            "mean_ticks": float(np.mean([e.ticks for e in eps]))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PPO for the slime volleyball network.")
    ap.add_argument("--kernel", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--elite", type=Path, help="a genome .bin to play against instead of the coach")
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--episodes", type=int, default=16, help="matches per update")
    ap.add_argument("--updates", type=int, default=100000)
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--eval-matches", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", type=Path)
    args = ap.parse_args(argv)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    actor, critic = Actor().to(dev), make_critic().to(dev)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=args.lr, eps=1e-5)
    start, total_ticks, best = 0, 0, -1e9
    if args.resume:
        ck = torch.load(args.resume, map_location=dev, weights_only=False)
        actor.load_flat(ck["params"].numpy())
        critic.load_state_dict(ck["critic"])
        opt.load_state_dict(ck["opt"])
        start, total_ticks, best = ck["update"], ck["ticks"], ck.get("best", best)
    args.out.mkdir(parents=True, exist_ok=True)
    opponent = np.fromfile(args.elite, dtype="<f4") if args.elite else COACH
    opp_name = "elite" if args.elite else "coach"
    log = open(args.out / "log.jsonl", "a")
    t_start = time.time()
    print(f"device {dev}, opponent {opp_name}, {args.procs} rollout processes, out {args.out}", flush=True)

    def save(name: str, update: int) -> None:
        torch.save({"params": torch.as_tensor(actor.flat()), "critic": critic.state_dict(), "opt": opt.state_dict(),
                    "update": update, "ticks": total_ticks, "best": best}, args.out / name)

    update = start - 1  # so the final save below is right even if the loop never runs
    with RolloutPool(args.kernel, args.procs) as pool:
        for update in range(start, args.updates):
            if args.max_hours and time.time() - t_start > args.max_hours * 3600:
                break
            t0 = time.time()
            params = actor.flat()
            jobs = [(params, opponent, int(rng.integers(1 << 31)), int(rng.integers(2)), 3000,
                     int(rng.integers(1 << 31))) for _ in range(args.episodes)]
            eps: list[Episode] = pool.run(jobs)
            t_roll = time.time() - t0

            obs = torch.as_tensor(np.concatenate([e.obs for e in eps]), device=dev)
            bits = torch.as_tensor(np.concatenate([e.bits for e in eps]), device=dev).float()
            old_logp = torch.as_tensor(np.concatenate([e.logp for e in eps]), device=dev)
            with torch.no_grad():
                values = critic(obs).squeeze(-1).cpu().numpy()
            advs, rets, pos = [], [], 0
            for e in eps:
                a, r = gae(e.reward, values[pos : pos + e.ticks], args.gamma, args.lam)
                advs.append(a)
                rets.append(r)
                pos += e.ticks
            adv = torch.as_tensor(np.concatenate(advs), device=dev)
            ret = torch.as_tensor(np.concatenate(rets), device=dev)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            n = obs.shape[0]
            total_ticks += n

            t1 = time.time()
            stats = {}
            for _ in range(args.epochs):
                perm = torch.randperm(n, device=dev)
                for i in range(0, n, args.minibatch):
                    idx = perm[i : i + args.minibatch]
                    lg = actor(obs[idx])
                    b = bits[idx]
                    logp = (b * F.logsigmoid(lg) + (1 - b) * F.logsigmoid(-lg)).sum(-1)
                    ratio = torch.exp(logp - old_logp[idx])
                    a = adv[idx]
                    pg = -torch.min(ratio * a, torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * a).mean()
                    vf = 0.5 * (critic(obs[idx]).squeeze(-1) - ret[idx]).pow(2).mean()
                    p = torch.sigmoid(lg)
                    ent = -(p * F.logsigmoid(lg) + (1 - p) * F.logsigmoid(-lg)).sum(-1).mean()
                    loss = pg + 0.5 * vf - args.ent * ent
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 0.5)
                    opt.step()
                    stats = {"pg": pg.item(), "vf": vf.item(), "entropy": ent.item(),
                             "kl": (old_logp[idx] - logp).mean().item()}
            t_upd = time.time() - t1

            row = {"update": update + 1, "ticks": total_ticks, "wall_s": round(time.time() - t_start, 1),
                   "roll_s": round(t_roll, 2), "upd_s": round(t_upd, 2), "batch": n,
                   "train_points_for": float(np.mean([e.points_for for e in eps])),
                   "train_points_against": float(np.mean([e.points_against for e in eps])),
                   "train_win_rate": float(np.mean([e.points_for > e.points_against for e in eps])),
                   **{k: round(v, 5) for k, v in stats.items()}}
            if (update + 1) % args.eval_every == 0:
                ev = evaluate(pool, actor.flat(), COACH, args.eval_matches)
                row["eval"] = {k: round(v, 3) for k, v in ev.items()}
                if ev["margin"] > best:
                    best = ev["margin"]
                    save("best.pt", update + 1)
                save("latest.pt", update + 1)
            log.write(json.dumps(row) + "\n")
            log.flush()
            if (update + 1) % 5 == 0 or "eval" in row:
                print(json.dumps(row), flush=True)
        save("latest.pt", update + 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
