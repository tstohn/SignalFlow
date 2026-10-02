#!/usr/bin/env python3
"""
Training curves as ONE PNG from runs/<run>/history.json (written by train.py after every epoch).

    .venv/bin/python overview_scripts/plot_training.py                       # newest run under runs/
    .venv/bin/python overview_scripts/plot_training.py runs/v1_0_2/holdout_VCC25__adata_Validation
    .venv/bin/python overview_scripts/plot_training.py runs/v1_0_1/holdout_* runs/v1_0_2/holdout_*   # overlay runs
    .venv/bin/python overview_scripts/plot_training.py --watch 60            # redraw every 60 s during training

Writes <first run dir>/training_curves.png (--out to change), atomically, so the file is never half
written. Open it in VS Code Remote: an open image tab reloads when the file changes, no browser or
port forwarding needed. One panel per metric, one line per run (never two y-axes); cell-eval metrics
exist only on the epochs where cell-eval ran, so they are drawn as points joined by a thin line.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]   # fixed order, one per run
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

# (key, title, better) -- better: "up", "down" or None (not asserted)
PANELS = [
    ("train_loss", "train loss", "down"),
    ("val_loss", "validation loss", "down"),
    ("val_frac_var_explained", "validation: fraction of variance explained", "up"),
    ("vcc_avg_score", "cell-eval average score", "up"),
    ("vcc_pds_cosine", "cell-eval pds_cosine", None),
    ("vcc_dir_fidelity", "cell-eval dir_fidelity", None),
    ("vcc_dir_reach", "cell-eval dir_reach", None),
    ("vcc_sig_jaccard", "cell-eval sig_jaccard", None),
    ("vcc_lfc_nmae", "cell-eval lfc_nmae", "down"),
    ("vcc_expr_mse", "cell-eval expr_mse", "down"),
]
GAINS = [("gain_d", "g_d"), ("gain_s", "g_s"), ("gain_m", "g_m")]


def newest(root: Path) -> list[Path]:
    hs = sorted(root.glob("**/history.json"), key=lambda p: p.stat().st_mtime)
    return hs[-1:]


def find_history(arg: str | None) -> list[Path]:
    """The history.json for a run folder / file; a folder without its own history.json (e.g. runs/<version>,
    which holds holdout_*/ and full/) resolves to the newest one beneath it. [] = none yet."""
    if not arg:
        return newest(REPO / "runs")
    p = Path(arg)
    if p.is_dir():
        return [p / "history.json"] if (p / "history.json").exists() else newest(p)
    return [p]


def label(p: Path) -> str:
    try:
        return str(p.parent.resolve().relative_to(REPO / "runs"))
    except ValueError:
        return p.parent.name


def load(p: Path) -> list[dict]:
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):          # caught mid-write by a watching run
        return []


def series(hist: list[dict], key: str) -> tuple[list, list]:
    pts = [(e["epoch"], e[key]) for e in hist if e.get(key) is not None]
    return [p[0] for p in pts], [p[1] for p in pts]


def draw(paths: list[Path], out: Path) -> str:
    runs = [(label(p), load(p)) for p in paths]
    fig, axes = plt.subplots(3, 4, figsize=(17, 9.5), dpi=110)
    fig.patch.set_facecolor("white")
    flat = list(axes.flat)
    for ax, (key, title, better) in zip(flat, PANELS):
        drawn = False
        for i, (name, hist) in enumerate(runs):
            x, y = series(hist, key)
            if not x:
                continue
            drawn = True
            c = SERIES[i % len(SERIES)]
            ax.plot(x, y, color=c, lw=1.6, marker="o" if key.startswith("vcc_") else None, ms=4, label=name)
            if key == "vcc_avg_score":
                xs, ys = series(hist, "vcc_avg_smooth")
                if xs:
                    ax.plot(xs, ys, color=c, lw=3, alpha=0.35)
            if key == "val_loss" and i == 0:
                xi, yi = series(hist, "val_identity_loss")
                if xi:
                    ax.plot(xi, yi, color=MUTED, lw=1, ls="--", label="identity baseline")
        if key == "vcc_avg_score":
            ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(title + {"up": "  (higher is better)", "down": "  (lower is better)", None: ""}[better],
                     fontsize=9, color=INK, loc="left")
        if not drawn:
            ax.text(0.5, 0.5, "no data yet", ha="center", va="center", color=MUTED, fontsize=9, transform=ax.transAxes)
    # last panels: anchor gains of the first run that has them, then the legend
    ax = flat[len(PANELS)]
    hist = next((h for _, h in runs if any("gain_d" in e for e in h)), [])
    for j, (k, name) in enumerate(GAINS):
        x, y = series(hist, k)
        if x:
            ax.plot(x, y, color=SERIES[j], lw=1.6, label=name)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_title("anchor gains (first run with an anchored model)", fontsize=9, color=INK, loc="left")
    if not hist:
        ax.text(0.5, 0.5, "no data yet", ha="center", va="center", color=MUTED, fontsize=9, transform=ax.transAxes)
    else:
        ax.legend(frameon=False, fontsize=8)
    leg = flat[len(PANELS) + 1]
    leg.axis("off")
    latest = []
    for i, (name, hist) in enumerate(runs):
        leg.plot([], [], color=SERIES[i % len(SERIES)], lw=2, label=f"{name}  ({len(hist)} epochs)")
        if hist:
            e = hist[-1]
            latest.append(f"{name}: epoch {e['epoch']}  train {e.get('train_loss', float('nan')):.3f}"
                          + (f"  val {e['val_loss']:.3f}" if e.get("val_loss") is not None else ""))
    leg.legend(loc="upper left", frameon=False, fontsize=9, title="runs", title_fontsize=9)
    leg.text(0, 0.35, "\n".join(latest), fontsize=8, color=MUTED, va="top", transform=leg.transAxes)
    for ax in flat[: len(PANELS) + 1]:
        ax.set_xlabel("epoch", fontsize=8, color=MUTED)
        ax.grid(True, color=GRID, lw=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8, colors=MUTED)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
    fig.suptitle(f"SignalFlow training  --  updated {time.strftime('%Y-%m-%d %H:%M:%S')}", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    tmp = out.with_name(out.name + ".tmp")
    fig.savefig(tmp, format="png")
    plt.close(fig)
    os.replace(tmp, out)
    return "; ".join(latest) or "no epochs yet"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run folders or history.json files (default: the newest run)")
    ap.add_argument("--out", default=None, help="output PNG (default: <first run>/training_curves.png)")
    ap.add_argument("--watch", type=int, default=0, metavar="SECONDS", help="redraw every N seconds until Ctrl-C")
    args = ap.parse_args()
    while True:
        paths = [q for r in args.runs for q in find_history(r)] if args.runs else find_history(None)
        paths = [p for p in paths if p.exists()]
        if paths:
            out = Path(args.out) if args.out else paths[0].parent / "training_curves.png"
            print(f"{time.strftime('%H:%M:%S')}  {out}  |  {draw(paths, out)}", flush=True)
        elif args.watch:
            print(f"{time.strftime('%H:%M:%S')}  waiting for the first epoch (no history.json yet) ...", flush=True)
        else:
            raise SystemExit("no history.json yet -- it appears after the first epoch")
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
