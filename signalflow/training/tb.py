"""TensorBoard logging for a training run.

One `RunLog` per run writes to `<run_dir>/tb/`. View every run together with

    make tensorboard          (= tensorboard --logdir <out_dir>)

Curves are named `group/name`, so TensorBoard puts related ones in one panel:

    loss/train, loss/val, loss/val_identity   flow-matching loss (val_identity = predict-no-change floor)
    val/frac_var_explained                     share of the target's variance the flow explains
    cell_eval/avg_scaled                       the ONE average (higher is better)
    cell_eval/avg_smooth                       the smoothed value the early-stopping rule watches (vcc only)
    cell_eval/mean_shift_baseline, identity_baseline   the same average for the baselines, for reference
    cell_eval_scaled/<member>, cell_eval_raw/<member>  the six members, scaled (0 = no skill, 1 = perfect) / raw
    train/lr                                   learning rate

Cell-eval points appear only on the epochs where it is scored (`train.cell_eval_every`).
Logging is best-effort: if tensorboard cannot be imported the run continues without it.
"""

from __future__ import annotations

from pathlib import Path


class RunLog:
    def __init__(self, run_dir: Path) -> None:
        self.w = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.w = SummaryWriter(str(Path(run_dir) / "tb"))
        except Exception as e:  # pragma: no cover
            print(f"NOTE: tensorboard logging is off ({type(e).__name__}: {e})")

    def scalar(self, tag: str, value: float, step: int) -> None:
        if self.w is not None and value is not None and value == value:  # skips NaN
            self.w.add_scalar(tag, float(value), step)

    def epoch(self, entry: dict, lr: float | None = None) -> None:
        """Log one epoch's `entry` (the same dict that goes to history.json)."""
        ep = entry["epoch"]
        self.scalar("loss/train", entry.get("train_loss"), ep)
        self.scalar("loss/val", entry.get("val_loss"), ep)
        self.scalar("loss/val_identity", entry.get("val_identity_loss"), ep)
        self.scalar("val/frac_var_explained", entry.get("val_frac_var_explained"), ep)
        if lr is not None:
            self.scalar("train/lr", lr, ep)
        for k, v in entry.items():
            if not k.startswith("vcc_") or not isinstance(v, (int, float)) or k.startswith("vcc_n_"):
                continue
            name = k[4:]
            if name == "avg_score":
                self.scalar("cell_eval/avg_scaled", v, ep)
            elif name == "avg_smooth":
                self.scalar("cell_eval/avg_smooth", v, ep)
            elif name.startswith("s_"):
                self.scalar(f"cell_eval_scaled/{name[2:]}", v, ep)
            else:
                self.scalar(f"cell_eval_raw/{name}", v, ep)

    def baselines(self, ref: dict | None, step: int) -> None:
        for m, s in (ref or {}).items():
            self.scalar(f"cell_eval/{m}_baseline", s["avg_score"], step)

    def close(self) -> None:
        if self.w is not None:
            self.w.close()
