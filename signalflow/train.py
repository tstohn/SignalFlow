"""Train the conditional flow-matching velocity field.

    python -m signalflow.train --config configs/prototype.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from .data.dataset import FlowDataset, make_loader
from .flow import cfm_loss
from .models.velocity import VelocityField


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(ds: FlowDataset, mcfg: dict) -> VelocityField:
    return VelocityField(
        n_genes=ds.n_genes,
        n_perts=ds.n_perts,
        n_contexts=ds.n_contexts,
        n_state=ds.n_state,
        hidden=int(mcfg.get("hidden", 512)),
        n_blocks=int(mcfg.get("n_blocks", 3)),
        pert_dim=int(mcfg.get("pert_dim", 64)),
        state_dim=int(mcfg.get("state_dim", 64)),
        ctx_dim=int(mcfg.get("ctx_dim", 32)),
        time_dim=int(mcfg.get("time_dim", 32)),
        dropout=float(mcfg.get("dropout", 0.0)),
        head=mcfg.get("head", "plain"),
        gene_mask=torch.from_numpy(ds.gene_mask.astype("float32")),
    )


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def run_val(model, loader, sigma, device) -> dict[str, float]:
    model.eval()
    tot = {"loss": 0.0, "identity_loss": 0.0}
    n = 0
    for batch in loader:
        _, s = cfm_loss(model, to_device(batch, device), sigma)
        b = batch["x0"].shape[0]
        for k in tot:
            tot[k] += s[k] * b
        n += b
    model.train()
    out = {k: v / max(n, 1) for k, v in tot.items()}
    out["frac_var_explained"] = 1.0 - out["loss"] / max(out["identity_loss"], 1e-12)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    tcfg, mcfg = cfg["train"], cfg["model"]
    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)

    processed = cfg["data"]["processed_dir"]
    device = pick_device(args.device or tcfg.get("device", "auto"))
    epochs = args.epochs or int(tcfg.get("epochs", 30))
    bs = int(tcfg.get("batch_size", 256))
    sigma = float(tcfg.get("sigma", 0.0))

    train_ds = FlowDataset(processed, "train", seed=seed)
    # validation targets are held-out cells, but the control cells they flow
    # FROM also come from a held-out pool -- otherwise the source distribution
    # is one the model has memorised.
    val_ds = FlowDataset(processed, "val", control_split="val", seed=seed + 1)

    train_dl = make_loader(train_ds, bs, seed)
    val_dl = make_loader(val_ds, bs, seed, shuffle=False)

    model = build_model(train_ds, mcfg).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(
        f"device={device}  genes={train_ds.n_genes}  perts={train_ds.n_perts}  "
        f"contexts={train_ds.n_contexts}  state={train_ds.n_state}\n"
        f"train cells={len(train_ds)}  val cells={len(val_ds)}  "
        f"params={n_par/1e6:.2f}M  head={mcfg.get('head','plain')}\n"
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcfg.get("lr", 1e-3)),
        weight_decay=float(tcfg.get("weight_decay", 1e-4)),
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(epochs * len(train_dl), 1)
    )

    out_dir = Path(tcfg.get("out_dir", "runs/prototype"))
    out_dir.mkdir(parents=True, exist_ok=True)
    history, best = [], float("inf")

    for ep in range(1, epochs + 1):
        t0 = time.time()
        run, n = 0.0, 0
        for batch in train_dl:
            batch = to_device(batch, device)
            loss, _ = cfm_loss(model, batch, sigma)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            b = batch["x0"].shape[0]
            run += loss.item() * b
            n += b

        tr = run / max(n, 1)
        va = run_val(model, val_dl, sigma, device)
        history.append({"epoch": ep, "train_loss": tr, **{f"val_{k}": v for k, v in va.items()}})
        print(
            f"epoch {ep:3d}  train {tr:.4f}  val {va['loss']:.4f}  "
            f"(identity {va['identity_loss']:.4f}, "
            f"explained {va['frac_var_explained']*100:5.1f}%)  "
            f"{time.time()-t0:.1f}s"
        )

        if va["loss"] < best:
            best = va["loss"]
            torch.save(
                {"model": model.state_dict(), "config": cfg, "epoch": ep, "val": va},
                out_dir / "best.pt",
            )

    torch.save({"model": model.state_dict(), "config": cfg, "epoch": epochs}, out_dir / "last.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"\nbest val loss {best:.4f}  ->  {out_dir/'best.pt'}")


if __name__ == "__main__":
    main()
