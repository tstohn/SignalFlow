"""Package a prediction for the VCC26 portal, and upload it: three commands.

    package   predictions.h5ad  ->  prediction.vcc           local only; sends NOTHING anywhere
    upload    prediction.vcc    ->  the portal                sends it and starts scoring; asks first
    submit    predictions.h5ad  ->  .vcc  ->  the portal      both of the above, in one call

    python -m signalflow.submission.submit_vcc26 package --pred runs/prototype/predictions.h5ad
    python -m signalflow.submission.submit_vcc26 upload  --file runs/prototype/predictions.vcc -m "my model v1" --wait
    python -m signalflow.submission.submit_vcc26 submit  --pred runs/prototype/predictions.h5ad -m "my model v1" --wait

THIS SCRIPT IS FOR YOU TO RUN. Nothing in the repo runs it. `upload` and
`submit` send data to the portal, so each asks you to type `submit` first
(`--yes` skips the question).

WHY `package` AND `upload` EXIST SEPARATELY, WHEN `submit` DOES BOTH
    They differ in what they cost. Packaging is the slow, RAM-hungry step
    (`vcc prep` needs ~22 GiB for a full-size file) and it is local, free and
    repeatable. Uploading is fast but spends one of your daily submissions and
    puts a score on the leaderboard. Split, you can package once, look at the
    result, and upload later -- or upload the same `.vcc` from another machine --
    without ever packaging twice. `submit` is for when you do not need that.

WHAT THE PORTAL WANTS
    A single `.vcc` file covering every context. You never build one by hand:

        predict.py  ->  predictions.h5ad  --vcc prep-->  prediction.vcc  --vcc submit-->  portal

    `vcc prep` validates the `.h5ad` against the challenge format and packages
    it (a tar holding `meta.json` and a zstd-compressed `pred.h5ad`). The counts
    are preserved, gene order is fixed up if needed, and `.uns` is dropped.
    `vcc submit` uploads the `.vcc`. This script drives both through the `vcc`
    command-line tool.

    The format rules (all three contexts in one file, the exact 300
    perturbations, exactly 400 cells each, all 18,533 genes, raw whole-number
    counts, no control cells, <= 1e6 counts per cell, <= 400,000 cells,
    <= 4.75e9 stored entries) are enforced by `predict.py` when it writes the
    file, checked again by `package`, and a third time by `vcc prep` itself.

WHY `package` HAS A PRE-FLIGHT
    `vcc prep` loads the ENTIRE `.h5ad` into RAM before it checks anything
    (`ad.read_h5ad`, no backed mode), and only warns about memory afterwards.
    An oversized file on too small a machine is killed by the OS with no
    message. So `package` first opens only the labels and the matrix header
    -- seconds, no matrix -- and checks the labels, the stored-entry count and
    whether packaging will fit in this machine's RAM, before prep loads a byte.

WHERE `vcc` IS
    A `uv tool` installs it in its own environment, not in this venv. The
    script looks for it next to this Python, then on PATH, then in
    `~/.local/bin` (where `uv tool` puts it); `--vcc-bin` overrides all three.
    Login is not handled here: `vcc login`, or `export VCC_TOKEN=...`, as the
    challenge's CLI guide describes.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import anndata as ad
import h5py
import pandas as pd

from .file_rules import check_genes, check_layout

HARD_CAP_NNZ = 4_750_000_000
GIB = float(2**30)

# Mirror of vcc/sizing.py::prep_peak_gib for the file predict.py writes: sparse
# float32 counts, genes already in the required order, so packaging makes no
# cast copy and no reordering copy. vcc's own function is the authority, but it
# runs only AFTER the file has been loaded -- too late to matter when the load
# itself is what runs out of memory.
PREP_COPIES = 1.3
PREP_COPIES_CAST = 2.2
PREP_OVERHEAD_GIB = 2.0


def prep_peak_gib(nnz: int, casts: bool = False) -> float:
    """Rough peak RAM `vcc prep` needs to package a matrix with `nnz` entries.

    SciPy indexes CSR with int32 while it can and promotes to int64 above 2**31
    entries, so the cost per entry steps from 8 bytes to 12 exactly there.
    """
    per_nnz = 8 if nnz <= 2**31 else 12
    copies = PREP_COPIES_CAST if casts else PREP_COPIES
    return copies * per_nnz * nnz / GIB + PREP_OVERHEAD_GIB


def total_ram_gib() -> float | None:
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / GIB
    except (ValueError, OSError, AttributeError):
        return None


def find_vcc(explicit: str | None) -> str:
    """The `vcc` executable: --vcc, else next to this Python, else PATH, else ~/.local/bin."""
    candidates = [
        explicit,
        str(Path(sys.executable).parent / "vcc"),
        shutil.which("vcc"),
        str(Path.home() / ".local" / "bin" / "vcc"),
    ]
    for c in candidates:
        if c and Path(c).is_file() and os.access(c, os.X_OK):
            return c
    raise SystemExit(
        "cannot find the `vcc` command. Install it with `uv tool install vcc-cli` "
        "(then `uv tool update-shell`), or pass its path with --vcc."
    )


def read_controls(controls: Path) -> dict:
    """The three files the portal's controls bundle ships."""
    need = {n: controls / n for n in ("manifest.json", "gene_names.csv", "pert_counts.csv")}
    absent = [n for n, p in need.items() if not p.is_file()]
    if absent:
        raise SystemExit(f"{controls}: missing {', '.join(absent)} (the controls bundle's files)")
    manifest = json.loads(need["manifest.json"].read_text())
    return dict(
        manifest=manifest,
        genes_path=need["gene_names.csv"],
        perts_path=need["pert_counts.csv"],
        genes=pd.read_csv(need["gene_names.csv"]).iloc[:, 0].astype(str).tolist(),
        perts=pd.read_csv(need["pert_counts.csv"]).iloc[:, 0].astype(str).tolist(),
    )


def preflight(pred: Path, ctl: dict, ignore_memory: bool) -> dict:
    """Everything checkable from labels and the matrix header alone. Raises on any."""
    m = ctl["manifest"]
    if not pred.is_file():
        raise SystemExit(f"--pred {pred} does not exist")

    a = ad.read_h5ad(pred, backed="r")            # obs / var only; no matrix loaded
    try:
        if a.uns.get("signalflow", {}).get("partial"):
            raise SystemExit(
                f"{pred.name} was written with --dry-run-perts, so it holds only some of the "
                f"perturbations; the portal would reject it. Run predict.py without that flag."
            )
        check_genes(a.var_names, ctl["genes"])
        check_layout(
            a.obs,
            want_perts=ctl["perts"],
            pert_col=m["pert_col"],
            ctx_col=m["context_col"],
            per_pert=int(m["cells_per_pert"]),
            control=m["control_label"],
            expect_contexts=m["contexts"],
        )
        n_cells = a.n_obs
    finally:
        a.file.close()

    with h5py.File(pred, "r") as f:
        if "X" not in f or not isinstance(f["X"], h5py.Group):
            raise SystemExit("X must be a sparse matrix (a dense array is over the density cap on its own)")
        enc = f["X"].attrs.get("encoding-type", "")
        if enc != "csr_matrix":
            raise SystemExit(f"X is stored as {enc!r}; predict.py writes csr_matrix")
        nnz = int(f["X/indptr"][-1])
        dtype = f["X/data"].dtype

    if nnz > HARD_CAP_NNZ:
        raise SystemExit(
            f"{nnz:,} stored entries ({nnz / n_cells:,.0f} per cell) exceeds the portal's "
            f"{HARD_CAP_NNZ:,} limit. That is a density limit, not a file-size one: "
            f"the model predicts too many expressed genes per cell."
        )

    casts = dtype != "float32"
    need, have = prep_peak_gib(nnz, casts=casts), total_ram_gib()
    print(
        f"  pre-flight OK: {n_cells:,} cells, {nnz:,} stored entries ({nnz / n_cells:,.0f} per cell, "
        f"{nnz / HARD_CAP_NNZ:.1%} of the portal cap), X is {dtype}"
    )
    print(
        f"  packaging with `vcc prep` needs roughly {need:.0f} GiB of RAM"
        + (f"; this machine has {have:.0f} GiB" if have else "")
    )
    if have is not None and need > have:
        msg = (
            f"packaging this file needs about {need:.0f} GiB of RAM but this machine has "
            f"{have:.0f} GiB. `vcc prep` loads the whole matrix first, so it would be killed by the "
            f"OS with no message. Either run this on a machine with more RAM (the file can be "
            f"copied as-is), or make the model predict fewer expressed genes per cell."
        )
        if not ignore_memory:
            raise SystemExit(f"{msg}\n  (--ignore-memory to try anyway, e.g. if you have swap)")
        print(f"  WARNING: {msg}\n  --ignore-memory given: continuing.")
    return dict(n_cells=n_cells, nnz=nnz)


def prep_command(vcc: str, pred: Path, ctl: dict, out: Path, force: bool, dry_run: bool) -> list[str]:
    """`vcc prep`, with long flags only: `-p` means --pert-col here but --perts in `vcc sample`."""
    m = ctl["manifest"]
    cmd = [
        vcc, "prep", str(pred),
        "--genes", str(ctl["genes_path"]),
        "--perts", str(ctl["perts_path"]),
        "--output", str(out),
        "--pert-col", m["pert_col"],
        "--context-col", m["context_col"],
        "--contexts", ",".join(m["contexts"]),
        "--cells-per-pert", str(m["cells_per_pert"]),
        "--ntc-name", m["control_label"],
    ]
    if force:
        cmd.append("--force")
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def submit_command(vcc: str, vcc_file: Path, name: str, description: str | None, wait: bool) -> list[str]:
    cmd = [vcc, "submit", str(vcc_file), "--model-name", name]
    if description:
        cmd += ["--description", description]
    if wait:
        cmd.append("--wait")
    return cmd


def confirm_upload(vcc_file: Path, name: str, assume_yes: bool) -> None:
    print(
        "\nAbout to UPLOAD and start scoring:\n"
        f"  file   {vcc_file}  ({vcc_file.stat().st_size / 1e9:.2f} GB)\n"
        f"  model  {name}\n"
        "  A submission that reaches scoring counts against your daily limit (renews at\n"
        "  midnight UTC), and your team can have only one submission in flight at a time."
    )
    if assume_yes:
        print("  --yes given: not asking.")
        return
    if not sys.stdin.isatty():
        raise SystemExit("refusing to upload without a terminal to confirm on; pass --yes to skip the question")
    if input("Type 'submit' to upload, anything else to stop: ").strip() != "submit":
        raise SystemExit("stopped; nothing was uploaded.")


def package(args, vcc: str, dry_run: bool = False) -> Path:
    """Preflight, then `vcc prep`. Local only. Returns the .vcc path."""
    pred = Path(args.pred)
    out = Path(args.out) if args.out else pred.with_suffix(".vcc")
    ctl = read_controls(Path(args.controls))
    print(f"  contexts {ctl['manifest']['contexts']}, {len(ctl['perts'])} perturbations, "
          f"{ctl['manifest']['cells_per_pert']} cells each, {len(ctl['genes']):,} genes")
    preflight(pred, ctl, args.ignore_memory)

    cmd = prep_command(vcc, pred, ctl, out, args.force, dry_run)
    print("\n  $ " + " ".join(cmd) + "\n")
    if subprocess.run(cmd).returncode != 0:
        raise SystemExit("`vcc prep` failed; nothing was written or uploaded")
    return out


def upload(args, vcc: str, vcc_file: Path) -> int:
    """Confirm, then `vcc submit`. This is the step that sends data to the portal."""
    if not vcc_file.is_file():
        raise SystemExit(f"{vcc_file} does not exist; run `package` first")
    confirm_upload(vcc_file, args.model_name, args.yes)
    cmd = submit_command(vcc, vcc_file, args.model_name, args.description, args.wait)
    print("\n  $ " + " ".join(cmd) + "\n")
    return subprocess.run(cmd).returncode


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True, metavar="{package,upload,submit}")

    binary = argparse.ArgumentParser(add_help=False)
    binary.add_argument("--vcc-bin", default=None, help="path to the `vcc` executable (default: found automatically)")

    make = argparse.ArgumentParser(add_help=False)
    make.add_argument("--pred", required=True, help="the .h5ad written by predict.py")
    make.add_argument("--controls", default="data/VCC26/controls", help="the controls bundle folder")
    make.add_argument("--out", default=None, help="the .vcc to write [default: next to --pred]")
    make.add_argument("--ignore-memory", action="store_true", help="package even if the RAM estimate exceeds this machine's")
    make.add_argument("--force", action="store_true", help="overwrite an existing .vcc")

    send = argparse.ArgumentParser(add_help=False)
    send.add_argument("-m", "--model-name", required=True, help="the leaderboard name")
    send.add_argument("-d", "--description", default=None)
    send.add_argument("--wait", action="store_true", help="block until scoring finishes")
    send.add_argument("--yes", action="store_true", help="skip the confirmation question")

    pk = sub.add_parser("package", parents=[binary, make],
                        help="predictions.h5ad -> .vcc, on this machine; sends nothing",
                        description="Validate the predictions and write the .vcc. Local only; nothing is uploaded.")
    pk.add_argument("--dry-run", action="store_true", help="validate only (`vcc prep --dry-run`); write nothing")

    up = sub.add_parser("upload", parents=[binary, send],
                        help="send an existing .vcc to the portal and start scoring",
                        description="Upload a .vcc made earlier by `package`. Does not package. Asks before sending.")
    up.add_argument("--file", required=True, help="the .vcc file to upload")

    sub.add_parser("submit", parents=[binary, make, send],
                   help="package, then upload: both, in one call",
                   description="`package` followed by `upload`. Asks before sending.")
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    vcc = find_vcc(args.vcc_bin)

    if args.command == "package":
        out = package(args, vcc, dry_run=args.dry_run)
        if args.dry_run:
            print("\ndry run passed; nothing was written.")
        else:
            print(f"\npackaged -> {out}\nnothing has been uploaded. To upload it:\n"
                  f"  python -m signalflow.submission.submit_vcc26 upload --file {out} -m \"<model name>\" --wait")
    elif args.command == "upload":
        raise SystemExit(upload(args, vcc, Path(args.file)))
    else:  # submit
        out = package(args, vcc)
        raise SystemExit(upload(args, vcc, out))


if __name__ == "__main__":
    main()
