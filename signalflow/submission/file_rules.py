"""The VCC2026 portal's file rules that live in a file's LABELS, not its values.

Shared by two callers, on purpose:

    predict.py        checks the labels it is ABOUT to write, before spending
                      any time computing predictions
    submit_vcc26.py   checks a finished file by opening only `.obs` / `.var`,
                      before `vcc prep` loads the whole matrix

That second use is the reason this is its own module. `vcc prep` reads the
entire file into RAM before it checks anything, so on a big prediction a wrong
label is discovered only after gigabytes have been loaded. Everything here is
readable from `.obs` and `.var` alone, in seconds.

Value rules (whole non-negative counts, per-cell cap, density) need the matrix
and are checked where it is written: `predict.CountsWriter`.
"""

from __future__ import annotations

import pandas as pd


def check_genes(var_names, genes) -> None:
    """Rule 5: `.var` is exactly the expected gene list, in order."""
    if list(var_names) != list(genes):
        raise SystemExit(
            f"rule 5: .var ({len(var_names):,} genes) is not the expected gene list "
            f"({len(genes):,} genes) in the same order"
        )


def check_layout(
    obs: pd.DataFrame,
    want_perts,
    pert_col: str,
    ctx_col: str,
    per_pert: int,
    control: str,
    expect_contexts=None,
    max_cells: int = 400_000,
) -> None:
    """Rules 1-4, 7 and the total-cell cap, from `.obs` alone. Raises on any."""
    for col in (pert_col, ctx_col):
        if col not in obs:
            raise SystemExit(f"rule 1/2: .obs is missing {col!r}")

    if len(obs) > max_cells:
        raise SystemExit(
            f"{len(obs):,} cells exceeds the {max_cells:,}-cell cap "
            f"(--max-cells); check nothing was concatenated twice"
        )

    got_ctx = set(obs[ctx_col].astype(str))
    if expect_contexts is not None and got_ctx != set(expect_contexts):
        raise SystemExit(f"rule 1: contexts {sorted(got_ctx)} != {sorted(expect_contexts)}")

    got_perts = set(obs[pert_col].astype(str))
    if control in got_perts:
        raise SystemExit(f"rule 7: {control!r} rows are present")
    if got_perts != set(want_perts):
        extra = sorted(got_perts - set(want_perts))[:3]
        missing = sorted(set(want_perts) - got_perts)[:3]
        raise SystemExit(f"rule 3: perturbations differ (extra {extra}, missing {missing})")

    sizes = obs.groupby([obs[ctx_col].astype(str), obs[pert_col].astype(str)], observed=True).size()
    bad = sizes[sizes != per_pert]
    if len(bad):
        raise SystemExit(
            f"rule 4: {len(bad)} (context, perturbation) group(s) are not "
            f"{per_pert} cells, e.g. {bad.index[0]} has {int(bad.iloc[0])}"
        )
