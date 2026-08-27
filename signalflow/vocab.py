"""Vocabularies.

Two *separate* index spaces, deliberately kept apart:

  GeneVocab  -- the readout space. Which genes we predict expression for.
                Union of the per-dataset panels (prototype: 1213 genes).

  PertVocab  -- the perturbation space. Which genes can be knocked out.
                Index 0 is reserved for "non-targeting" (the control).
                Backed by the full VCC26 symbol list (18,533), because the
                perturbed genes are mostly NOT in the readout panel:
                only 4 of the 80 prototype perturbations are.

Swapping in a bigger readout panel or a different perturbation list later is
just pointing these at a different csv. A one-column csv of symbols, or a csv
with a `gene_name` / `gene` column, both work.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

CONTROL_LABEL = "non-targeting"


def _read_symbols(path: str | Path) -> list[str]:
    df = pd.read_csv(path)
    for col in ("gene_name", "gene", "symbol"):
        if col in df.columns:
            names = df[col]
            break
    else:
        if df.shape[1] != 1:
            raise ValueError(
                f"{path}: expected a single column or one of "
                f"gene_name/gene/symbol, got {list(df.columns)}"
            )
        names = df.iloc[:, 0]
    # order-preserving dedup
    return list(dict.fromkeys(str(x) for x in names))


@dataclass
class Vocab:
    names: list[str]

    def __post_init__(self) -> None:
        self._idx = {n: i for i, n in enumerate(self.names)}
        if len(self._idx) != len(self.names):
            raise ValueError("vocabulary contains duplicates")

    def __len__(self) -> int:
        return len(self.names)

    def __contains__(self, name: str) -> bool:
        return name in self._idx

    def index(self, name: str) -> int:
        return self._idx[name]

    def indices(self, names) -> list[int]:
        return [self._idx[n] for n in names]

    def to_csv(self, path: str | Path, header: str) -> None:
        pd.DataFrame({header: self.names}).to_csv(path, index=False)

    @classmethod
    def from_csv(cls, path: str | Path) -> "Vocab":
        return cls(_read_symbols(path))


class GeneVocab(Vocab):
    """Readout gene space."""

    @classmethod
    def from_union(cls, panels) -> "GeneVocab":
        """Build from the gene panels actually present in the data."""
        seen: dict[str, None] = {}
        for panel in panels:
            for g in panel:
                seen.setdefault(str(g), None)
        return cls(sorted(seen))


class PertVocab(Vocab):
    """Perturbation space. Index 0 is always the control."""

    def __init__(self, names: list[str]) -> None:
        names = [n for n in names if n != CONTROL_LABEL]
        super().__init__([CONTROL_LABEL] + names)

    @property
    def control_index(self) -> int:
        return 0


def load_vocabs(processed_dir: str | Path) -> tuple[GeneVocab, PertVocab]:
    d = Path(processed_dir)
    meta = json.loads((d / "meta.json").read_text())
    genes = GeneVocab.from_csv(d / meta["gene_vocab"])
    perts = PertVocab.from_csv(d / meta["pert_vocab"])
    return genes, perts
