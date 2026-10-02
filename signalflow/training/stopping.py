"""Plateau detection for early stopping.

The cell-eval score is noisy: it is measured on a few thousand cells, and
several of its members swing by tenths from one epoch to the next. "Stop when it
has not set a new maximum" is therefore a poor rule -- one lucky epoch sets a bar
the model may never clear again, and a wobble upward of 0.001 counts as progress
and resets the clock. `PlateauStopper` adds the three things that make the
decision robust:

    window     average the last `window` measurements before comparing, so one
               lucky or unlucky scoring decides nothing (1 = use the raw value)
    min_delta  the smoothed value has to beat the best so far by at least this to
               count as progress, so a tiny wobble does not reset the clock
    patience   how many consecutive measurements without progress before stopping

Higher is better. To watch a loss, pass its negative.
"""

from __future__ import annotations

import math


class PlateauStopper:
    def __init__(self, patience: int, min_delta: float = 0.0, window: int = 1) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.patience, self.min_delta, self.window = int(patience), float(min_delta), int(window)
        self.values: list[float] = []
        self.best = -math.inf          # the REFERENCE: it only rises when a smoothed value beats it
        self.best_step = 0             #   by min_delta, so it can lag below values already seen
        self.top = -math.inf           # the highest SMOOTHED value seen so far (what "best" means to a reader)
        self.top_step = 0              # the step at which that highest value was seen
        self.smoothed = float("nan")   # the smoothed value of the latest measurement
        self.bad = 0                   # consecutive measurements without progress

    def update(self, value: float, step: int = 0) -> bool:
        """Record one measurement. Returns True if it counted as progress.

        A non-finite value is no progress and is kept out of the window, so a
        single failed scoring cannot poison the average.
        """
        if not math.isfinite(value):
            self.bad += 1
            return False
        self.values.append(value)
        recent = self.values[-self.window:]
        self.smoothed = sum(recent) / len(recent)
        if self.smoothed > self.top:
            self.top, self.top_step = self.smoothed, step
        if self.smoothed > self.best + self.min_delta:
            self.best, self.best_step, self.bad = self.smoothed, step, 0
            return True
        self.bad += 1
        return False

    @property
    def needs(self) -> float:
        """The smoothed value the next measurement has to EXCEED to count as progress."""
        return self.best + self.min_delta

    @property
    def should_stop(self) -> bool:
        return self.patience > 0 and self.bad >= self.patience
