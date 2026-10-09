"""Causal expression decisions from anonymized-frame probabilities only."""

from __future__ import annotations

import math

import numpy as np


class TemporalExpressionFilter:
    """Suppress isolated guesses without hiding sustained expression changes.

    Durations use capture/video timestamps, not the processing frame count.
    The returned confidence is the averaged probability of the reported class;
    it is not a vote percentage or an artificially increased score.
    """

    def __init__(self, *, time_constant=0.30, min_confidence=0.50,
                 min_margin=0.12, hold_confidence=0.30, hold_margin=0.03,
                 switch_seconds=0.25, initial_seconds=0.10,
                 uncertain_seconds=0.90, reset_gap=0.75, min_observations=2):
        self.time_constant = float(time_constant)
        self.min_confidence = float(min_confidence)
        self.min_margin = float(min_margin)
        self.hold_confidence = float(hold_confidence)
        self.hold_margin = float(hold_margin)
        self.switch_seconds = float(switch_seconds)
        self.initial_seconds = float(initial_seconds)
        self.uncertain_seconds = float(uncertain_seconds)
        self.reset_gap = float(reset_gap)
        self.min_observations = int(min_observations)
        if (not all(math.isfinite(v) for v in (self.time_constant, self.min_confidence,
                self.min_margin, self.switch_seconds, self.initial_seconds,
                self.uncertain_seconds, self.reset_gap,
                self.hold_confidence, self.hold_margin))
                or self.time_constant <= 0 or self.reset_gap <= 0
                or min(self.switch_seconds, self.initial_seconds, self.uncertain_seconds) < 0
                or not 0 < self.min_confidence <= 1 or not 0 <= self.min_margin <= 1
                or not 0 < self.hold_confidence <= self.min_confidence
                or not 0 <= self.hold_margin <= self.min_margin
                or self.min_observations < 2):
            raise ValueError("Invalid temporal expression filter parameters")
        self.reset()

    def reset(self):
        self.probabilities = None
        self.label = None
        self.last_timestamp = None
        self.pending_label = None
        self.pending_since = None
        self.pending_count = 0
        self.unsupported_since = None

    def update(self, probabilities, timestamp: float) -> dict:
        values = np.asarray(probabilities, dtype=np.float64)
        timestamp = float(timestamp)
        if (values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all()
                or (values < 0).any() or values.sum() <= 0
                or not math.isfinite(timestamp)):
            raise ValueError("Expected finite nonnegative class probabilities and timestamp")
        values = values / values.sum()
        if (self.last_timestamp is not None and
                (timestamp <= self.last_timestamp or timestamp - self.last_timestamp > self.reset_gap)):
            self.reset()
        if self.probabilities is None:
            self.probabilities = values.copy()
        else:
            if len(values) != len(self.probabilities):
                raise ValueError("Expression class count changed within a track")
            alpha = -math.expm1(-(timestamp - self.last_timestamp) / self.time_constant)
            self.probabilities += alpha * (values - self.probabilities)
        self.last_timestamp = timestamp
        best = int(self.probabilities.argmax())
        second = float(np.partition(self.probabilities, -2)[-2])
        margin = float(self.probabilities[best] - second)
        # Current-frame support prevents the averaging tail from counting as
        # fresh evidence when the face or expression has already changed.
        supported = (self.probabilities[best] >= self.min_confidence
                     and margin >= self.min_margin
                     and values.argmax() == best and values[best] >= self.min_confidence)
        if supported and best != self.label:
            if self.pending_label != best:
                self.pending_label, self.pending_since, self.pending_count = best, timestamp, 1
            else:
                self.pending_count += 1
            dwell = self.initial_seconds if self.label is None else self.switch_seconds
            if self.pending_count >= self.min_observations and timestamp - self.pending_since >= dwell:
                self.label = best
                self.pending_label = self.pending_since = None
                self.pending_count = 0
                self.unsupported_since = None
        elif best == self.label:
            self.pending_label = self.pending_since = None
            self.pending_count = 0
        else:
            self.pending_label = self.pending_since = None
            self.pending_count = 0
        if self.label is not None:
            # A confirmed label is held through a few weak/noisy frames.  A
            # competing class still needs the strict entry gate above; it must
            # not erase the last usable result while it is accumulating.
            hold_best = int(self.probabilities.argmax()) == self.label
            hold_supported = (hold_best
                              and self.probabilities[self.label] >= self.hold_confidence
                              and margin >= self.hold_margin)
            competing_transition = self.pending_label is not None and self.pending_label != self.label
            if (hold_supported or competing_transition):
                self.unsupported_since = None
            elif self.unsupported_since is None:
                self.unsupported_since = timestamp
            elif timestamp - self.unsupported_since >= self.uncertain_seconds:
                self.label = None
                self.pending_label = self.pending_since = None
                self.pending_count = 0
                self.unsupported_since = None
        return {
            "label": self.label,
            "confidence": None if self.label is None else float(self.probabilities[self.label]),
            "probabilities": self.probabilities.tolist(),
            "state": "uncertain" if self.label is None else "transition" if self.pending_label is not None else "stable" if supported else "low_support",
            "margin": margin,
        }
