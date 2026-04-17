"""
RetailFlow AI — Behavioural Intent Predictor
Classifies customer movement intent using dwell time and velocity heuristics.
"""

import numpy as np
from collections import deque


class IntentPredictor:
    """
    Classifies each tracked person's behavioural intent based on movement
    velocity and dwell time.

    Intent categories
    -----------------
    high_intent  : Long dwell + low movement  — potential purchaser
    browsing     : Moderate dwell + moderate movement
    transiting   : High velocity — passing through
    """

    HISTORY_LEN: int = 10
    VELOCITY_LOW: float = 2.0    # px/frame — threshold for "stationary"
    VELOCITY_HIGH: float = 5.5   # px/frame — threshold for "transit"
    DWELL_BUYER: float = 30.0    # seconds

    _LABELS = {
        "high_intent": "High Intent",
        "browsing": "Browsing",
        "transiting": "Transiting",
    }

    def __init__(self):
        self._history: dict[int, deque] = {}

    # ── Public API ─────────────────────────────────────────────────────────

    def update(self, track_id: int, center: list[float]) -> None:
        """Record the latest centre point for a given track."""
        if track_id not in self._history:
            self._history[track_id] = deque(maxlen=self.HISTORY_LEN)
        self._history[track_id].append(center)

    def predict(self, track_id: int, dwell_time: float) -> str:
        """
        Returns a human-readable intent label for the given track.

        Parameters
        ----------
        track_id   : stable tracking ID
        dwell_time : seconds since this track was first observed
        """
        velocity = self._velocity(track_id)

        if dwell_time >= self.DWELL_BUYER and velocity < self.VELOCITY_LOW:
            key = "high_intent"
        elif velocity >= self.VELOCITY_HIGH:
            key = "transiting"
        else:
            key = "browsing"

        return self._LABELS[key]

    def evict(self, active_ids: set[int]) -> None:
        """Remove history entries for tracks that are no longer active."""
        stale = [tid for tid in self._history if tid not in active_ids]
        for tid in stale:
            del self._history[tid]

    # ── Private ────────────────────────────────────────────────────────────

    def _velocity(self, track_id: int) -> float:
        """Mean Euclidean distance between consecutive recorded positions."""
        pts = self._history.get(track_id)
        if not pts or len(pts) < 2:
            return 0.0
        distances = [
            float(np.linalg.norm(np.array(pts[i]) - np.array(pts[i - 1])))
            for i in range(1, len(pts))
        ]
        return sum(distances) / len(distances)
