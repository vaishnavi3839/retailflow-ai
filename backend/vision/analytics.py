"""
RetailFlow AI — Zone Analytics
Defines spatial zones in normalised coordinates, counts occupancy,
raises queue alerts, and renders overlays on video frames.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Zone definitions — normalised [x1, y1, x2, y2] in 0.0–1.0 space
# ---------------------------------------------------------------------------

DEFAULT_ZONES: dict[str, list[float]] = {
    "Aisle A — Snacks": [0.05, 0.10, 0.38, 0.80],
    "Aisle B — Beverages": [0.45, 0.10, 0.82, 0.55],
    "Checkout": [0.60, 0.65, 0.92, 0.92],
}

# Colours per zone: BGR tuples used by OpenCV
ZONE_COLOURS: dict[str, tuple] = {
    "Aisle A — Snacks": (0, 200, 140),
    "Aisle B — Beverages": (0, 160, 220),
    "Checkout": (30, 160, 255),
}

ALERT_COLOUR = (0, 80, 230)   # Red-ish in BGR when threshold exceeded


# ---------------------------------------------------------------------------
# ZoneManager
# ---------------------------------------------------------------------------

class ZoneManager:
    """
    Tracks per-zone occupancy across frames and provides queue alerting.
    """

    def __init__(self, zones: dict | None = None):
        self.zones: dict[str, list[float]] = zones or DEFAULT_ZONES
        self.zone_counts: dict[str, int] = {name: 0 for name in self.zones}

    # ── Public API ─────────────────────────────────────────────────────────

    def check_zones(
        self, detections: list[dict], frame_width: int, frame_height: int
    ) -> dict[str, int]:
        """
        Computes how many tracked persons fall inside each defined zone.

        Parameters
        ----------
        detections  : list of detection dicts produced by RetailTracker
        frame_width : pixel width of the source frame
        frame_height: pixel height of the source frame

        Returns
        -------
        dict mapping zone name -> occupant count for this frame
        """
        counts: dict[str, int] = {name: 0 for name in self.zones}

        for det in detections:
            cx, cy = det["center"]
            nx, ny = cx / frame_width, cy / frame_height

            for name, (x1, y1, x2, y2) in self.zones.items():
                if x1 <= nx <= x2 and y1 <= ny <= y2:
                    counts[name] += 1

        self.zone_counts = counts
        return counts

    def get_queue_status(self, threshold: int = 3) -> dict:
        """
        Evaluates the Checkout zone against the configured threshold.

        Returns a structured status dict consumed by the frontend.
        """
        checkout_count = self.zone_counts.get("Checkout", 0)

        if checkout_count >= threshold + 3:
            return {
                "alert": True,
                "message": f"Critical queue — {checkout_count} persons waiting",
                "level": "critical",
                "count": checkout_count,
            }
        elif checkout_count >= threshold:
            return {
                "alert": True,
                "message": f"Queue threshold exceeded — {checkout_count} persons",
                "level": "warning",
                "count": checkout_count,
            }
        return {
            "alert": False,
            "message": "Checkout queue within limits",
            "level": "nominal",
            "count": checkout_count,
        }

    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        """
        Renders zone rectangles and occupancy labels directly onto the frame.
        Zones with occupants are highlighted with a translucent fill.
        """
        h, w = frame.shape[:2]
        overlay = frame.copy()

        for name, (x1, y1, x2, y2) in self.zones.items():
            px1, py1 = int(x1 * w), int(y1 * h)
            px2, py2 = int(x2 * w), int(y2 * h)
            count = self.zone_counts.get(name, 0)
            colour = ZONE_COLOURS.get(name, (200, 200, 200))

            # Semi-transparent fill when occupied
            if count > 0:
                cv2.rectangle(overlay, (px1, py1), (px2, py2), colour, -1)

            # Blend fill onto frame
            frame = cv2.addWeighted(overlay, 0.12, frame, 0.88, 0)
            overlay = frame.copy()

            # Border
            border_colour = ALERT_COLOUR if name == "Checkout" and count >= 3 else colour
            cv2.rectangle(frame, (px1, py1), (px2, py2), border_colour, 2)

            # Label pill background
            label = f"{name}  {count}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
            lx, ly = px1, py1 - 6
            cv2.rectangle(frame, (lx - 2, ly - th - 4), (lx + tw + 4, ly + 2), border_colour, -1)
            cv2.putText(
                frame, label,
                (lx, ly - 2),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )

        return frame


# ---------------------------------------------------------------------------
# HeatmapManager (optional overlay)
# ---------------------------------------------------------------------------

class HeatmapManager:
    """
    Maintains a spatial grid that accumulates presence over time,
    rendered as a JET colour map overlay for session heat analysis.
    """

    def __init__(self, frame_width: int, frame_height: int, grid_size: int = 30):
        self.grid_size = grid_size
        self.rows = frame_height // grid_size
        self.cols = frame_width // grid_size
        self.grid = np.zeros((self.rows, self.cols), dtype=np.float32)

    def update(self, detections: list[dict]) -> None:
        for det in detections:
            gx = int(det["center"][0] // self.grid_size)
            gy = int(det["center"][1] // self.grid_size)
            if 0 <= gy < self.rows and 0 <= gx < self.cols:
                self.grid[gy, gx] += 1

    def render(self, frame: np.ndarray, alpha: float = 0.40) -> np.ndarray:
        """Blends the heat overlay with the source frame."""
        if np.max(self.grid) == 0:
            return frame
        norm = cv2.normalize(self.grid, None, 0, 255, cv2.NORM_MINMAX)
        up = cv2.resize(norm.astype(np.uint8), (frame.shape[1], frame.shape[0]))
        coloured = cv2.applyColorMap(up, cv2.COLORMAP_JET)
        return cv2.addWeighted(frame, 1 - alpha, coloured, alpha, 0)
