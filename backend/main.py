"""
RetailFlow AI — FastAPI Backend
Provides MJPEG video streaming, REST metrics, WebSocket real-time feed,
historical data, alert log, and system health endpoints.
"""

import cv2
import time
import json
import asyncio
from collections import deque
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from vision.tracker import RetailTracker
from vision.analytics import ZoneManager
from database import save_log, get_recent_logs

# ---------------------------------------------------------------------------
# Application Bootstrap
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RetailFlow AI",
    description="Real-time computer vision retail analytics platform",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Core Engines
# ---------------------------------------------------------------------------

tracker = RetailTracker()
zone_manager = ZoneManager()

# ---------------------------------------------------------------------------
# Shared State
# ---------------------------------------------------------------------------

current_metrics: dict = {
    "total_footfall": 0,
    "zone_occupancy": {},
    "queue_status": {"alert": False, "message": "System initialising", "level": "nominal"},
    "fps": 0.0,
    "latency_ms": 0.0,
    "session_peak": 0,
    "detection_confidence": 0.0,
    "timestamp": "",
}

alert_log: deque = deque(maxlen=50)          # Ring buffer of last 50 alert events
footfall_history: deque = deque(maxlen=120)  # Last 120 data points (~2 min at 1 fps)
connected_clients: list[WebSocket] = []


# ---------------------------------------------------------------------------
# Video Streaming Generator
# ---------------------------------------------------------------------------

def gen_frames(queue_threshold: int = 3):
    """
    Reads from the default camera (or a video file), runs YOLO tracking,
    updates global state, and yields MJPEG frames.
    """
    global current_metrics

    cap = cv2.VideoCapture(r"C:\Users\Vaishnavi\Videos\Screen Recordings\Screen Recording 2026-04-17 135549.mp4")
    if not cap.isOpened():
        # Graceful degradation: yield a blank frame with an overlay
        blank = _build_offline_frame()
        while True:
            ret, buf = cv2.imencode(".jpg", blank)
            if ret:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            time.sleep(0.1)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    last_db_save = time.time()
    frame_times: deque = deque(maxlen=30)

    while True:
        t0 = time.time()

        success, frame = cap.read()
        if not success:
            break

        # ── Inference ───────────────────────────────────────────────────────
        annotated_frame, detections = tracker.process_frame(frame)

        # ── Zone Analysis ───────────────────────────────────────────────────
        zone_counts = zone_manager.check_zones(detections, frame_w, frame_h)
        queue_info = zone_manager.get_queue_status(threshold=queue_threshold)
        annotated_frame = zone_manager.draw_zones(annotated_frame)

        # ── Performance Metrics ─────────────────────────────────────────────
        latency_ms = (time.time() - t0) * 1000
        frame_times.append(latency_ms)
        avg_fps = 1000.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0.0
        confidence = tracker.last_confidence

        total = len(detections)

        # ── Alert Log ───────────────────────────────────────────────────────
        if queue_info["alert"]:
            entry = {
                "ts": datetime.utcnow().isoformat(timespec="seconds"),
                "level": queue_info["level"],
                "message": queue_info["message"],
                "zone": "Checkout",
                "count": zone_counts.get("Checkout", 0),
            }
            if not alert_log or alert_log[-1]["message"] != entry["message"]:
                alert_log.append(entry)

        # ── Update Global State ─────────────────────────────────────────────
        current_metrics.update(
            {
                "total_footfall": total,
                "zone_occupancy": zone_counts,
                "queue_status": queue_info,
                "fps": round(avg_fps, 1),
                "latency_ms": round(latency_ms, 1),
                "session_peak": max(current_metrics["session_peak"], total),
                "detection_confidence": round(confidence, 3),
                "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
            }
        )

        footfall_history.append(
            {"ts": current_metrics["timestamp"], "count": total}
        )

        # ── Periodic DB Persistence (every 30 s) ────────────────────────────
        if time.time() - last_db_save > 30:
            busiest = max(zone_counts, key=zone_counts.get) if zone_counts else "none"
            try:
                save_log(total, busiest)
            except Exception as exc:
                print(f"[DB ERROR] {exc}")
            last_db_save = time.time()

        # ── Encode MJPEG ────────────────────────────────────────────────────
        ret, buf = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ret:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            )

    cap.release()


def _build_offline_frame():
    """Returns a 640x480 dark frame with an 'offline' message."""
    frame = __import__("numpy").zeros((480, 640, 3), dtype="uint8")
    cv2.putText(
        frame, "VIDEO SOURCE OFFLINE", (100, 240),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 180, 130), 2,
    )
    return frame


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["meta"])
def health_check():
    return {
        "status": "online",
        "service": "RetailFlow AI",
        "version": "2.0.0",
        "uptime_since": datetime.utcnow().isoformat(timespec="seconds"),
    }


@app.get("/video_feed", tags=["stream"])
async def video_feed():
    """MJPEG stream endpoint. Embed as <img src='/video_feed'>."""
    return StreamingResponse(
        gen_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/metrics", tags=["analytics"])
async def get_metrics():
    """Current live snapshot of all analytics metrics."""
    return current_metrics


@app.get("/history", tags=["analytics"])
async def get_history(limit: int = 60):
    """
    Returns the in-memory footfall time series (up to 120 points)
    plus persisted records from the database.
    """
    mem = list(footfall_history)[-limit:]
    try:
        db_rows = get_recent_logs(limit=limit)
        persisted = [
            {"ts": r.timestamp.isoformat(timespec="seconds"), "count": r.total_count}
            for r in db_rows
        ]
    except Exception:
        persisted = []
    return {"memory": mem, "persisted": persisted}


@app.get("/alerts", tags=["analytics"])
async def get_alerts(limit: int = 20):
    """Returns the most recent queue / occupancy alert events."""
    return {"alerts": list(alert_log)[-limit:]}


@app.get("/system", tags=["meta"])
async def system_status():
    """Hardware / inference performance snapshot."""
    return {
        "fps": current_metrics.get("fps", 0),
        "latency_ms": current_metrics.get("latency_ms", 0),
        "detection_confidence": current_metrics.get("detection_confidence", 0),
        "session_peak": current_metrics.get("session_peak", 0),
        "active_ws_clients": len(connected_clients),
        "alert_count": len(alert_log),
    }


# ---------------------------------------------------------------------------
# WebSocket — Real-Time Push
# ---------------------------------------------------------------------------

@app.websocket("/ws/metrics")
async def ws_metrics(websocket: WebSocket):
    """
    Pushes the full metrics payload to the client every second.
    The dashboard connects here for zero-polling real-time updates.
    """
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        while True:
            payload = json.dumps(
                {
                    **current_metrics,
                    "history_tail": list(footfall_history)[-30:],
                    "recent_alerts": list(alert_log)[-5:],
                }
            )
            await websocket.send_text(payload)
            await asyncio.sleep(1)
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        if websocket in connected_clients:
            connected_clients.remove(websocket)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
