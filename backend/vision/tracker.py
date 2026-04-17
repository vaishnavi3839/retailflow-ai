"""
RetailFlow AI — Vision Tracker
Wraps YOLOv8 with BoT-SORT to produce annotated frames and structured
detection payloads for downstream zone and analytics processing.
"""

import cv2
from ultralytics import YOLO


class RetailTracker:
    """
    Encapsulates a YOLOv8 model configured for persistent multi-object
    tracking via the BoT-SORT algorithm.
    """

    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = YOLO(model_path)
        self.last_confidence: float = 0.0

    def process_frame(self, frame) -> tuple:
        """
        Runs detection + tracking on a single BGR frame.

        Returns
        -------
        annotated_frame : np.ndarray
            The input frame with bounding boxes, track IDs, and confidence
            labels rendered by Ultralytics.
        detections : list[dict]
            Structured list of active tracks with keys:
                id       – stable track ID across frames
                coords   – [x1, y1, x2, y2] bounding box pixels
                center   – [cx, cy] centre point pixels
                conf     – detection confidence (0–1)
        """
        results = self.model.track(
            frame,
            persist=True,
            tracker="botsort.yaml",
            verbose=False,
            classes=[0],  # Person class only
        )

        annotated_frame = results[0].plot(
            line_width=2,
            font_size=0.4,
        )

        detections: list[dict] = []
        confs: list[float] = []

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.int().cpu().tolist()
            confidences = results[0].boxes.conf.cpu().tolist()

            for box, track_id, conf in zip(boxes, ids, confidences):
                cx = (box[0] + box[2]) / 2.0
                cy = (box[1] + box[3]) / 2.0
                detections.append(
                    {
                        "id": track_id,
                        "coords": box.tolist(),
                        "center": [cx, cy],
                        "conf": round(float(conf), 3),
                    }
                )
                confs.append(conf)

        self.last_confidence = float(sum(confs) / len(confs)) if confs else 0.0
        return annotated_frame, detections
