import cv2
import numpy as np
import torch
from ultralytics import YOLO
import logging
import os
from typing import List, Dict, Any

# Fix for PyTorch 2.6+ security update
try:
    from ultralytics.nn.tasks import DetectionModel
    import ultralytics.nn.modules.conv as conv_layers
    import ultralytics.nn.modules.block as block_layers
    import ultralytics.nn.modules.head as head_layers
    import torch.nn as nn
    torch.serialization.add_safe_globals([
        DetectionModel,
        conv_layers.Conv,
        conv_layers.Concat,
        block_layers.C2f,
        block_layers.Bottleneck,
        block_layers.DFL,
        block_layers.SPPF,
        nn.modules.container.Sequential,
        nn.modules.container.ModuleList,
        nn.modules.conv.Conv2d,
        nn.modules.batchnorm.BatchNorm2d,
        nn.modules.activation.SiLU,
        nn.modules.pooling.MaxPool2d,
        nn.modules.upsampling.Upsample,
        head_layers.Detect
    ])
except ImportError:
    pass

logger = logging.getLogger(__name__)

class CVService:
    def __init__(self):
        # In a real environment, weights would be pre-downloaded
        # For this demo, we assume YOLOv8n (nano) for speed
        try:
            # We use a context manager to temporarily allow unsafe globals if needed, 
            # or rely on the add_safe_globals above.
            self.model = YOLO('yolov8n.pt') 
        except Exception as e:
            logger.warning(f"Failed to load YOLO model: {e}. Falling back to mock detection.")
            self.model = None

    async def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """
        Samples frames and runs object detection/quality analysis.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps > 0 else 0

        # Sample 3 points: Start, Middle, End
        sample_indices = [0, frame_count // 2, frame_count - 1]
        detections = []
        blur_scores = []

        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # 1. Blur Detection (Laplacian Variance)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_scores.append(blur_score)

            # 2. Object Detection
            if self.model:
                results = self.model(frame, verbose=False)
                for r in results:
                    for c in r.boxes.cls:
                        detections.append(self.model.names[int(c)])

        cap.release()

        # Aggregate results
        unique_objects = list(set(detections))
        avg_blur = np.mean(blur_scores) if blur_scores else 0
        
        # Stability / Noise (Simplified for demo)
        tech_score = min(100, (avg_blur / 500) * 100) # Arbitrary normalization

        reasoning = f"Detected {len(unique_objects)} unique objects including {', '.join(unique_objects[:3])}."
        if avg_blur < 100:
            reasoning += " Warning: Low focus score detected in sampled frames."

        return {
            "duration": duration,
            "objects": unique_objects,
            "technical_score": tech_score,
            "blur_score": avg_blur,
            "reasoning": reasoning,
            "confidence": 0.92 if self.model else 0.5
        }

cv_service = CVService()
