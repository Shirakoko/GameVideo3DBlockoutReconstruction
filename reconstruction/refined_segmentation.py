"""Explicit local Grounding DINO/SAM inference; no implicit remote access."""
from pathlib import Path
import json

import numpy as np

from blender.pipeline import sha256

CATEGORIES = ("environment", "ground", "rock", "tree", "barrier", "building", "grass", "shrub", "person")
PROMPTS = {"person": "person", "tree": "tree", "rock": "rock", "ground": "ground",
           "barrier": "wooden fence", "building": "building", "grass": "grass", "shrub": "bush"}


class LocalSegmenter:
    def __init__(self, model_directory):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor
        root = Path(model_directory).resolve()
        receipt = json.loads((root/"models.lock.json").read_text(encoding="utf-8"))
        if receipt.get("schema") != "refined.models/1":
            raise ValueError("Prepare and verify the selected local models first")
        for name in ("detector", "sam"):
            entry = receipt["models"][name]
            if entry["license"] != "apache-2.0":
                raise ValueError("Unreviewed model license")
            for file, digest in entry["files"].items():
                path = (root/name/file).resolve()
                if path.parent != root/name or not path.is_file() or sha256(path) != digest:
                    raise ValueError("Local model bytes do not match their receipt")
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(0)
        self.detector_processor = AutoProcessor.from_pretrained(root/"detector", local_files_only=True)
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(root/"detector", local_files_only=True).to(self.device).eval()
        self.sam_processor = SamProcessor.from_pretrained(root/"sam", local_files_only=True)
        self.sam = SamModel.from_pretrained(root/"sam", local_files_only=True).to(self.device).eval()
        self.receipt = receipt

    def segment(self, rgb):
        import cv2
        from PIL import Image
        image = Image.fromarray(rgb)
        h, w = rgb.shape[:2]
        detections = []
        with self.torch.inference_mode():
            for category, prompt in PROMPTS.items():
                inputs = self.detector_processor(images=image, text=prompt+".", return_tensors="pt").to(self.device)
                output = self.detector(**inputs)
                results = self.detector_processor.post_process_grounded_object_detection(output, inputs.input_ids,
                    threshold=.30, text_threshold=.25, target_sizes=[(h, w)])[0]
                boxes = results["boxes"].detach().cpu().numpy()
                scores = results["scores"].detach().cpu().numpy()
                order = np.argsort(-scores)
                kept = []
                for index in order:
                    box = np.clip(boxes[index], [0, 0, 0, 0], [w, h, w, h])
                    if min(box[2]-box[0], box[3]-box[1]) < 3:
                        continue
                    if any(_box_iou(box, previous) > .6 for previous in kept):
                        continue
                    kept.append(box)
                    detections.append((category, box.tolist(), float(scores[index])))
                    if len(kept) == 6:
                        break
            labels = np.zeros((h, w), dtype=np.uint8)
            confidence = np.zeros((h, w), dtype=np.float32)
            person = np.zeros((h, w), dtype=bool)
            if detections:
                boxes = [d[1] for d in detections]
                sam_inputs = self.sam_processor(image, input_boxes=[boxes], return_tensors="pt").to(self.device)
                prediction = self.sam(**sam_inputs, multimask_output=True)
                masks = self.sam_processor.image_processor.post_process_masks(prediction.pred_masks.cpu(),
                    sam_inputs["original_sizes"].cpu(), sam_inputs["reshaped_input_sizes"].cpu())[0]
                quality = prediction.iou_scores[0].cpu().numpy()
                for index, (category, box, score) in enumerate(detections):
                    best = int(np.argmax(quality[index]))
                    mask = masks[index, best].numpy().astype(bool)
                    score *= float(quality[index, best])
                    if category == "person":
                        person |= mask
                    else:
                        selected = mask & (score > confidence)
                        labels[selected] = CATEGORIES.index(category)
                        confidence[selected] = score
            # Expand edge uncertainty to prevent clothes/hair becoming scenery.
            person = cv2.dilate(person.astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool)
            labels[person] = CATEGORIES.index("person")
        return labels, dict(provider="local_grounding_dino_sam", device=self.device,
                            detected_boxes=len(detections), person_pixels=int(person.sum()),
                            category_pixels={name: int((labels == i).sum()) for i, name in enumerate(CATEGORIES)},
                            provenance="predicted", real_semantic_accuracy_validated=False)


def _box_iou(a, b):
    low, high = np.maximum(a[:2], b[:2]), np.minimum(a[2:], b[2:])
    intersection = np.prod(np.maximum(high-low, 0))
    area = np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-intersection
    return intersection/area if area > 0 else 0
