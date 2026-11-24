"""RGB providers: assisted annotation, optional LOCAL DINO + SAM, fixture rule.

Model weights are never downloaded implicitly. Bbox-only detections remain
semantic observations and cannot supply world geometry.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
from PIL import Image, ImageDraw

CATEGORIES = ("person", "tree", "rock", "ground", "barrier")
PRIMITIVES = dict(person="capsule", tree="cylinder", rock="box", ground="plane", barrier="box")


def confidence(score, provenance, method, evidence):
    return dict(score=float(score), provenance=provenance, method=method, evidence=evidence)


def unknown(reason):
    return confidence(0, "unobserved", "not_available", reason)


@dataclass
class Detection:
    key: str
    category: str
    bbox: list
    mask: np.ndarray
    mask_kind: str
    confidence: dict
    mask_confidence: dict
    identity_hint: str | None = None
    static_hint: bool | None = None

    def validate(self, rgb):
        h, w = rgb.shape[:2]
        if self.category not in CATEGORIES:
            raise ValueError(f"Unsupported MVP category: {self.category}")
        if self.mask.shape != (h, w) or self.mask.dtype != np.bool_ or not self.mask.any():
            raise ValueError("Nonempty boolean object mask at native RGB raster required")
        if len(self.bbox) != 4 or not np.isfinite(self.bbox).all():
            raise ValueError("Finite bbox_xyxy required")
        x0, y0, x1, y1 = self.bbox
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise ValueError("BBox outside native raster")
        vv, uu = np.where(self.mask)
        if not (uu.min() >= x0 and uu.max() < x1 and vv.min() >= y0 and vv.max() < y1):
            raise ValueError("Mask pixels outside bbox (xyxy uses exclusive upper bounds)")
        for c in (self.confidence, self.mask_confidence):
            if not np.isfinite(c["score"]) or not 0 <= c["score"] <= 1 or c["provenance"] not in ("inferred", "predicted", "ground_truth") or not c["evidence"]:
                raise ValueError("Detection confidence/provenance/evidence required")
        if self.static_hint is not None and not isinstance(self.static_hint, bool):
            raise ValueError("static_hint must be boolean or null")
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("Stable detection key required")


def mask_bbox(mask):
    y, x = np.where(mask)
    return [int(x.min()), int(y.min()), int(x.max()+1), int(y.max()+1)]


class AnnotationProvider:
    name = "assisted_rgb_annotations"

    def __init__(self, path):
        self.path = Path(path).resolve()
        self.document = json.loads(self.path.read_text(encoding="utf-8"))
        if self.document.get("schema") != "stage_c.rgb_annotations/1" or not self.document.get("evidence"):
            raise ValueError("stage_c.rgb_annotations/1 with explicit review evidence required")
        self.rows = {}
        for row in self.document["frames"]:
            if row["frame_index"] in self.rows:
                raise ValueError("Duplicate annotation frame")
            self.rows[row["frame_index"]] = row

    def detect(self, rgb, frame_index):
        row = self.rows.get(frame_index)
        if row is None:
            return []
        h, w = rgb.shape[:2]
        if row["raster_size"] != [w, h]:
            raise ValueError("Annotation raster differs from RGB; implicit resize prohibited")
        result = []
        for item in row["objects"]:
            m = item["mask"]
            if m["kind"] == "file":
                with Image.open(self.path.parent/m["path"]) as image:
                    a = np.asarray(image.convert("L"))
                if not np.isin(a, [0, 255]).all():
                    raise ValueError("File masks must contain binary 0/255 pixels")
                mask, kind = a == 255, "instance"
            elif m["kind"] == "polygon":
                vertices = np.asarray(m["vertices"], dtype=float)
                if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3 or not np.isfinite(vertices).all() or (vertices < 0).any() or (vertices[:, 0] >= w).any() or (vertices[:, 1] >= h).any():
                    raise ValueError("Polygon needs >=3 finite native-raster vertices")
                canvas = Image.new("L", (w, h))
                ImageDraw.Draw(canvas).polygon([tuple(p) for p in vertices], fill=255)
                mask, kind = np.asarray(canvas) == 255, "polygon"
            elif m["kind"] == "bbox_proxy":
                x0, y0, x1, y1 = item["bbox_xyxy"]
                mask = np.zeros((h, w), dtype=bool)
                mask[int(y0):int(y1), int(x0):int(x1)] = True
                kind = "bbox_proxy"
            else:
                raise ValueError("Mask kind must be file, polygon or bbox_proxy")
            c = confidence(item["score"], "inferred", self.name, self.document["evidence"])
            d = Detection(item["key"], item["category"], item.get("bbox_xyxy", mask_bbox(mask)), mask, kind, c,
                confidence(m.get("score", item["score"]), "inferred", f"assisted_{kind}", m.get("evidence", self.document["evidence"])),
                item.get("identity_hint"), item.get("static_hint"))
            d.validate(rgb)
            result.append(d)
        return result


class LocalVisionProvider:
    """Optional transformers Grounding DINO + SAM; explicit local model dirs.

    pip install torch transformers is user-selected; not a core dependency.
    Without SAM, bbox masks are proxies, not object silhouettes.
    """
    name = "local_grounding_dino_sam"

    def __init__(self, detector_path, sam_path=None, threshold=0.35):
        if not 0 < threshold < 1:
            raise ValueError("Detection threshold must be between 0 and 1")
        paths = [Path(detector_path)] + ([Path(sam_path)] if sam_path else [])
        if any(not p.is_dir() for p in paths):
            raise ValueError("Explicit existing LOCAL model directories required; no download")
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor
        except ImportError as error:
            raise ValueError("Optional local vision requires torch and transformers plus local weights") from error
        self.torch, self.threshold = torch, threshold
        self.processor = AutoProcessor.from_pretrained(detector_path, local_files_only=True)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(detector_path, local_files_only=True).eval()
        self.sam = self.sam_processor = None
        if sam_path:
            self.sam = SamModel.from_pretrained(sam_path, local_files_only=True).eval()
            self.sam_processor = SamProcessor.from_pretrained(sam_path, local_files_only=True)

    def detect(self, rgb, frame_index):
        image = Image.fromarray(rgb)
        prompts = {"person": "person", "tree": "tree", "rock": "rock", "ground": "ground", "barrier": "wooden fence"}
        # One prompt/category avoids fragile free-text-to-category remapping.
        result = []
        for category, prompt in prompts.items():
            inputs = self.processor(images=image, text=prompt+".", return_tensors="pt")
            with self.torch.no_grad():
                output = self.model(**inputs)
            detections = self.processor.post_process_grounded_object_detection(output, inputs.input_ids,
                threshold=self.threshold, text_threshold=0.25, target_sizes=[image.size[::-1]])[0]
            for box, score in zip(detections["boxes"].tolist(), detections["scores"].tolist()):
                w, h = image.size
                x0, y0, x1, y1 = np.clip(box, [0, 0, 0, 0], [w, h, w, h])
                if x1-x0 < 2 or y1-y0 < 2:
                    continue
                bbox = [int(np.floor(x0)), int(np.floor(y0)), int(np.ceil(x1)), int(np.ceil(y1))]
                mask = np.zeros((h, w), dtype=bool)
                mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = True
                kind, mask_score = "bbox_proxy", score
                if self.sam:
                    sam_input = self.sam_processor(image, input_boxes=[[bbox]], return_tensors="pt")
                    with self.torch.no_grad():
                        segmentation = self.sam(**sam_input)
                    masks = self.sam_processor.image_processor.post_process_masks(segmentation.pred_masks,
                        sam_input["original_sizes"], sam_input["reshaped_input_sizes"])[0][0]
                    best = int(segmentation.iou_scores[0, 0].argmax())
                    mask &= masks[best].cpu().numpy().astype(bool)
                    mask_score = float(segmentation.iou_scores[0, 0, best])
                    kind = "instance"
                if not mask.any():
                    continue
                d = Detection(f"{category}-{len(result)}", category, bbox, mask, kind,
                    confidence(score, "predicted", "local_grounding_dino", "Local RGB inference; not measured semantic accuracy"),
                    confidence(mask_score, "predicted", "local_sam" if self.sam else "bbox_proxy", "Local segmentation score; not calibrated accuracy"))
                d.validate(rgb)
                # Same-category overlapping outputs are NMS'd below.
                if not any(a.category == category and mask_iou(a.mask, d.mask) > 0.6 for a in result):
                    result.append(d)
        return result


def mask_iou(a, b):
    union = np.count_nonzero(a | b)
    return np.count_nonzero(a & b)/union if union else 0.


class SyntheticColorProvider:
    """Fixture ONLY, exact synthetic palette; never run on real RGB."""
    name = "synthetic_palette_rule_NOT_real_vision"
    palette = dict(person=(220, 50, 60), tree=(40, 170, 70), rock=(130, 140, 155), ground=(160, 120, 65), barrier=(230, 170, 40))

    def detect(self, rgb, frame_index):
        from scipy.ndimage import label
        result = []
        for category, color in self.palette.items():
            components, count = label(np.all(rgb == color, axis=2))
            for index in range(1, count+1):
                mask = components == index
                if mask.sum() < 8:
                    continue
                c = confidence(1, "inferred", self.name, "Exact fixture palette rule, not a real-game detector or ground truth label")
                result.append(Detection(f"{category}-{index}", category, mask_bbox(mask), mask, "instance", c, c.copy()))
        return result
