# surgery_guard_batch_notrain_no_aneurysm.py
# OWLv2 (Google) zero-shot boxes + (optional) SAM2 masks + tiling + mask-based filtering + tight boxes
# Batch mode: process multiple images in one run (list or folder).
# Prototype only. Not for clinical use.

import argparse
import json
import os
import glob
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import nms

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# SAM2 (optional)
try:
    from transformers import Sam2Processor, Sam2Model
    SAM2_AVAILABLE = True
except Exception:
    SAM2_AVAILABLE = False


DEFAULT_DETECTOR_MODEL = "google/owlv2-base-patch16-ensemble"
DEFAULT_SAM2_MODEL = "facebook/sam2.1-hiera-tiny"

DEFAULT_IMAGE_PATH = r"E:\K Temp Storage\Kitahack\Data\test2.jpg"
DEFAULT_INPUT_DIR = r"E:\K Temp Storage\Kitahack\Data"   # << NEW: default folder batch input
DEFAULT_OUT_DIR = r"E:\K Temp Storage\Kitahack\outputs2"

# Removed aneurysm entirely
PROMPTS = {
    "bleeding": [
        "fresh bright red blood on tissue",
        "blood pooling on tissue",
        "blood oozing from a point",
        "active bleeding point",
        "hemorrhage on tissue",
    ],
    "left_equipment": [
        "surgical instrument",
        "forceps",
        "clamp",
        "retractor",
        "surgical scissors",
        "scalpel",
        "needle holder",
        "suction tube",
        "gauze",
        "surgical sponge",
        "hemostat",
    ],
}

GROUP_NAME_EN = {
    "bleeding": "bleeding",
    "left_equipment": "equipment",
    "unknown": "unknown",
}
GROUP_NAME_ZH = {
    "bleeding": "出血点",
    "left_equipment": "遗留器械",
    "unknown": "未知",
}

COLOR_MAP = {
    "bleeding": (0, 0, 255),         # red (BGR)
    "left_equipment": (255, 0, 0),   # blue
    "unknown": (200, 200, 200),
}

DEFAULT_DET_GLOBAL_THRESHOLD = 0.12

GROUP_SCORE_THRESH = {
    "left_equipment": 0.22,
    "bleeding": 0.20,
    "unknown": 0.25,
}


@dataclass
class Detection:
    query: str
    group: str
    score: float
    box_xyxy: List[float]  # [x1,y1,x2,y2]


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def enable_fast_cuda():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def best_amp_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_image_pil(image_path: str) -> Image.Image:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")
    return Image.open(image_path).convert("RGB")


def resize_keep_aspect(image: Image.Image, max_side: int) -> Image.Image:
    w, h = image.size
    if max(w, h) <= max_side:
        return image
    scale = max_side / float(max(w, h))
    nw, nh = int(round(w * scale)), int(round(h * scale))
    return image.resize((nw, nh), Image.BILINEAR)


def grayworld_white_balance(bgr: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(bgr.astype(np.float32))
    mb, mg, mr = b.mean(), g.mean(), r.mean()
    m = (mb + mg + mr) / 3.0
    b = np.clip(b * (m / (mb + 1e-6)), 0, 255)
    g = np.clip(g * (m / (mg + 1e-6)), 0, 255)
    r = np.clip(r * (m / (mr + 1e-6)), 0, 255)
    return cv2.merge([b, g, r]).astype(np.uint8)


def mild_contrast_enhance(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def build_queries(use_templates: bool = True) -> Tuple[List[str], Dict[str, str]]:
    templates = ["{q}"]
    if use_templates:
        templates = ["{q}", "a photo of {q}", "in a surgical field, {q}"]

    all_q: List[str] = []
    q2g: Dict[str, str] = {}
    for g, qs in PROMPTS.items():
        for q in qs:
            q = q.strip()
            for t in templates:
                qq = t.format(q=q)
                if qq not in q2g:
                    all_q.append(qq)
                    q2g[qq] = g
    return all_q, q2g


def gen_tiles(image: Image.Image, tile: int, overlap: float) -> List[Tuple[Image.Image, int, int]]:
    w, h = image.size
    if tile <= 0 or max(w, h) <= tile:
        return [(image, 0, 0)]

    stride = max(1, int(round(tile * (1.0 - overlap))))
    xs = list(range(0, max(1, w - tile + 1), stride))
    ys = list(range(0, max(1, h - tile + 1), stride))
    if xs[-1] != w - tile:
        xs.append(w - tile)
    if ys[-1] != h - tile:
        ys.append(h - tile)

    out = []
    for y0 in ys:
        for x0 in xs:
            crop = image.crop((x0, y0, x0 + tile, y0 + tile))
            out.append((crop, x0, y0))
    return out


def postprocess_grounded(processor, outputs, target_sizes_hw: torch.Tensor, threshold: float, text_labels):
    if hasattr(processor, "post_process_grounded_object_detection"):
        return processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes_hw,
            threshold=threshold,
            text_labels=text_labels,
        )
    raise AttributeError("Processor lacks post_process_grounded_object_detection")


def run_detector(
    image: Image.Image,
    processor,
    model,
    amp_dtype: torch.dtype,
    threshold: float,
    per_call_max: int,
    use_templates: bool,
) -> List[Detection]:
    queries, q2g = build_queries(use_templates=use_templates)
    text_labels = [queries]

    inputs = processor(text=text_labels, images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.inference_mode():
        if model.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                outputs = model(**inputs)
        else:
            outputs = model(**inputs)

    target_sizes = torch.tensor([(image.height, image.width)], dtype=torch.int64)
    results = postprocess_grounded(processor, outputs, target_sizes, threshold, text_labels)[0]

    boxes = results.get("boxes", None)
    scores = results.get("scores", None)
    labels = results.get("text_labels", results.get("labels", None))

    if boxes is None or scores is None or boxes.numel() == 0:
        return []

    dets: List[Detection] = []
    for i in range(int(scores.shape[0])):
        score = float(scores[i].item())
        box = [float(x) for x in boxes[i].tolist()]

        lbl = "unknown"
        if labels is not None:
            try:
                lbl = str(labels[i])
            except Exception:
                lbl = "unknown"

        dets.append(Detection(query=lbl, group=q2g.get(lbl, "unknown"), score=score, box_xyxy=box))

    dets.sort(key=lambda d: d.score, reverse=True)
    return dets[:per_call_max]


def run_detector_tiled(
    image: Image.Image,
    processor,
    model,
    amp_dtype: torch.dtype,
    threshold: float,
    tile: int,
    overlap: float,
    per_tile_max: int,
    use_templates: bool,
) -> List[Detection]:
    tiles = gen_tiles(image, tile, overlap)
    all_dets: List[Detection] = []

    for (crop, x0, y0) in tiles:
        dets = run_detector(crop, processor, model, amp_dtype, threshold, per_tile_max, use_templates)
        for d in dets:
            x1, y1, x2, y2 = d.box_xyxy
            d.box_xyxy = [x1 + x0, y1 + y0, x2 + x0, y2 + y0]
        all_dets.extend(dets)

    return all_dets


def groupwise_nms_and_filter(dets: List[Detection], nms_iou: float, max_dets: int) -> List[Detection]:
    filtered = []
    for d in dets:
        th = GROUP_SCORE_THRESH.get(d.group, GROUP_SCORE_THRESH["unknown"])
        if d.score >= th:
            filtered.append(d)

    if not filtered:
        return []

    out: List[Detection] = []
    for group in ["left_equipment", "bleeding", "unknown"]:
        gd = [d for d in filtered if d.group == group]
        if not gd:
            continue
        boxes = torch.tensor([d.box_xyxy for d in gd], dtype=torch.float32)
        scores = torch.tensor([d.score for d in gd], dtype=torch.float32)
        keep = nms(boxes, scores, nms_iou).tolist()
        out.extend([gd[i] for i in keep])

    out.sort(key=lambda d: d.score, reverse=True)
    return out[:max_dets]


def _red_mask_hsv(hsv: np.ndarray) -> np.ndarray:
    lower1 = np.array([0, 80, 60], dtype=np.uint8)
    upper1 = np.array([10, 255, 255], dtype=np.uint8)
    lower2 = np.array([170, 80, 60], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)
    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    return cv2.bitwise_or(m1, m2)


def bleeding_heuristic_candidates(
    bgr: np.ndarray,
    min_area: int = 160,
    max_area_frac: float = 0.08,
) -> List[Detection]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = _red_mask_hsv(hsv)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    H, W = mask.shape[:2]
    img_area = float(H * W)

    dets: List[Detection] = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        if area / img_area > max_area_frac:
            continue

        x1, y1, x2, y2 = x, y, x + w, y + h
        roi = mask[y1:y2, x1:x2]
        red_ratio = float((roi > 0).mean())

        score = 0.18 + 0.50 * red_ratio
        score = float(np.clip(score, 0.18, 0.55))

        dets.append(Detection(
            query="heuristic_red_region",
            group="bleeding",
            score=score,
            box_xyxy=[float(x1), float(y1), float(x2), float(y2)],
        ))

    return dets


def run_sam2_masks(
    image: Image.Image,
    boxes_xyxy: List[List[float]],
    sam2_processor,
    sam2_model,
    amp_dtype: torch.dtype,
) -> np.ndarray:
    if len(boxes_xyxy) == 0:
        return np.zeros((0, image.size[1], image.size[0]), dtype=bool)

    sam_inputs = sam2_processor(images=image, input_boxes=[boxes_xyxy], return_tensors="pt")
    sam_inputs = {k: v.to(sam2_model.device) if hasattr(v, "to") else v for k, v in sam_inputs.items()}

    with torch.inference_mode():
        if sam2_model.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                sam_outputs = sam2_model(**sam_inputs, multimask_output=False)
        else:
            sam_outputs = sam2_model(**sam_inputs, multimask_output=False)

    try:
        masks = sam2_processor.post_process_masks(
            sam_outputs.pred_masks.cpu(),
            sam_inputs["original_sizes"],
        )[0]
    except TypeError:
        masks = sam2_processor.post_process_masks(
            sam_outputs.pred_masks.cpu(),
            sam_inputs["original_sizes"],
            sam_inputs.get("reshaped_input_sizes", None),
        )[0]

    if masks.ndim == 4:
        masks = masks[:, 0, :, :]
    return (masks.numpy() > 0.0)


def bbox_from_mask(mask: np.ndarray) -> Optional[List[float]]:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return [float(x1), float(y1), float(x2), float(y2)]


def mask_red_ratio(mask_bool: np.ndarray, hsv: np.ndarray) -> float:
    red = _red_mask_hsv(hsv) > 0
    m = mask_bool.astype(bool)
    denom = float(m.sum())
    if denom <= 1.0:
        return 0.0
    return float((red & m).sum() / denom)


def mask_mean_sat(mask_bool: np.ndarray, hsv: np.ndarray) -> float:
    m = mask_bool.astype(bool)
    if m.sum() <= 1:
        return 0.0
    sat = hsv[:, :, 1]
    return float(sat[m].mean()) / 255.0


def filter_and_refine_with_masks(
    dets: List[Detection],
    masks_bool: np.ndarray,
    image_bgr: np.ndarray,
    bleed_min_red_ratio: float,
    bleed_min_sat: float,
    bleed_max_area_frac: float,
    tighten_boxes: bool,
) -> Tuple[List[Detection], np.ndarray]:
    if masks_bool is None or masks_bool.shape[0] != len(dets):
        return dets, masks_bool

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    H, W = hsv.shape[:2]
    img_area = float(H * W)

    kept_dets: List[Detection] = []
    kept_masks: List[np.ndarray] = []

    for d, m in zip(dets, masks_bool):
        area_frac = float(m.sum()) / img_area if img_area > 0 else 0.0

        if d.group == "bleeding":
            rr = mask_red_ratio(m, hsv)
            ms = mask_mean_sat(m, hsv)

            if rr < bleed_min_red_ratio:
                continue
            if ms < bleed_min_sat:
                continue
            if area_frac > bleed_max_area_frac:
                continue

            d.score = float(min(0.99, d.score + 0.12 * rr + 0.08 * ms))

        if tighten_boxes:
            bb = bbox_from_mask(m)
            if bb is not None:
                d.box_xyxy = bb

        kept_dets.append(d)
        kept_masks.append(m)

    if not kept_dets:
        return [], np.zeros((0, image_bgr.shape[0], image_bgr.shape[1]), dtype=bool)

    return kept_dets, np.stack(kept_masks, axis=0)


def _pil_text_bbox(draw: ImageDraw.ImageDraw, text: str, font) -> Tuple[int, int]:
    try:
        bb = draw.textbbox((0, 0), text, font=font)
        return (bb[2] - bb[0], bb[3] - bb[1])
    except Exception:
        tw, th = draw.textsize(text, font=font)
        return tw, th


def _safe_int_box_xyxy(box_xyxy: List[float], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box_xyxy
    x1 = int(round(x1)); y1 = int(round(y1)); x2 = int(round(x2)); y2 = int(round(y2))

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    if x2 == x1:
        x2 = min(w - 1, x1 + 1)
    if y2 == y1:
        y2 = min(h - 1, y1 + 1)

    return x1, y1, x2, y2


def draw_label_pil(
    bgr: np.ndarray,
    x1: int,
    y1: int,
    text: str,
    color_bgr: Tuple[int, int, int],
    font: Optional[ImageFont.FreeTypeFont],
):
    H, W = bgr.shape[:2]
    x1 = max(0, min(W - 1, int(x1)))
    y1 = max(0, min(H - 1, int(y1)))

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)
    draw = ImageDraw.Draw(im)

    if font is None:
        font = ImageFont.load_default()

    pad = 4
    tw, th = _pil_text_bbox(draw, text, font=font)

    y_top = y1 - th - pad * 2
    if y_top < 0:
        y_top = 0

    x_right = x1 + tw + pad * 2
    if x_right > W - 1:
        x_right = W - 1

    x0 = min(x1, x_right)
    x1b = max(x1, x_right)
    y0 = min(y_top, y1)
    y1b = max(y_top, y1)

    color_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
    draw.rectangle([x0, y0, x1b, y1b], fill=color_rgb)
    draw.text((x0 + pad, y0 + pad), text, font=font, fill=(255, 255, 255))

    out = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    bgr[:] = out


def overlay_pretty(
    image: Image.Image,
    dets: List[Detection],
    masks_bool: Optional[np.ndarray],
    mask_alpha: float,
    use_zh_label: bool,
    font_path: Optional[str],
    font_size: int,
) -> np.ndarray:
    rgb = np.array(image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    font = None
    if font_path and os.path.exists(font_path):
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            font = None

    if masks_bool is not None and masks_bool.shape[0] == len(dets):
        for i, det in enumerate(dets):
            mask = masks_bool[i]
            if mask.sum() == 0:
                continue
            color = COLOR_MAP.get(det.group, COLOR_MAP["unknown"])
            overlay = bgr.copy()
            overlay[mask] = color
            bgr = cv2.addWeighted(overlay, mask_alpha, bgr, 1 - mask_alpha, 0)

    H, W = bgr.shape[:2]
    name_map = GROUP_NAME_ZH if use_zh_label else GROUP_NAME_EN

    for det in dets:
        x1, y1, x2, y2 = _safe_int_box_xyxy(det.box_xyxy, W, H)
        color = COLOR_MAP.get(det.group, COLOR_MAP["unknown"])
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)

        tag = f"{name_map.get(det.group, det.group)} | {det.score:.2f}"
        draw_label_pil(bgr, x1, y1, tag, color, font)

    return bgr


def summarize(dets: List[Detection], min_score: float) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for g in ["bleeding", "left_equipment"]:
        hits = [d for d in dets if d.group == g and d.score >= min_score]
        out[g] = {
            "name": GROUP_NAME_EN.get(g, g),
            "present": len(hits) > 0,
            "count": len(hits),
            "top_score": max([h.score for h in hits], default=0.0),
        }
    return out


def list_images_from_args(images: List[str], input_dir: str, recursive: bool, patterns: str) -> List[str]:
    found: List[str] = []

    if images:
        for p in images:
            if os.path.isfile(p):
                found.append(p)

    if input_dir:
        pats = []
        for part in patterns.replace(";", ",").split(","):
            part = part.strip()
            if part:
                pats.append(part)
        if not pats:
            pats = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]

        for pat in pats:
            if recursive:
                g = os.path.join(input_dir, "**", pat)
                found.extend(glob.glob(g, recursive=True))
            else:
                g = os.path.join(input_dir, pat)
                found.extend(glob.glob(g))

    # de-dup while preserving order
    seen = set()
    out = []
    for p in found:
        ap = os.path.abspath(p)
        if ap not in seen:
            out.append(ap)
            seen.add(ap)
    return out


def load_ovd_processor(detector_model: str, use_fast_processor: bool):
    try:
        return AutoProcessor.from_pretrained(detector_model, use_fast=use_fast_processor)
    except TypeError:
        return AutoProcessor.from_pretrained(detector_model)


def load_ovd_model(detector_model: str, device: torch.device, amp_dtype: torch.dtype):
    try:
        model = AutoModelForZeroShotObjectDetection.from_pretrained(detector_model, dtype=amp_dtype)
    except TypeError:
        model = AutoModelForZeroShotObjectDetection.from_pretrained(detector_model, torch_dtype=amp_dtype)
    return model.to(device).eval()


def load_sam2_model(sam2_model_name: str, device: torch.device, amp_dtype: torch.dtype):
    sam_proc = Sam2Processor.from_pretrained(sam2_model_name)
    try:
        sam_mdl = Sam2Model.from_pretrained(sam2_model_name, dtype=amp_dtype)
    except TypeError:
        sam_mdl = Sam2Model.from_pretrained(sam2_model_name, torch_dtype=amp_dtype)
    return sam_proc, sam_mdl.to(device).eval()


def process_one_image(
    image_path: str,
    out_dir: str,
    processor,
    model,
    amp_dtype: torch.dtype,
    args,
    sam_proc=None,
    sam_model=None,
) -> Dict:
    image = load_image_pil(image_path)
    image = resize_keep_aspect(image, args.max_side)

    bgr0 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    if args.preprocess:
        bgr0 = grayworld_white_balance(bgr0)
        bgr0 = mild_contrast_enhance(bgr0)
        image = Image.fromarray(cv2.cvtColor(bgr0, cv2.COLOR_BGR2RGB))

    dets = run_detector_tiled(
        image=image,
        processor=processor,
        model=model,
        amp_dtype=amp_dtype,
        threshold=args.global_thresh,
        tile=args.tile,
        overlap=args.overlap,
        per_tile_max=args.per_tile_max,
        use_templates=args.use_templates,
    )

    bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    if args.use_bleeding_heuristic:
        dets.extend(bleeding_heuristic_candidates(
            bgr=bgr,
            min_area=args.bleeding_min_area,
            max_area_frac=args.bleed_max_area_frac,
        ))

    dets = groupwise_nms_and_filter(dets, nms_iou=args.nms_iou, max_dets=args.max_dets)

    masks_bool = None
    if args.use_masks and sam_proc is not None and sam_model is not None:
        boxes = [d.box_xyxy for d in dets]
        masks_bool = run_sam2_masks(image, boxes, sam_proc, sam_model, amp_dtype)

        dets, masks_bool = filter_and_refine_with_masks(
            dets=dets,
            masks_bool=masks_bool,
            image_bgr=bgr,
            bleed_min_red_ratio=args.bleed_min_red_ratio,
            bleed_min_sat=args.bleed_min_sat,
            bleed_max_area_frac=args.bleed_max_area_frac,
            tighten_boxes=args.tighten_boxes,
        )

        dets = groupwise_nms_and_filter(dets, nms_iou=args.nms_iou, max_dets=args.max_dets)

        boxes = [d.box_xyxy for d in dets]
        masks_bool = run_sam2_masks(image, boxes, sam_proc, sam_model, amp_dtype)

    base = os.path.splitext(os.path.basename(image_path))[0]
    out_img = os.path.join(out_dir, f"{base}_pretty.jpg")
    out_json = os.path.join(out_dir, f"{base}_report.json")

    rendered = overlay_pretty(
        image=image,
        dets=dets,
        masks_bool=masks_bool,
        mask_alpha=args.mask_alpha,
        use_zh_label=args.use_zh_label,
        font_path=(args.font_path if args.font_path else None),
        font_size=args.font_size,
    )
    cv2.imwrite(out_img, rendered)

    if args.side_by_side:
        orig_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        compare = np.concatenate([orig_bgr, rendered], axis=1)
        cv2.imwrite(os.path.join(out_dir, f"{base}_compare.jpg"), compare)

    report = {
        "input_image": os.path.abspath(image_path),
        "resized_to": list(image.size),
        "models": {
            "detector": args.detector_model,
            "masks": args.sam2_model if args.use_masks else None,
        },
        "params": {
            "global_thresh": args.global_thresh,
            "group_score_thresh": GROUP_SCORE_THRESH,
            "tile": args.tile,
            "overlap": args.overlap,
            "per_tile_max": args.per_tile_max,
            "nms_iou": args.nms_iou,
            "max_dets": args.max_dets,
            "use_templates": args.use_templates,
            "preprocess": args.preprocess,
            "use_bleeding_heuristic": args.use_bleeding_heuristic,
            "bleed_min_red_ratio": args.bleed_min_red_ratio,
            "bleed_min_sat": args.bleed_min_sat,
            "bleed_max_area_frac": args.bleed_max_area_frac,
            "tighten_boxes": args.tighten_boxes,
            "use_fast_processor": args.use_fast_processor,
        },
        "summary": summarize(dets, args.min_score_report),
        "detections": [
            {
                "group": d.group,
                "group_name_en": GROUP_NAME_EN.get(d.group, d.group),
                "group_name_zh": GROUP_NAME_ZH.get(d.group, d.group),
                "query": d.query,
                "score": d.score,
                "box_xyxy": d.box_xyxy,
            }
            for d in dets
        ],
        "outputs": {
            "pretty_image": os.path.abspath(out_img),
            "compare_image": os.path.abspath(os.path.join(out_dir, f"{base}_compare.jpg")) if args.side_by_side else None,
            "json": os.path.abspath(out_json),
        },
        "notes": [
            "No-training prototype. Not clinically reliable.",
            "Aneurysm class removed. Only bleeding + equipment remain.",
        ],
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)

    # Batch inputs
    ap.add_argument("--images", nargs="*", default=[])
    ap.add_argument("--input_dir", default=DEFAULT_INPUT_DIR)  # << CHANGED: default folder
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--patterns", default="*.jpg,*.jpeg,*.png,*.bmp,*.webp")

    ap.add_argument("--device", default="cuda")  # or "auto"
    ap.add_argument("--max_side", type=int, default=960)

    ap.add_argument("--detector_model", default=DEFAULT_DETECTOR_MODEL)
    ap.add_argument("--use_fast_processor", action="store_true")
    ap.add_argument("--global_thresh", type=float, default=DEFAULT_DET_GLOBAL_THRESHOLD)

    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--per_tile_max", type=int, default=20)

    ap.add_argument("--nms_iou", type=float, default=0.55)
    ap.add_argument("--max_dets", type=int, default=16)

    ap.add_argument("--use_templates", action="store_true")
    ap.add_argument("--preprocess", action="store_true")

    ap.add_argument("--use_bleeding_heuristic", action="store_true")
    ap.add_argument("--bleeding_min_area", type=int, default=160)

    ap.add_argument("--use_masks", action="store_true")
    ap.add_argument("--sam2_model", default=DEFAULT_SAM2_MODEL)
    ap.add_argument("--mask_alpha", type=float, default=0.38)

    ap.add_argument("--bleed_min_red_ratio", type=float, default=0.18)
    ap.add_argument("--bleed_min_sat", type=float, default=0.25)
    ap.add_argument("--bleed_max_area_frac", type=float, default=0.08)
    ap.add_argument("--tighten_boxes", action="store_true")

    ap.add_argument("--side_by_side", action="store_true")
    ap.add_argument("--min_score_report", type=float, default=0.30)

    ap.add_argument("--use_zh_label", action="store_true")
    ap.add_argument("--font_path", default="")  # e.g. C:\Windows\Fonts\msyh.ttc
    ap.add_argument("--font_size", type=int, default=18)

    args = ap.parse_args()
    ensure_dir(args.out_dir)

    device = pick_device(args.device)
    enable_fast_cuda()
    amp_dtype = best_amp_dtype(device)

    # Collect image list (folder first, then fall back to DEFAULT_IMAGE_PATH)
    img_list = list_images_from_args(args.images, args.input_dir, args.recursive, args.patterns)

    if not img_list:
        if os.path.isfile(DEFAULT_IMAGE_PATH):
            img_list = [os.path.abspath(DEFAULT_IMAGE_PATH)]
        else:
            raise FileNotFoundError(
                f"No input images found in: {args.input_dir}\n"
                "Use --images or --input_dir, or check your --patterns."
            )

    print(f"Found {len(img_list)} image(s) to process.")

    # Load models once
    processor = load_ovd_processor(args.detector_model, use_fast_processor=args.use_fast_processor)
    model = load_ovd_model(args.detector_model, device=device, amp_dtype=amp_dtype)

    sam_proc, sam_model = None, None
    if args.use_masks:
        if not SAM2_AVAILABLE:
            raise RuntimeError("SAM2 not available. Try: pip install -U transformers")
        sam_proc, sam_model = load_sam2_model(args.sam2_model, device=device, amp_dtype=amp_dtype)

    batch_reports = []
    for idx, p in enumerate(img_list, 1):
        print(f"[{idx}/{len(img_list)}] {p}")
        rep = process_one_image(
            image_path=p,
            out_dir=args.out_dir,
            processor=processor,
            model=model,
            amp_dtype=amp_dtype,
            args=args,
            sam_proc=sam_proc,
            sam_model=sam_model,
        )
        batch_reports.append(rep)

    batch_json = os.path.join(args.out_dir, "batch_report.json")
    with open(batch_json, "w", encoding="utf-8") as f:
        json.dump(batch_reports, f, ensure_ascii=False, indent=2)

    print("Saved batch report:", os.path.abspath(batch_json))


if __name__ == "__main__":
    main()