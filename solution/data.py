"""Labels, prompts and split checks; usable on CPU without torch."""
import json
import math
import random
from collections import Counter
from pathlib import Path

from scripts.labcommon import CLASSES, read_jsonl


def prompt(coord="norm1000", width=640, height=360):
    if coord == "norm1000":
        scale = "integer coordinates normalized to 0–1000 relative to image width and height"
    elif coord == "pixel":
        scale = f"integer pixel coordinates in the original {width}x{height} image"
    else:
        raise ValueError(f"Unknown coordinate scale: {coord}")
    return (
        "Locate all road damage and manhole covers in this dashcam image. "
        "Allowed labels: pothole, crack, manhole. Return ONLY a JSON list, "
        'each item {"label":"pothole|crack|manhole","bbox_2d":[x1,y1,x2,y2]}. '
        f"Use {scale}, with x1<x2 and y1<y2. "
        "Sort boxes top to bottom, then left to right. Return [] if there are none. "
        "Do not include markdown, explanations or extra keys."
    )


def yolo_to_box(label, yolo, coord="norm1000", width=640, height=360):
    """Strict conversion of one box; invalid geometry raises ValueError."""
    if label not in CLASSES or len(yolo) != 4:
        raise ValueError("Invalid label or YOLO box")
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in yolo):
        raise ValueError("YOLO values must be finite and in [0,1]")
    if coord not in ("norm1000", "pixel"):
        raise ValueError(coord)
    xc, yc, w, h = yolo
    if w <= 0 or h <= 0:
        raise ValueError("Box must have positive width and height")
    sx, sy = (1000, 1000) if coord == "norm1000" else (width, height)
    values = [(xc-w/2)*sx, (yc-h/2)*sy, (xc+w/2)*sx, (yc+h/2)*sy]
    bb = [max(0, min(limit, round(v))) for v, limit in zip(values, (sx, sy, sx, sy))]
    if bb[2] <= bb[0] or bb[3] <= bb[1]:
        raise ValueError("Box collapsed after rounding; inspect the annotation")
    return {"label": label, "bbox_2d": bb}


def _target_box(obj, coord="norm1000", width=640, height=360):
    """Exclude unusable supervision without changing official annotations."""
    try:
        return yolo_to_box(obj["label"], obj["yolo"], coord, width, height)
    except ValueError as exc:
        if str(exc) not in ("Box must have positive width and height",
                            "Box collapsed after rounding; inspect the annotation"):
            raise
        return None


def objects_to_answer(objects, coord="norm1000", width=640, height=360):
    boxes = [_target_box(o, coord, width, height) for o in objects]
    boxes = [b for b in boxes if b is not None]
    boxes.sort(key=lambda b: (b["bbox_2d"][1], b["bbox_2d"][0], b["label"], b["bbox_2d"]))
    return json.dumps(boxes, ensure_ascii=False, separators=(",", ":"))


def user_message(image, text):
    return {"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": text}]}


def subset_train(rows, fraction=1.0, seed=42):
    if not rows or not 0 < fraction <= 1:
        raise ValueError("Train split must be nonempty and fraction in (0,1]")
    # Same deterministic ordering for both E4 values; the 10% set is a prefix.
    rows = sorted(rows, key=lambda r: r["image_id"])
    random.Random(seed).shuffle(rows)
    return rows[:max(1, int(len(rows) * fraction))]


def audit_data(data_dir="data"):
    root = Path(data_dir)
    meta = json.loads((root / "splits/meta.json").read_text(encoding="utf-8"))
    stats, seen_sessions, seen_ids = {}, {}, set()
    for split in ("train", "val", "test"):
        rows = read_jsonl(root / "splits" / f"{split}.jsonl")
        counts = Counter({c: 0 for c in CLASSES})
        unusable = Counter({c: 0 for c in CLASSES})
        unusable_examples = []
        sessions, areas = set(), {c: [] for c in CLASSES}
        for row in rows:
            iid, session = row["image_id"], row["session"]
            if iid in seen_ids:
                raise ValueError(f"Duplicate image_id across splits: {iid}")
            seen_ids.add(iid)
            if seen_sessions.get(session, split) != split:
                raise ValueError(f"Session leakage: {session}")
            seen_sessions[session] = split
            sessions.add(session)
            if not (root / row["image_path"]).is_file():
                raise FileNotFoundError(row["image_path"])
            objects_to_answer(row["objects"], width=row["width"], height=row["height"])
            for index, o in enumerate(row["objects"]):
                counts[o["label"]] += 1
                areas[o["label"]].append(o["yolo"][2] * o["yolo"][3])
                if _target_box(o, width=row["width"], height=row["height"]) is None:
                    unusable[o["label"]] += 1
                    if len(unusable_examples) < 10:
                        unusable_examples.append({"image_id": iid, "object_index": index,
                                                  "label": o["label"], "yolo": o["yolo"],
                                                  "reason": "zero_area" if min(o["yolo"][2:]) == 0
                                                  else "collapsed_after_rounding_or_clipping"})
        stats[split] = {"images": len(rows), "boxes": dict(counts), "sessions": len(sessions),
                        "median_box_area": {c: _median(a) for c, a in areas.items()},
                        "unusable_target_boxes": dict(unusable), "unusable_target_examples": unusable_examples}
        expected = meta["counts"][split]
        if expected["images"] != len(rows) or expected["boxes"] != dict(counts):
            raise ValueError(f"Counts disagree with meta.json for {split}")
    return {"dataset_version": meta["dataset_version"], "session_overlap": False, "splits": stats}


def _median(values):
    if not values:
        return None
    s, n = sorted(values), len(values)
    return (s[(n-1)//2] + s[n//2]) / 2


class BatchStream:
    """Deterministic shuffled batches, including resumable position and RNG."""
    def __init__(self, size, batch_size, seed):
        if size < 1 or batch_size < 1:
            raise ValueError("Empty dataset or invalid batch size")
        self.order = list(range(size))
        self.rng = random.Random(seed)
        self.rng.shuffle(self.order)
        self.cursor = 0
        self.batch_size = batch_size

    def next_indices(self):
        result = []
        while len(result) < self.batch_size:
            if self.cursor == len(self.order):
                self.rng.shuffle(self.order)
                self.cursor = 0
            result.append(self.order[self.cursor])
            self.cursor += 1
        return result

    def state_dict(self):
        return {"order": self.order[:], "cursor": self.cursor, "rng": self.rng.getstate()}

    def load_state_dict(self, state):
        if sorted(state["order"]) != list(range(len(self.order))):
            raise ValueError("Resume dataset size differs")
        self.order, self.cursor = state["order"][:], state["cursor"]
        self.rng.setstate(state["rng"])
