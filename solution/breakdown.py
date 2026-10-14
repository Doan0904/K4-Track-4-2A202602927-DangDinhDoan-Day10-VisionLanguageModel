"""CPU error breakdown from real test predictions: per session, per GT-box size, FP types.

    python -m solution.breakdown            # needs data/splits/test.jsonl and work/e{1,2,3}.jsonl
Writes report/breakdown.json. Uses the official parser/IoU of scripts/score.py, no model, no GPU.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts.labcommon import CLASSES, read_jsonl
from scripts.score import IOU_THR, _gt_boxes, iou_xyxy, parse_output

SIZE_BINS = (("small<0.2%", 0.0, 0.002), ("medium0.2-1%", 0.002, 0.01), ("large>=1%", 0.01, 10.0))


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) / 1e6


def match_image(gt, pred):
    """Official greedy same-class IoU>=0.5 matching; returns matched gt ids and pred ids."""
    used_g, used_p = set(), set()
    for lab in {g["label"] for g in gt} | {p["label"] for p in pred}:
        gi = [i for i, g in enumerate(gt) if g["label"] == lab]
        pi = [j for j, p in enumerate(pred) if p["label"] == lab]
        pairs = sorted(((iou_xyxy(gt[i]["bbox"], pred[j]["bbox"]), i, j) for i in gi for j in pi), reverse=True)
        for v, i, j in pairs:
            if v < IOU_THR:
                break
            if i in used_g or j in used_p:
                continue
            used_g.add(i)
            used_p.add(j)
    return used_g, used_p


def fp_type(p, gt):
    best_same = max((iou_xyxy(p["bbox"], g["bbox"]) for g in gt if g["label"] == p["label"]), default=0.0)
    best_other = max((iou_xyxy(p["bbox"], g["bbox"]) for g in gt if g["label"] != p["label"]), default=0.0)
    if best_other >= IOU_THR:
        return "wrong_class"
    if best_same >= 0.1:
        return "localization(0.1<=IoU<0.5)"
    if p["label"] not in CLASSES:
        return "unknown_label"
    return "no_overlap_with_same_class_gt"


def analyse(rows, preds):
    tp = fp = fn = 0
    per_size = {k: [0, 0] for k, _, _ in SIZE_BINS}          # [matched, total] ground-truth boxes
    per_class_gt = {c: [0, 0] for c in CLASSES}
    fp_types, repeated_images, big_pred_images = Counter(), 0, 0
    pred_counts, gt_counts = [], []
    empty_gt_images = empty_gt_with_fp = 0
    for row in rows:
        gt = _gt_boxes(row)
        pred = parse_output(preds[row["image_id"]]) or []
        used_g, used_p = match_image(gt, pred)
        tp += len(used_g)
        fp += len(pred) - len(used_p)
        fn += len(gt) - len(used_g)
        pred_counts.append(len(pred))
        gt_counts.append(len(gt))
        for i, g in enumerate(gt):
            for name, lo, hi in SIZE_BINS:
                if lo <= area(g["bbox"]) < hi:
                    per_size[name][1] += 1
                    per_size[name][0] += int(i in used_g)
            per_class_gt[g["label"]][1] += 1
            per_class_gt[g["label"]][0] += int(i in used_g)
        for j, p in enumerate(pred):
            if j not in used_p:
                fp_types[fp_type(p, gt)] += 1
        if len(pred) >= 6:
            big_pred_images += 1
        near_duplicates = sum(1 for a in range(len(pred)) for b in range(a + 1, len(pred))
                              if pred[a]["label"] == pred[b]["label"]
                              and iou_xyxy(pred[a]["bbox"], pred[b]["bbox"]) > 0.7)
        repeated_images += int(near_duplicates >= 2)
        if not gt:
            empty_gt_images += 1
            empty_gt_with_fp += int(bool(pred))
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "f1": round(f1, 4),
            "mean_pred_per_image": round(sum(pred_counts) / len(rows), 3),
            "mean_gt_per_image": round(sum(gt_counts) / len(rows), 3),
            "images_with_6plus_predictions": big_pred_images,
            "images_with_2plus_near_duplicate_predictions": repeated_images,
            "images_without_gt": empty_gt_images, "images_without_gt_but_predicted": empty_gt_with_fp,
            "gt_recall_by_box_size": {k: {"matched": m, "total": t, "recall": round(m / t, 3) if t else None}
                                      for k, (m, t) in per_size.items()},
            "gt_recall_by_class": {k: {"matched": m, "total": t, "recall": round(m / t, 3) if t else None}
                                   for k, (m, t) in per_class_gt.items()},
            "fp_types": dict(fp_types)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--out", default="report/breakdown.json")
    ap.add_argument("--runs", nargs="+", default=["e1", "e2", "e3"])
    args = ap.parse_args()
    rows = read_jsonl(Path(args.data_dir) / "splits/test.jsonl")
    result = {"split": "test", "n_images": len(rows), "runs": {}, "by_session": {}}
    sessions = defaultdict(list)
    for r in rows:
        sessions[r["session"]].append(r)
    for run in args.runs:
        preds = {r["image_id"]: r["raw"] for r in read_jsonl(Path(args.work_dir) / f"{run}.jsonl")}
        result["runs"][run] = analyse(rows, preds)
        for session, items in sorted(sessions.items()):
            entry = result["by_session"].setdefault(session, {
                "images": len(items), "gt_boxes": sum(len(r["objects"]) for r in items)})
            entry[run] = analyse(items, preds)["f1"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
