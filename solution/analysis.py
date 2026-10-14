"""CPU analysis from actual predictions: paired CI, coordinate control, annotated images."""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from scripts.labcommon import CLASSES, read_jsonl
from scripts.score import _gt_boxes, _match, _to_norm1000, evaluate, iou_xyxy, parse_output
from solution.config import write_json
from solution.data import audit_data


def prediction_map(path, rows):
    preds = {}
    for row in read_jsonl(path):
        if row["image_id"] in preds:
            raise ValueError("Duplicate prediction image_id")
        preds[row["image_id"]] = row["raw"]
    if set(preds) != {r["image_id"] for r in rows}:
        raise ValueError("Analysis requires complete predictions for the exact split")
    return preds


def per_image_counts(rows, preds, coord="norm1000"):
    result = []
    for row in rows:
        pred = parse_output(preds[row["image_id"]]) or []
        pred = _to_norm1000(pred, coord, row["width"], row["height"])
        per, _ = _match(_gt_boxes(row), pred)
        result.append(tuple(sum(v[k] for v in per.values()) for k in range(3)))
    return result


def f1(counts):
    tp, fp, fn = (sum(r[k] for r in counts) for k in range(3))
    return 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0


def paired_ci(rows, a, b, coord_a="norm1000", coord_b="norm1000", n_boot=1000, seed=42):
    if not rows or n_boot < 40:
        raise ValueError("Paired bootstrap requires nonempty rows and at least 40 replicates")
    ca, cb = per_image_counts(rows, a, coord_a), per_image_counts(rows, b, coord_b)
    rng, deltas = random.Random(seed), []
    for _ in range(n_boot):
        indices = [rng.randrange(len(rows)) for _ in rows]
        deltas.append(f1([cb[i] for i in indices])-f1([ca[i] for i in indices]))
    deltas.sort()
    low, high = deltas[int(.025*n_boot)], deltas[int(.975*n_boot)-1]
    return {"delta_f1_b_minus_a": f1(cb)-f1(ca), "delta_ci95": [low, high],
            "excludes_zero": low > 0 or high < 0, "bootstrap_unit": "image", "n_boot": n_boot}


def error_summary(rows, preds, coord="norm1000"):
    counts, examples = Counter(), []
    for row in rows:
        raw = preds[row["image_id"]]
        parsed = parse_output(raw)
        if parsed is None:
            counts["invalid_json"] += 1
        parsed = _to_norm1000(parsed or [], coord, row["width"], row["height"])
        gt = _gt_boxes(row)
        per, _ = _match(gt, parsed)
        tp, fp, fn = (sum(v[k] for v in per.values()) for k in range(3))
        if fp:
            counts["extra_or_mislocalized_prediction"] += 1
        if fn:
            counts["missed_or_mislocalized_gt"] += 1
        if any(g["label"] != p["label"] and iou_xyxy(g["bbox"], p["bbox"]) >= .5
               for g in gt for p in parsed):
            counts["class_confusion"] += 1
        if not raw.strip():
            counts["empty_output"] += 1
        examples.append({"image_id": row["image_id"], "raw": raw, "tp": tp, "fp": fp, "fn": fn})
    return {"categories_count_images": dict(counts),
            "note": "Categories may overlap; FP/FN do not distinguish missing boxes from poor localization.",
            "raw_examples": sorted(examples, key=lambda r: (r["fp"]+r["fn"]), reverse=True)[:5]}


def draw_example(row, data_dir, out_path, pred=None, coord="norm1000", title="ground truth"):
    with Image.open(Path(data_dir)/row["image_path"]) as src:
        image = src.convert("RGB")
    canvas = Image.new("RGB", (image.width, image.height+70), "white")
    canvas.paste(image, (0, 70))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), title.encode("ascii", "replace").decode(), fill="black")
    draw.text((8, 23), row["image_id"], fill="black")
    draw.text((8, 41), "GREEN: ground truth | RED: prediction", fill="black")
    pairs = [(g, "lime") for g in _gt_boxes(row)]
    if pred is not None:
        pairs += [(p, "red") for p in _to_norm1000(parse_output(pred) or [], coord, row["width"], row["height"])]
    for box, color in pairs:
        x1, y1, x2, y2 = box["bbox"]
        xy = (x1*image.width/1000, y1*image.height/1000+70,
              x2*image.width/1000, y2*image.height/1000+70)
        draw.rectangle(xy, outline=color, width=2)
        draw.text((xy[0], max(70, xy[1]-12)), box["label"], fill=color)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def dataset_figures(data_dir="data", out_dir="report/figures/labels"):
    audit_data(data_dir)
    rows = read_jsonl(Path(data_dir)/"splits/train.jsonl")
    for i, row in enumerate(rows[:6]):
        draw_example(row, data_dir, Path(out_dir)/f"label_{i+1:02d}.png")
    return min(len(rows), 6)


def prediction_figures(rows, preds, data_dir, out_dir, name, coord="norm1000"):
    ranked = list(zip(rows, per_image_counts(rows, preds, coord)))
    correct = [r for r, (tp, fp, fn) in ranked if tp and fp+fn == 0]
    wrong = [r for r, (_, fp, fn) in sorted(ranked, key=lambda x: x[1][1]+x[1][2], reverse=True) if fp+fn]
    chosen = [(r, "correct") for r in correct[:3]]+[(r, "error") for r in wrong[:3]]
    seen = {r["image_id"] for r, _ in chosen}
    for r, (tp, fp, fn) in ranked:
        if len(chosen) >= 6:
            break
        if r["image_id"] not in seen:
            chosen.append((r, "mixed" if fp+fn else "correct"))
            seen.add(r["image_id"])
    captions = []
    for i, (r, category) in enumerate(chosen):
        counts = per_image_counts([r], {r["image_id"]: preds[r["image_id"]]}, coord)[0]
        filename = f"{name}_{i+1:02d}_{category}.png"
        draw_example(r, data_dir, Path(out_dir)/filename, preds[r["image_id"]], coord,
                     f"{name} {category}: TP={counts[0]} FP={counts[1]} FN={counts[2]}")
        captions.append(f"![{name} {category}]({filename})\n\n"
                        f"`{r['image_id']}` — TP={counts[0]}, FP={counts[1]}, FN={counts[2]}. "
                        "Box xanh: nhãn thật; box đỏ: dự đoán.\n")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir)/"README.md").write_text("# Ảnh minh hoạ từ dự đoán thật\n\n"+"\n".join(captions), encoding="utf-8")
    return len(chosen)


def build_analysis(data_dir="data", work_dir="work", evidence_dir="report/evidence", output="report/analysis.json"):
    rows = read_jsonl(Path(data_dir)/"splits/test.jsonl")
    audit = audit_data(data_dir)
    evidence = {p.stem: json.loads(p.read_text()) for p in sorted(Path(evidence_dir).glob("e*.json"))}
    result = {"dataset_version": audit["dataset_version"], "audit": audit, "errors": {}, "paired": {}}
    preds = {}
    for key, ev in evidence.items():
        if ev["split"] != "test" or ev["dataset_version"] != result["dataset_version"] or ev["missing_predictions"]:
            raise ValueError(f"Invalid evidence for analysis: {key}")
        pred_path = Path(work_dir)/f"{key}.jsonl"
        if not pred_path.exists():
            raise FileNotFoundError(f"Raw predictions required: {pred_path}")
        preds[key] = prediction_map(pred_path, rows)
        measured = evaluate(rows, preds[key], coord=ev["coord"])
        if abs(measured["micro"]["f1"]-ev["micro"]["f1"]) > 1e-10:
            raise ValueError(f"Raw predictions disagree with evidence: {key}")
        result["errors"][key] = error_summary(rows, preds[key], ev["coord"])
    for a, b in (("e0", "e1"), ("e1", "e2"), ("e2", "e3"), ("e4_data_010", "e4_data_100")):
        if a in preds and b in preds:
            result["paired"][f"{b}-{a}"] = paired_ci(rows, preds[a], preds[b], evidence[a]["coord"], evidence[b]["coord"])
    if "e1" in preds:
        norm = evaluate(rows, preds["e1"], coord="norm1000")
        pixel = evaluate(rows, preds["e1"], coord="pixel")
        result["coordinate_control"] = {"correct_norm1000_f1": norm["micro"]["f1"],
                                        "wrong_pixel_f1": pixel["micro"]["f1"],
                                        "method": "Same raw E1 outputs, same GT; only evaluation coordinate interpretation changes."}
        result["figure_count"] = prediction_figures(rows, preds["e1"], data_dir, Path(output).parent/"figures/predictions", "e1")
    if "e5" in preds:
        from solution.reward import grounding_reward
        selection = json.loads((Path(work_dir)/"configs/selection.json").read_text())
        base = selection["base"]
        result["paired"][f"e5-{base}"] = paired_ci(rows, preds[base], preds["e5"], evidence[base]["coord"], evidence["e5"]["coord"])
        empty, giant, rewards = 0, 0, []
        for row in rows:
            raw = preds["e5"][row["image_id"]]
            parsed = parse_output(raw)
            empty += int(parsed == [])
            giant += int(any((p["bbox"][2]-p["bbox"][0])*(p["bbox"][3]-p["bbox"][1]) >= .5*1000**2
                             for p in parsed or []))
            rewards.append(grounding_reward(raw, _gt_boxes(row)))
        result["e5"] = {"source_sft": base, "mean_test_reward": sum(rewards)/len(rows),
                        "empty_output_rate": empty/len(rows), "giant_box_image_rate": giant/len(rows),
                        "giant_definition": "predicted box area >=50% of original frame",
                        "note": "Heuristics signal possible reward hacking; inspect examples and training logs."}
    write_json(output, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--work-dir", default="work")
    p.add_argument("--labels-only", action="store_true")
    args = p.parse_args()
    if args.labels_only:
        print(f"Created {dataset_figures(args.data_dir)} label images")
    else:
        print(json.dumps(build_analysis(args.data_dir, args.work_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
