"""Optional E5 reward; uses the unchanged official parser/IoU implementation."""
from scripts.score import iou_xyxy, parse_output


def grounding_reward(raw, gt_boxes):
    pred = parse_output(raw)
    if pred is None:
        return 0.0
    if not gt_boxes and not pred:
        return 1.0
    pairs = sorted((iou_xyxy(g["bbox"], p["bbox"]), i, j)
                   for i, g in enumerate(gt_boxes) for j, p in enumerate(pred)
                   if g["label"] == p["label"])
    used_g, used_p, total = set(), set(), 0.0
    for value, i, j in reversed(pairs):
        if value < 0.5:
            break
        if i not in used_g and j not in used_p:
            used_g.add(i)
            used_p.add(j)
            total += value
    return 0.2 + 0.8 * total / max(len(gt_boxes), len(pred), 1)


def grpo_reward(completions, gt_boxes, **kwargs):
    raw = []
    for c in completions:
        if isinstance(c, str):
            raw.append(c)
        else:
            content = c[-1]["content"]
            raw.append(content if isinstance(content, str) else "".join(
                p.get("text", "") for p in content if p.get("type") == "text"))
    return [grounding_reward(text, gt) for text, gt in zip(raw, gt_boxes)]
