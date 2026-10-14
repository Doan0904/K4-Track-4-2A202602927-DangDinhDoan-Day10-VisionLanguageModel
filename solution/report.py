"""Fill report blocks from real scorer evidence; never invent missing values."""
import argparse
import json
import math
import re
from pathlib import Path

from solution.run import latest_checkpoint


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def render_report(root=".", write=False):
    root = Path(root)
    keys = ("e0", "e1", "e2", "e3", "e4_data_010", "e4_data_100")
    ev = {key: read(root/"report/evidence"/f"{key}.json") for key in keys}
    analysis = read(root/"report/analysis.json")
    version = analysis["dataset_version"]
    for key, result in ev.items():
        if (result["score_version"] != 1 or result["split"] != "test" or result["missing_predictions"]
                or result["dataset_version"] != version):
            raise ValueError(f"Invalid official evidence: {key}")
        if key != "e0":
            for field in ("trainable_params", "peak_vram_gb", "train_minutes"):
                v = result["meta"].get(field)
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
                    raise ValueError(f"Missing real training metadata: {key}.{field}")
    blocks = {"VERSION": f"`dataset_version`: `{version}`. Số liệu lấy từ trọng tài, cùng split test."}
    table = ["| Mã | Tham số train | VRAM (GB) | Phút train | F1 | CI95 | FP/ảnh | pothole | crack | manhole |",
             "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
    for key, r in ev.items():
        meta, ci = r["meta"], r["f1_ci95"]
        values = [key.upper(), str(int(meta.get("trainable_params", 0))), f"{meta['peak_vram_gb']:.3f}",
                  f"{meta['train_minutes']:.3f}" if key != "e0" else "—", f"{r['micro']['f1']:.4f}",
                  f"[{ci[0]:.4f}, {ci[1]:.4f}]", f"{r['fp_per_image']:.4f}"]
        values += [f"{r['per_class'][c]['f1']:.4f}" for c in ("pothole", "crack", "manhole")]
        table.append("| "+" | ".join(values)+" |")
    blocks["TABLE"] = "\n".join(table)+"\n\nE0 không train; VRAM E0 đo khi inference. Các run khác ghi peak allocated VRAM khi train."
    run = read(latest_checkpoint(root/"work/e1")/"run.json")
    counts = run["architecture"]
    blocks["Q1"] = (f"Đo trên model: ViT backbone {counts['vit_backbone']:,}; projector {counts['projector']:,}; "
                    f"LLM + embedding {counts['llm_including_embeddings']:,}; tổng {counts['total']:,} tham số.\n\n"
                    +"; ".join(f"{k.upper()}: {int(ev[k]['meta']['trainable_params']):,} trainable" for k in ("e1", "e2", "e3"))+".")
    err = analysis["errors"]["e0"]
    top = sorted(err["categories_count_images"].items(), key=lambda x: x[1], reverse=True)[:2]
    example = err["raw_examples"][0]
    blocks["Q2"] = (f"E0 parse_valid_rate={ev['e0']['parse_valid_rate']:.4f}. Nhóm lỗi nhiều nhất theo quy tắc đếm ảnh: "
                    +(", ".join(f"`{k}` ({n} ảnh)" for k, n in top) or "không có lỗi được quy tắc phát hiện")
                    +f". Ví dụ output thô `{example['image_id']}`:\n\n````text\n{example['raw']}\n````")
    control = analysis["coordinate_control"]
    blocks["Q3"] = (f"Cùng output E1: đúng norm1000 F1={control['correct_norm1000_f1']:.4f}; "
                    f"hiểu nhầm thành pixel F1={control['wrong_pixel_f1']:.4f}. Không train thêm trong kiểm tra này.")
    encoding = read(root/"work/e1/encoding_check.json")
    blocks["Q4"] = (f"Mẫu `{encoding['image_id']}`: {encoding['image_tokens']} token ảnh, "
                    f"{encoding['prompt_tokens']} token bị mask và {encoding['answer_tokens']} token được giám sát.\n\n"
                    f"````text\n{encoding['supervised_text']}\n````")
    audit = analysis["audit"]["splits"]
    blocks["Q6"] = "\n".join(f"- {s}: {v['images']} ảnh, {v['sessions']} session; "
                               +", ".join(f"{c}={v['boxes'][c]} box" for c in ("pothole", "crack", "manhole"))
                               for s, v in audit.items())+"\n\nKhông có session trùng giữa các split theo audit."
    paired = analysis["paired"]
    def pair_text(key):
        p = paired[key]
        lo, hi = p["delta_ci95"]
        conclusion = "CI loại trừ 0 trong bootstrap theo ảnh" if p["excludes_zero"] else "CI chứa 0; chưa đủ bằng chứng thay đổi có hướng"
        return f"`{key}`: ΔF1={p['delta_f1_b_minus_a']:.4f}, paired CI95=[{lo:.4f}, {hi:.4f}]; {conclusion}."
    selection = read(root/"work/configs/selection.json")
    blocks["Q7"] = "\n\n".join(pair_text(k) for k in ("e2-e1", "e3-e2"))+f"\n\nSetup nền chọn bằng validation: {selection['base'].upper()} (F1 val={selection['validation_f1']:.4f}). Chi phí ghi ở bảng trên."
    blocks["Q8"] = f"Test thực tế {ev['e0']['n_images']} ảnh. Độ rộng CI95: "+"; ".join(
        f"{k.upper()}={ev[k]['f1_ci95'][1]-ev[k]['f1_ci95'][0]:.4f}" for k in ("e0", "e1", "e2", "e3"))+"."
    worst = min(("pothole", "crack", "manhole"), key=lambda c: ev['e1']['per_class'][c]['f1'])
    area = audit["test"]["median_box_area"][worst]
    blocks["Q9"] = (f"Trong E1, lớp kém nhất là {worst}: F1={ev['e1']['per_class'][worst]['f1']:.4f}; "
                    + (f"diện tích box trung vị trên test={area*100:.4f}% ảnh. " if area is not None else "test không có box lớp này. ")
                    + f"Có {analysis['figure_count']} ảnh trong [bộ ảnh minh hoạ](report/figures/predictions/README.md). "
                    "Cần đọc từng ảnh để xác nhận giả thuyết về nguyên nhân lỗi.")
    small = read(latest_checkpoint(root/"work/e4_data_010")/"run.json")
    large_path = Path(selection["checkpoint"])
    if not large_path.is_absolute():
        large_path = root/large_path
    large = read(large_path/"run.json")
    blocks["Q10"] = (f"Setup nền {selection['base'].upper()}; tập 10%={small['train_images']} ảnh, "
                     f"100%={large['train_images']} ảnh. "+pair_text("e4_data_100-e4_data_010"))
    if "e5" in analysis:
        stats = analysis["e5"]
        bonus = read(root/"report/evidence/e5.json")
        blocks["E5"] = (f"E5 F1={bonus['micro']['f1']:.4f}, CI95={bonus['f1_ci95']}; "
                        f"reward trung bình test={stats['mean_test_reward']:.4f}, "
                        f"tỷ lệ output rỗng={stats['empty_output_rate']:.4f}, "
                        f"ảnh có box ≥50% diện tích={stats['giant_box_image_rate']:.4f}. "
                        +pair_text(f"e5-{stats['source_sft']}")+
                        " Đọc log và ảnh để xác định reward hacking; các tỷ lệ chỉ là tín hiệu kiểm tra.")
    else:
        blocks["E5"] = "Chưa chạy GRPO, chưa có bonus evidence hoặc kết luận cải thiện."
    text = (root/"REPORT.md").read_text(encoding="utf-8")
    for key, value in blocks.items():
        pattern = rf"(<!-- AUTO:{key}:START -->).*?(<!-- AUTO:{key}:END -->)"
        text, count = re.subn(pattern, lambda m: m[1]+"\n"+value+"\n"+m[2], text, flags=re.S)
        if count != 1:
            raise ValueError(f"Missing/duplicate report marker: {key}")
    text = text.replace("Trạng thái: **đã hoàn thiện mã và cấu hình; chưa chạy mô hình hoặc huấn luyện GPU**.",
                        "Trạng thái: **đã có số liệu E0–E3 và E4 từ lần chạy GPU; cần đọc và duyệt phân tích trước khi nộp**.")
    text = text.replace("Kết quả chưa được đo trên GPU; chưa thể xác nhận F1 E1 vượt E0 ≥0.10 hay parse_valid_rate ≥0.95.",
                        f"E1−E0 F1={ev['e1']['micro']['f1']-ev['e0']['micro']['f1']:.4f}; E1 parse_valid_rate={ev['e1']['parse_valid_rate']:.4f}. Đối chiếu ngưỡng rubric trước khi nộp.")
    if "<điền" in text:
        raise ValueError("Report still contains unfilled measurement blocks")
    destination = root/"REPORT.md" if write else root/"report/REPORT_PREVIEW.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    return text


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--write", action="store_true")
    args = p.parse_args()
    render_report(write=args.write)
    print("Wrote REPORT.md" if args.write else "Wrote report/REPORT_PREVIEW.md")


if __name__ == "__main__":
    main()
