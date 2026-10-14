"""Raw prediction JSONL with provenance and explicit interruption recovery."""
import hashlib
import json
import time
from pathlib import Path

from PIL import Image

from scripts.labcommon import read_jsonl
from solution.config import write_json
from solution.data import prompt, user_message
from solution.modeling import load_base, load_checkpoint, require_gpu


def predict(cfg, data_dir, split, pred_path, checkpoint=None, batch_size=4, limit=None, resume=False):
    torch = require_gpu()
    if batch_size < 1 or limit is not None and limit < 1:
        raise ValueError("batch_size/limit must be positive")
    if split == "test" and limit is not None:
        raise ValueError("Use --split val for a limited debug run; test must be complete")
    meta = json.loads((Path(data_dir) / "splits/meta.json").read_text())
    rows = read_jsonl(Path(data_dir) / "splits" / f"{split}.jsonl")
    if limit:
        rows = rows[:limit]
    path = Path(pred_path)
    manifest_path = path.with_suffix(".manifest.json")
    # Resume E0 against the exact base-model commit resolved during its first call.
    if resume and manifest_path.exists() and not checkpoint and cfg["revision"] == "main":
        old = json.loads(manifest_path.read_text())
        cfg["revision"] = old["identity"]["config"]["revision"]
    identity = {"dataset_version": meta["dataset_version"], "split": split, "coord": cfg["coord"],
                "image_ids": [r["image_id"] for r in rows], "config": cfg,
                "checkpoint_run": json.loads((Path(checkpoint) / "run.json").read_text()) if checkpoint else None}
    def digest():
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    fingerprint = digest()
    existing = []
    if path.exists():
        if not resume:
            raise FileExistsError(f"Predictions already exist: {path}; use --resume explicitly")
        manifest = json.loads(manifest_path.read_text())
        if manifest["fingerprint"] != fingerprint:
            raise ValueError("Prediction resume belongs to a different config/checkpoint/split")
        existing = read_jsonl(path)
        if [r["image_id"] for r in existing] != identity["image_ids"][:len(existing)]:
            raise ValueError("Predictions are not a unique prefix of the requested split")
    if len(existing) == len(rows):
        print("All predictions already present; model load skipped")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    model, processor = load_checkpoint(cfg, checkpoint) if checkpoint else load_base(cfg)
    if not checkpoint:
        cfg["revision"] = getattr(model.config, "_commit_hash", None) or cfg["revision"]
        fingerprint = digest()
    write_json(manifest_path, {"fingerprint": fingerprint, "identity": identity, "finished": False})
    processor.tokenizer.padding_side = "left"
    model.eval()
    model.config.use_cache = True
    start = time.perf_counter()
    truncated, done = 0, len(existing)
    if existing:
        truncated = sum(bool(r.get("truncated")) for r in existing)
    with path.open("a" if resume else "w", encoding="utf-8") as output, torch.inference_mode():
        for offset in range(done, len(rows), batch_size):
            chunk = rows[offset:offset+batch_size]
            messages = []
            for row in chunk:
                with Image.open(Path(data_dir) / row["image_path"]) as src:
                    image = src.convert("RGB")
                messages.append([user_message(image, prompt(cfg["coord"], row["width"], row["height"]))])
            inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt", padding=True)
            inputs.pop("token_type_ids", None)
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            prefix_width = inputs["input_ids"].shape[1]
            with torch.autocast("cuda", dtype=torch.float16):
                generated = model.generate(**inputs, max_new_tokens=cfg["max_new_tokens"], do_sample=False,
                                           pad_token_id=processor.tokenizer.pad_token_id)
            tails = generated[:, prefix_width:]
            raw = processor.batch_decode(tails, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
            for row, text, tokens in zip(chunk, raw, tails):
                cut = len(tokens) >= cfg["max_new_tokens"] and not (tokens == end_id).any().item()
                output.write(json.dumps({"image_id": row["image_id"], "raw": text, "truncated": cut}, ensure_ascii=False)+"\n")
                truncated += int(cut)
            output.flush()
            done = offset+len(chunk)
            print(f"{split}: {done}/{len(rows)}")
            del inputs, generated, tails
    torch.cuda.synchronize()
    write_json(manifest_path, {"fingerprint": fingerprint, "identity": identity, "finished": True,
                              "truncated_predictions": truncated, "inference_minutes": (time.perf_counter()-start)/60,
                              "peak_vram_gb": torch.cuda.max_memory_allocated()/1024**3})
    return path


def score_predictions(pred, name, out, data_dir="data", split="test", coord="norm1000", checkpoint=None):
    """Delegate evidence generation to the unchanged official scorer."""
    from scripts.score import main
    args = ["--pred", str(pred), "--name", name, "--out", str(out), "--splits-dir", str(Path(data_dir)/"splits"),
            "--split", split, "--coord", coord]
    split_meta = json.loads((Path(data_dir)/"splits/meta.json").read_text())
    if checkpoint:
        run = json.loads((Path(checkpoint) / "run.json").read_text())
        if run["dataset_version"] != split_meta["dataset_version"] or run["coord"] != coord:
            raise ValueError("Checkpoint dataset/coordinate scale differs from scoring request")
        if not run["finished"] or run["completed_steps"] != run["max_steps"]:
            raise ValueError("Incomplete training: use this checkpoint for debugging, not final evidence")
        for key in ("trainable_params", "peak_vram_gb", "train_minutes"):
            args += ["--meta", f"{key}={run[key]}"]
        args += ["--meta", f"completed_steps={run['completed_steps']}"]
    else:
        manifest = json.loads(Path(pred).with_suffix(".manifest.json").read_text())
        args += ["--meta", "trainable_params=0", "--meta", f"peak_vram_gb={manifest['peak_vram_gb']}",
                 "--meta", f"inference_minutes={manifest['inference_minutes']}"]
    main(args)
