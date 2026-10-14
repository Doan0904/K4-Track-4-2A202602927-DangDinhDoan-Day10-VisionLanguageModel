"""Explicit SFT entry point; never called on import or by the CPU checks."""
import json
import random
import shutil
import time
from pathlib import Path

from scripts.labcommon import read_jsonl
from solution.config import write_json
from solution.data import BatchStream, audit_data, subset_train
from solution.encoding import collate, encode_example, inspect_encoding
from solution.modeling import (apply_setup, architecture_counts, load_base, load_checkpoint,
                               require_gpu, save_weights)


def package_versions():
    from importlib.metadata import PackageNotFoundError, version
    result = {}
    for name in ("torch", "torchvision", "transformers", "peft", "accelerate", "bitsandbytes", "trl"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def train(cfg, data_dir, out_dir, backup_dir=None, resume=None):
    torch = require_gpu()
    from transformers import get_cosine_schedule_with_warmup

    if cfg["name"] == "E0" or "lora" not in cfg:
        raise ValueError("SFT requires an E1/E2/E3 or resolved E4 config")
    output = Path(out_dir)
    if output.exists() and any(output.iterdir()) and resume is None:
        raise FileExistsError(f"{output} already contains a run; choose another directory or --resume")
    output.mkdir(parents=True, exist_ok=True)
    stats = audit_data(data_dir)
    all_rows = read_jsonl(Path(data_dir) / "splits/train.jsonl")
    rows = subset_train(all_rows, cfg["train_fraction"], cfg["seed"])
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    torch.cuda.reset_peak_memory_stats()
    if resume:
        previous_cfg = json.loads((Path(resume) / "config.json").read_text())
        previous = json.loads((Path(resume) / "run.json").read_text())
        # main may have resolved to an immutable commit during the first run.
        if cfg["revision"] == "main":
            cfg["revision"] = previous_cfg["revision"]
        if cfg != previous_cfg or stats["dataset_version"] != previous["dataset_version"]:
            raise ValueError("Resume config/dataset differs from the checkpoint")
        model, processor = load_checkpoint(cfg, resume, trainable=True)
        counts, setup = previous["architecture"], previous["setup"]
    else:
        model, processor = load_base(cfg)
        cfg["revision"] = getattr(model.config, "_commit_hash", None) or cfg["revision"]
        counts = architecture_counts(model)
        model, setup = apply_setup(model, cfg)
        previous = {}
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, round(cfg["warmup_ratio"]*cfg["max_steps"])),
        num_training_steps=cfg["max_steps"])
    scaler = torch.amp.GradScaler("cuda")
    stream = BatchStream(len(rows), cfg["batch_size"], cfg["seed"])
    step = 0
    if resume:
        # Only load a trusted checkpoint created by this code (optimizer/RNG includes Python objects).
        state = torch.load(Path(resume) / "training_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        stream.load_state_dict(state["stream"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        random.setstate(state["python_rng"])
        step = state["step"]
    check = inspect_encoding(processor, rows[0], data_dir, cfg["coord"], cfg["max_length"])
    write_json(output / "encoding_check.json", check)
    write_json(output / "data_audit.json", stats)
    write_json(output / "train_ids.json", [r["image_id"] for r in rows])
    print(json.dumps({"train_images": len(rows), "trainable_params": sum(p.numel() for p in params),
                      "image_tokens": check["image_tokens"], "supervised_text": check["supervised_text"]}, ensure_ascii=False))
    model.train()
    optimizer.zero_grad(set_to_none=True)
    start = time.perf_counter()
    previous_minutes = previous.get("train_minutes", 0)
    previous_peak = previous.get("peak_vram_gb", 0)
    log_path = output / "train_log.jsonl"
    # fp16 dynamic loss scaling: a few overflowed (inf) steps at the start are expected; the scaler halves
    # its scale and the step is retried on fresh data. Skipped steps are not counted as optimizer steps.
    skipped_total = previous.get("skipped_steps", 0)
    skipped_run = 0
    max_consecutive_skips = 25

    def run_info(finished=False):
        return {"name": cfg["name"], "dataset_version": stats["dataset_version"],
                "trainable_params": sum(p.numel() for p in params),
                "peak_vram_gb": max(previous_peak, torch.cuda.max_memory_allocated()/1024**3),
                "train_minutes": previous_minutes+(time.perf_counter()-start)/60,
                "completed_steps": step, "max_steps": cfg["max_steps"], "finished": finished,
                "train_images": len(rows), "architecture": counts, "setup": setup,
                "versions": package_versions(), "gpu": torch.cuda.get_device_name(0),
                "coord": cfg["coord"], "effective_batch_size": cfg["batch_size"]*cfg["grad_accum"],
                "skipped_steps": skipped_total}

    def checkpoint(finished=False):
        torch.cuda.synchronize()
        run = run_info(finished)
        destination = output / f"checkpoint-{step:04d}"
        if destination.exists():
            raise FileExistsError(destination)
        temporary = output / f".checkpoint-{step:04d}.tmp"
        if temporary.exists():
            shutil.rmtree(temporary)
        save_weights(model, processor, temporary, cfg, run)
        torch.save({"step": step, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(), "stream": stream.state_dict(),
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                    "python_rng": random.getstate()}, temporary / "training_state.pt")
        temporary.rename(destination)
        write_json(output / "run.json", run)
        write_json(output / "latest.json", {"checkpoint": destination.name})
        if backup_dir:
            backup = Path(backup_dir) / output.name
            backup.mkdir(parents=True, exist_ok=True)
            shutil.copytree(destination, backup / destination.name, dirs_exist_ok=True)
            for file in ("run.json", "latest.json", "train_log.jsonl", "encoding_check.json", "train_ids.json", "data_audit.json"):
                if (output / file).exists():
                    shutil.copy2(output / file, backup / file)
        print(f"Saved {destination}")

    try:
        with log_path.open("a", encoding="utf-8") as log:
            while step < cfg["max_steps"]:
                loss_sum = 0.0
                for _ in range(cfg["grad_accum"]):
                    batch = collate([encode_example(processor, rows[i], data_dir, cfg["coord"], cfg["max_length"])
                                     for i in stream.next_indices()], processor.tokenizer.pad_token_id)
                    batch = {k: v.to("cuda") for k, v in batch.items()}
                    with torch.autocast("cuda", dtype=torch.float16):
                        loss = model(**batch).loss
                    if not torch.isfinite(loss).item():
                        raise FloatingPointError(f"Nonfinite loss at optimizer step {step+1}")
                    loss_sum += loss.detach().float().item()/cfg["grad_accum"]
                    scaler.scale(loss/cfg["grad_accum"]).backward()
                    del batch, loss
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
                if not torch.isfinite(norm).item():
                    bad = [n for n, p in model.named_parameters()
                           if p.grad is not None and not torch.isfinite(p.grad).all().item()][:5]
                    scaler.step(optimizer)      # found_inf recorded by unscale_: the optimizer step is skipped
                    scaler.update()             # ... and the loss scale is reduced
                    optimizer.zero_grad(set_to_none=True)
                    skipped_total += 1
                    skipped_run += 1
                    log.write(json.dumps({"step": step, "skipped": True, "loss": loss_sum,
                                          "scale": scaler.get_scale(), "nonfinite_params": bad})+"\n")
                    log.flush()
                    print(f"skip (nonfinite grads) at step {step+1}: scale->{scaler.get_scale():g}, e.g. {bad[:2]}")
                    if skipped_run > max_consecutive_skips or scaler.get_scale() < 1:
                        raise FloatingPointError(
                            f"Nonfinite gradients persist after {skipped_run} skipped steps "
                            f"(scale={scaler.get_scale():g}); parameters: {bad}")
                    continue
                skipped_run = 0
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                item = {"step": step, "loss": loss_sum, "lr": scheduler.get_last_lr()[0],
                        "grad_norm": norm.item(), "peak_vram_gb": torch.cuda.max_memory_allocated()/1024**3}
                log.write(json.dumps(item)+"\n")
                log.flush()
                if step == 1 or step % 10 == 0:
                    print(f"step {step}/{cfg['max_steps']} loss={loss_sum:.4f} VRAM={item['peak_vram_gb']:.2f} GB")
                if step % cfg["save_steps"] == 0 or step == cfg["max_steps"]:
                    checkpoint(finished=step == cfg["max_steps"])
    except BaseException as exc:
        write_json(output / "failure.json", {"type": type(exc).__name__, "message": str(exc), "completed_steps": step})
        raise
    return output / f"checkpoint-{step:04d}"
