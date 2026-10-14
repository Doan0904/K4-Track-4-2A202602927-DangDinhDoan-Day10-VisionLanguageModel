"""Optional GPU E5, initialized from the validation-selected SFT checkpoint."""
import json
import shutil
import time
from pathlib import Path

import yaml

from scripts.labcommon import read_jsonl
from scripts.score import _gt_boxes
from solution.config import write_json
from solution.data import audit_data, prompt
from solution.modeling import load_checkpoint, require_gpu, save_weights
from solution.reward import grpo_reward
from solution.training import package_versions


def train_grpo(checkpoint, data_dir, out_dir, grpo_config="configs/e5.yaml", backup_dir=None):
    torch = require_gpu()
    from datasets import Dataset, Image as DatasetImage
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    checkpoint, output = Path(checkpoint), Path(out_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    cfg = json.loads((checkpoint / "config.json").read_text())
    sft = json.loads((checkpoint / "run.json").read_text())
    if not sft["finished"] or cfg["coord"] != "norm1000":
        raise ValueError("E5 must start from a completed norm1000 SFT checkpoint")
    stats = audit_data(data_dir)
    if sft["dataset_version"] != stats["dataset_version"]:
        raise ValueError("E5 dataset differs from SFT")
    gc = yaml.safe_load(Path(grpo_config).read_text())["grpo"]
    cfg.update(name="E5", max_steps=gc["max_steps"], lr=gc["learning_rate"],
               source_checkpoint=str(checkpoint), grpo=gc)
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    torch.cuda.reset_peak_memory_stats()
    model, processor = load_checkpoint(cfg, checkpoint, trainable=True)
    processor.tokenizer.padding_side = "left"
    rows = read_jsonl(Path(data_dir) / "splits/train.jsonl")
    dataset = Dataset.from_list([{
        "prompt": [{"role": "user", "content": prompt("norm1000", r["width"], r["height"])}],
        "image": str((Path(data_dir) / r["image_path"]).resolve()), "gt_boxes": _gt_boxes(r),
    } for r in rows]).cast_column("image", DatasetImage())
    # One training sample at a time; generation group contains four completions.
    args = GRPOConfig(
        output_dir=str(output / "trainer"), per_device_train_batch_size=1,
        gradient_accumulation_steps=4, generation_batch_size=4, num_generations=gc["num_generations"],
        max_completion_length=gc["max_completion_length"], max_steps=gc["max_steps"],
        learning_rate=gc["learning_rate"], beta=gc["beta"], use_vllm=False, fp16=True, bf16=False,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False, report_to="none", logging_steps=1, save_steps=20,
        seed=cfg["seed"], optim="adamw_torch", max_grad_norm=1.0)
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)

    class SaveProjector(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            # Trainer's PEFT checkpoints alone omit the fully trained projector.
            save_weights(model, processor, output / f"checkpoint-{state.global_step:04d}", cfg, info(state.global_step))
            backup()

    def info(step, finished=False):
        return {"name": "E5", "dataset_version": stats["dataset_version"],
                "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "peak_vram_gb": torch.cuda.max_memory_allocated()/1024**3,
                "train_minutes": (time.perf_counter()-started)/60,
                "completed_steps": step, "max_steps": gc["max_steps"], "finished": finished,
                "architecture": sft["architecture"], "setup": sft["setup"], "versions": package_versions(),
                "source_sft": str(checkpoint), "coord": "norm1000"}

    def backup():
        if backup_dir:
            shutil.copytree(output, Path(backup_dir)/output.name, dirs_exist_ok=True)

    trainer = GRPOTrainer(model=model, args=args, train_dataset=dataset,
                          processing_class=processor, reward_funcs=grpo_reward, callbacks=[SaveProjector()])
    # TRL 1.9.2 casts all QLoRA trainables to bf16 during construction even with fp16=True.
    # Restore fp32 master weights before AdamW is created, for T4 + fp16 GradScaler.
    for param in trainer.model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    trainer.train()
    torch.cuda.synchronize()
    run = info(trainer.state.global_step, finished=True)
    final = output / "final"
    save_weights(model, processor, final, cfg, run)
    write_json(output / "run.json", run)
    write_json(output / "grpo_log.json", trainer.state.log_history)
    write_json(output / "latest.json", {"checkpoint": "final"})
    backup()
    return final
