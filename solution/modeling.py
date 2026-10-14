"""GPU model lifecycle. All heavy imports are inside explicitly called functions."""
import math
import re
from pathlib import Path

from solution.config import write_json

LLM_TARGET = re.compile(r"model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)")
VIT_TARGET = re.compile(r"model\.visual\.blocks\.\d+\.(?:attn\.(?:qkv|proj)|mlp\.(?:linear_fc1|linear_fc2))")


def is_projector(name):
    return "visual.merger." in name or "visual.deepstack_merger_list." in name


def require_gpu():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install requirements-colab.txt in Colab/Kaggle first") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("This command requires CUDA. Enable GPU in Colab/Kaggle; no CPU fallback.")
    return torch


def load_base(cfg):
    torch = require_gpu()
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True, llm_int8_skip_modules=["model.visual", "lm_head"])
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        cfg["model_id"], revision=cfg["revision"], quantization_config=quantization,
        dtype=torch.float16, device_map={"": 0}, attn_implementation="sdpa")
    # Use the resolved model commit for processor too, avoiding moving-main drift.
    resolved = getattr(model.config, "_commit_hash", None) or cfg["revision"]
    processor = AutoProcessor.from_pretrained(cfg["model_id"], revision=resolved)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    # Bound image processing to a small, fixed budget. A 640x360 frame gives 220 tokens.
    processor.image_processor.size = {"shortest_edge": 65536, "longest_edge": 256*32*32}
    bad = [n for n, m in model.named_modules()
           if (".visual." in n or n == "lm_head") and m.__class__.__name__ in ("Linear4bit", "Linear8bitLt")]
    if bad:
        raise RuntimeError(f"ViT/lm_head were quantized unexpectedly: {bad[:5]}")
    model.config.use_cache = True
    return model, processor


def _numel(param):
    state = getattr(param, "quant_state", None)
    return math.prod(state.shape) if state is not None else param.numel()


def architecture_counts(model):
    result = {"vit_backbone": 0, "projector": 0, "llm_including_embeddings": 0, "other": 0}
    for name, param in model.named_parameters():
        key = ("projector" if is_projector(name) else "vit_backbone" if ".visual." in name
               else "llm_including_embeddings" if ".language_model." in name or "lm_head" in name else "other")
        result[key] += _numel(param)
    result["total"] = sum(result.values())
    return result


def apply_setup(model, cfg):
    import torch
    from peft import LoraConfig, get_peft_model

    for param in model.parameters():
        param.requires_grad_(False)
    llm = [n for n, _ in model.named_modules() if LLM_TARGET.fullmatch(n)]
    vit = [n for n, _ in model.named_modules() if VIT_TARGET.fullmatch(n)]
    if len(llm) != 252 or len(vit) != 96:
        raise RuntimeError(f"Unexpected architecture: LLM targets={len(llm)}, ViT targets={len(vit)}")
    targets = llm + (vit if cfg["lora"]["targets"] == "llm+vit" else [])
    # Exact full paths avoid accidentally matching vision o_proj/proj modules in E1.
    target_regex = "(?:" + "|".join(re.escape(n) for n in targets) + ")"
    lc = cfg["lora"]
    model = get_peft_model(model, LoraConfig(
        r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
        bias="none", target_modules=target_regex, task_type="CAUSAL_LM"))
    if cfg["train_projector"]:
        for name, param in model.named_parameters():
            if is_projector(name):
                if not param.is_floating_point():
                    raise RuntimeError(f"Cannot fully train a quantized projector: {name}")
                param.data = param.data.float()
                param.requires_grad_(True)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    trainable = {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}
    unexpected = [n for n in trainable if "lora_" not in n and not (cfg["train_projector"] and is_projector(n))]
    if unexpected:
        raise RuntimeError(f"Unexpected trainable parameters: {unexpected[:5]}")
    if cfg["train_projector"] and any(p.dtype != torch.float32 for n, p in model.named_parameters() if is_projector(n)):
        raise RuntimeError("Projector must have float32 master weights for fp16 GradScaler")
    return model, {"matched_llm_modules": len(llm), "matched_vit_modules": len(targets)-len(llm),
                   "trainable_params": sum(trainable.values()), "trainable_by_name": trainable}


def save_weights(model, processor, directory, cfg, run):
    from safetensors.torch import save_file

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory / "adapter", safe_serialization=True)
    processor.save_pretrained(directory / "processor")
    if cfg["train_projector"]:
        tensors = {n: p.detach().float().cpu().contiguous()
                   for n, p in model.named_parameters() if is_projector(n)}
        if not tensors:
            raise RuntimeError("Projector config enabled but no weights found")
        save_file(tensors, str(directory / "projector.safetensors"))
    write_json(directory / "config.json", cfg)
    write_json(directory / "run.json", run)


def load_checkpoint(cfg, directory, trainable=False):
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoProcessor

    directory = Path(directory)
    model, _ = load_base(cfg)
    model = PeftModel.from_pretrained(model, directory / "adapter", is_trainable=trainable)
    processor = AutoProcessor.from_pretrained(directory / "processor")
    processor.tokenizer.padding_side = "left"
    if cfg["train_projector"]:
        path = directory / "projector.safetensors"
        if not path.is_file():
            raise FileNotFoundError(f"Missing trained projector: {path}")
        saved = load_file(str(path), device="cpu")
        parameters = dict(model.named_parameters())
        expected = {n for n in parameters if is_projector(n)}
        if set(saved) != expected:
            raise RuntimeError("Saved projector names do not match model")
        with torch.no_grad():
            for name, tensor in saved.items():
                param = parameters[name]
                param.data = param.data.float()
                param.copy_(tensor.to(param.device))
                param.requires_grad_(trainable)
    if trainable:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = not trainable
    return model, processor
