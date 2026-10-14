"""Assistant-only loss, validated chat-template prefix, multimodal collation."""
from pathlib import Path

from PIL import Image

from solution.data import objects_to_answer, prompt, user_message

SEQUENCE_KEYS = ("input_ids", "attention_mask", "mm_token_type_ids", "labels")
IMAGE_KEYS = ("pixel_values", "image_grid_thw")


def encode_example(processor, row, data_dir="data", coord="norm1000", max_length=2048):
    import torch

    with Image.open(Path(data_dir) / row["image_path"]) as src:
        image = src.convert("RGB")
    user = user_message(image, prompt(coord, row["width"], row["height"]))
    answer = objects_to_answer(row["objects"], coord, row["width"], row["height"])
    options = dict(tokenize=True, return_dict=True, return_tensors="pt")
    prefix = processor.apply_chat_template([user], add_generation_prompt=True, **options)
    full = processor.apply_chat_template([user, {"role": "assistant", "content": answer}],
                                         add_generation_prompt=False, **options)
    n = prefix["input_ids"].shape[1]
    if not torch.equal(full["input_ids"][:, :n], prefix["input_ids"]):
        raise RuntimeError("Chat template prefix mismatch; refusing to mask an incorrect span")
    if full["input_ids"].shape[1] > max_length:
        raise ValueError(f"{row['image_id']}: sequence exceeds max_length={max_length}; do not truncate image tokens")
    labels = full["input_ids"].clone()
    labels[:, :n] = -100
    if not (labels != -100).any():
        raise RuntimeError("Empty assistant supervision")
    # A chat template may append a newline after im_end. Do not supervise that trailing newline.
    end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    ends = (labels[0] == end_id).nonzero(as_tuple=True)[0]
    if not len(ends):
        raise RuntimeError("Missing assistant im_end; model will not learn to stop")
    labels[:, int(ends[-1])+1:] = -100
    full["labels"] = labels
    full.pop("token_type_ids", None)
    unknown = set(full) - set(SEQUENCE_KEYS) - set(IMAGE_KEYS)
    if unknown:
        raise RuntimeError(f"Review unsupported processor fields before collating: {unknown}")
    image_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    count = int((full["input_ids"] == image_id).sum())
    if (row["width"], row["height"]) == (640, 360) and count != 220:
        raise RuntimeError(f"Expected 220 image tokens for 640x360, got {count}; check resize configuration")
    return dict(full)


def collate(examples, pad_token_id):
    import torch
    from torch.nn.functional import pad

    max_len = max(e["input_ids"].shape[1] for e in examples)
    result = {}
    for key in SEQUENCE_KEYS:
        present = [key in e for e in examples]
        if not any(present):
            continue
        if not all(present):
            raise ValueError(f"Inconsistent field {key}")
        fill = pad_token_id if key == "input_ids" else -100 if key == "labels" else 0
        result[key] = torch.cat([pad(e[key], (0, max_len-e[key].shape[1]), value=fill) for e in examples])
    for key in IMAGE_KEYS:
        result[key] = torch.cat([e[key] for e in examples], dim=0)
    return result


def inspect_encoding(processor, row, data_dir="data", coord="norm1000", max_length=2048):
    encoded = encode_example(processor, row, data_dir, coord, max_length)
    labels = encoded["labels"][0]
    image_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    return {"image_id": row["image_id"], "image_tokens": int((encoded["input_ids"] == image_id).sum()),
            "prompt_tokens": int((labels == -100).sum()), "answer_tokens": int((labels != -100).sum()),
            "supervised_text": processor.tokenizer.decode(labels[labels != -100], skip_special_tokens=False)}
