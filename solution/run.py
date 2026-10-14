"""CLI: python -m solution.run --help. GPU commands require --allow-gpu."""
import argparse
import json
from pathlib import Path

from solution.config import create_e4, load_config, write_json


def latest_checkpoint(directory):
    directory = Path(directory)
    if (directory / "config.json").is_file():
        return directory
    latest = json.loads((directory / "latest.json").read_text())
    result = directory / latest["checkpoint"]
    if not result.is_dir():
        raise FileNotFoundError(result)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    for command in ("train", "predict", "grpo"):
        p = sub.add_parser(command)
        p.add_argument("--allow-gpu", action="store_true", help="Explicitly enable model loading/training")
        p.add_argument("--data-dir", default="data")
        if command != "grpo":
            p.add_argument("--config", default="configs/e0.yaml" if command == "predict" else "configs/e1.yaml")
            p.add_argument("--revision", help="Pin an immutable Hugging Face model commit")
        if command in ("train", "grpo"):
            p.add_argument("--out", required=True)
            p.add_argument("--backup-dir")
        if command == "train":
            p.add_argument("--resume", help="Trusted checkpoint directory")
            p.add_argument("--batch-size", type=int)
            p.add_argument("--grad-accum", type=int)
            p.add_argument("--max-steps", type=int)
        elif command == "predict":
            p.add_argument("--checkpoint", help="Run directory or checkpoint; omit for E0")
            p.add_argument("--split", default="test", choices=("val", "test"))
            p.add_argument("--pred", required=True)
            p.add_argument("--batch-size", type=int, default=4)
            p.add_argument("--limit", type=int)
            p.add_argument("--resume", action="store_true")
            p.add_argument("--max-new-tokens", type=int)
            p.add_argument("--score-out")
            p.add_argument("--name")
        else:
            p.add_argument("--checkpoint", required=True)
    p = sub.add_parser("audit")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="report/data_audit.json")
    p = sub.add_parser("e4-configs")
    p.add_argument("--base-config", required=True)
    p.add_argument("--out", default="work/configs")
    p = sub.add_parser("select-e4")
    p.add_argument("--runs-dir", default="work")
    p.add_argument("--out", default="work/configs")
    p = sub.add_parser("score")
    p.add_argument("--pred", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--split", default="test", choices=("val", "test"))
    p.add_argument("--coord", default="norm1000", choices=("norm1000", "pixel"))
    args = ap.parse_args(argv)
    if args.command in ("train", "predict", "grpo") and not args.allow_gpu:
        ap.error("GPU execution is disabled. Run this yourself in Colab/Kaggle with --allow-gpu.")
    if args.command == "audit":
        from solution.data import audit_data
        result = audit_data(args.data_dir)
        write_json(args.out, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "e4-configs":
        print(*create_e4(args.base_config, args.out), sep="\n")
    elif args.command == "select-e4":
        candidates = []
        for n in (1, 2, 3):
            key = f"e{n}"
            result = json.loads((Path(args.runs_dir) / f"{key}_val_score.json").read_text())
            if result["split"] != "val" or result["missing_predictions"]:
                raise ValueError("Select E4 baseline using complete validation predictions")
            checkpoint = latest_checkpoint(Path(args.runs_dir)/key)
            cfg = json.loads((checkpoint/"config.json").read_text())
            run = json.loads((checkpoint/"run.json").read_text())
            if (not run["finished"] or run["dataset_version"] != result["dataset_version"]
                    or result["coord"] != cfg["coord"]):
                raise ValueError("Validation score/checkpoint mismatch")
            candidates.append((result, key, checkpoint))
        if len({r["dataset_version"] for r, _, _ in candidates}) != 1:
            raise ValueError("Validation datasets differ")
        if len({r["n_images"] for r, _, _ in candidates}) != 1:
            raise ValueError("Validation set sizes differ")
        score, key, checkpoint = max(candidates, key=lambda x: x[0]["micro"]["f1"])
        paths = create_e4(checkpoint / "config.json", args.out)
        selection = {"base": key, "checkpoint": str(checkpoint), "validation_f1": score["micro"]["f1"],
                     "dataset_version": score["dataset_version"],
                     "configs": [str(p) for p in paths], "selection_split": "val"}
        write_json(Path(args.out)/"selection.json", selection)
        print(json.dumps(selection, indent=2))
    elif args.command == "train":
        from solution.training import train
        cfg = load_config(args.config, batch_size=args.batch_size, grad_accum=args.grad_accum,
                          max_steps=args.max_steps, revision=args.revision)
        print(train(cfg, args.data_dir, args.out, args.backup_dir, args.resume))
    elif args.command == "predict":
        from solution.inference import predict, score_predictions
        if args.limit and args.score_out:
            raise ValueError("Do not score a partial debug run as a complete validation split")
        checkpoint = latest_checkpoint(args.checkpoint) if args.checkpoint else None
        cfg = load_config(checkpoint/"config.json" if checkpoint else args.config,
                          max_new_tokens=args.max_new_tokens, revision=args.revision)
        if checkpoint is None and cfg["name"] != "E0":
            raise ValueError("A trained experiment requires --checkpoint")
        predict(cfg, args.data_dir, args.split, args.pred, checkpoint, args.batch_size, args.limit, args.resume)
        if args.score_out:
            score_predictions(args.pred, args.name or cfg["name"], args.score_out, args.data_dir,
                              args.split, cfg["coord"], checkpoint)
    elif args.command == "grpo":
        from solution.grpo import train_grpo
        print(train_grpo(latest_checkpoint(args.checkpoint), args.data_dir, args.out, backup_dir=args.backup_dir))
    elif args.command == "score":
        from solution.inference import score_predictions
        checkpoint = latest_checkpoint(args.checkpoint) if args.checkpoint else None
        score_predictions(args.pred, args.name, args.out, args.data_dir, args.split, args.coord, checkpoint)


if __name__ == "__main__":
    main()
