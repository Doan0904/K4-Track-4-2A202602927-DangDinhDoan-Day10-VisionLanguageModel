"""Create source/results ZIPs without data images, secrets or model weights."""
import argparse
import zipfile
from pathlib import Path

FORBIDDEN = {".git", ".env", ".venv", "__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
WEIGHTS = {".pt", ".bin", ".ckpt", ".safetensors", ".pyc"}


def package(kind, out, root="."):
    root = Path(root).resolve()
    out = Path(out).resolve()
    if kind == "source":
        paths = [root/n for n in (".gitignore", "README.md", "GUIDE.md", "RUBRIC.md", "REPORT.md", "HUONG_DAN_CHAY.md",
                                    "lab_day10.ipynb", "requirements-colab.txt", "requirements-cpu.txt", "pytest.ini",
                                    "configs", "scripts", "tests", "solution", "report/LOCAL_CHECKS.md")]
    elif kind == "results":
        paths = [root/n for n in ("REPORT.md", "report", "work", "data/splits")]
    else:
        raise ValueError(kind)
    files = set()
    for path in paths:
        for f in path.rglob("*") if path.is_dir() else [path]:
            if not f.is_file() or f.resolve() == out:
                continue
            rel = f.relative_to(root)
            if any(p in FORBIDDEN for p in rel.parts) or f.suffix in WEIGHTS or f.suffix == ".zip":
                continue
            if kind == "results" and rel.parts[0] == "work" and f.suffix not in (".json", ".jsonl", ".yaml"):
                continue
            files.add(f)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for f in sorted(files):
            archive.write(f, f.relative_to(root))
    print(f"{out}: {len(files)} files, {out.stat().st_size:,} bytes")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=("source", "results"))
    p.add_argument("--out", required=True)
    args = p.parse_args()
    package(args.kind, args.out)


if __name__ == "__main__":
    main()
