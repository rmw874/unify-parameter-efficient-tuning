"""
Prepare and verify the input files and software used experimental runs.

Usage (from repo root):
    python exps/table2_mam/verify_setup.py prepare  # download and verify once
    python exps/table2_mam/verify_setup.py check    # read-only checks

"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(os.environ.get(
    "MODEL", ROOT / "checkpoints" / "table2_mam" / "pretrained_roberta_base"
)).expanduser().resolve()

# raw pretrained RoBERTa-base files used in both UCloud environments
EXPECTED_MODEL_SHA256 = {
    "pytorch_model.bin": "278b7a95739c4392fae9b818bb5343dde20be1b89318f37a6d939e1e1b9e461b",
    "config.json": "3fbb0eb8d123b8543387c7256bf3a4706f138ff340d5e577fe97daacaf6f000e",
    "vocab.json": "ed19656ea1707df69134c4af35c8ceda2cc9860bf2c3495026153a133670ab5e",
    "merges.txt": "fe36cab26d4f4421ed725e10a2e9ddb7f799449c603a96e7f29b5a3c82a95862",
}
EXPECTED_DATASET = {
    "train": {"n": 67349, "label_counts": [29780, 37569],
              "sha256_content": "9fa2f2c21fe065a8aa3dfc706b6c802d2cffa7309ed57ed22d0371b6bc59aa21"},
    "validation": {"n": 872, "label_counts": [428, 444],
                   "sha256_content": "dd5ac3ff6d6b82764596848824822006276abcfb7401d6c44078d01f852cfce4"},
}
EXPECTED_VERSIONS = {
    "python": "3.9.25",
    "torch": "2.7.0+cu128",
    "cuda_runtime": "12.8",
    "transformers": "4.9.0.dev0",
    "datasets": "1.11.0",
    "numpy": "1.23.5",
    "tokenizers": "0.10.3",
    "huggingface-hub": "0.0.12",
    "pyarrow": "10.0.1",
    "fsspec": "2022.11.0",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_versions():
    import torch
    import transformers
    import datasets
    import numpy

    got = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "numpy": numpy.__version__,
    }
    for package in ("tokenizers", "huggingface-hub", "pyarrow", "fsspec"):
        got[package] = importlib.metadata.version(package)
    for name, expected in EXPECTED_VERSIONS.items():
        actual = got[name]
        if actual != expected:
            raise RuntimeError(f"Wrong {name}: expected {expected}, found {actual}")

    expected_path = (ROOT / "src/transformers/__init__.py").resolve()
    actual_path = Path(transformers.__file__).resolve()
    if actual_path != expected_path:
        raise RuntimeError(
            f"Wrong Transformers installation: {actual_path}. "
            f"Expected local authors' fork: {expected_path}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; use a GPU job with the CUDA PyTorch build")
    print("Software versions match the recorded UCloud environment.")


def check_model():
    for filename, expected in EXPECTED_MODEL_SHA256.items():
        path = MODEL_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; run 'prepare' first")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Wrong pretrained model bytes: {path}\n"
                               f"Expected {expected}\nFound    {actual}")
    print(f"Pretrained RoBERTa files match the verified UCloud snapshot: {MODEL_DIR}")


def prepare_model():
    if all((MODEL_DIR / f).is_file() for f in EXPECTED_MODEL_SHA256):
        return
    if MODEL_DIR.exists() and any(MODEL_DIR.iterdir()):
        raise RuntimeError(f"Refusing to overwrite incomplete/nonempty model directory: {MODEL_DIR}")
    from transformers import AutoConfig, AutoTokenizer
    from transformers.file_utils import WEIGHTS_NAME, cached_path, hf_bucket_url

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config = AutoConfig.from_pretrained("roberta-base")
    if hasattr(config, "attn_mode") or hasattr(config, "ffn_mode"):
        raise RuntimeError("Expected a raw RoBERTa config, not a tuned configuration")
    config.save_pretrained(MODEL_DIR)
    AutoTokenizer.from_pretrained("roberta-base", use_fast=True).save_pretrained(MODEL_DIR)
    weights = cached_path(hf_bucket_url("roberta-base", WEIGHTS_NAME))
    shutil.copyfile(weights, MODEL_DIR / WEIGHTS_NAME)

    import torch
    state = torch.load(MODEL_DIR / WEIGHTS_NAME, map_location="cpu", weights_only=True)
    if any("classifier." in name for name in state):
        raise RuntimeError("Not raw pretrained weights: found a classifier head")
    del state


def check_dataset():
    from datasets import load_dataset, load_metric

    data = load_dataset("glue", "sst2")
    for split, expected in EXPECTED_DATASET.items():
        digest = hashlib.sha256()
        counts = [0, 0]
        rows = data[split]
        for row in rows:
            label = int(row["label"])
            if label not in (0, 1):
                raise RuntimeError(f"Unexpected SST-2 label: {label}")
            counts[label] += 1
            record = [int(row["idx"]), row["sentence"], label]
            encoded = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            digest.update(encoded.encode("utf-8"))
        observed = {"n": len(rows), "label_counts": counts,
                    "sha256_content": digest.hexdigest()}
        if observed != expected:
            raise RuntimeError(
                f"SST-2 {split} differs from UCloud:\n"
                f"Expected: {expected}\nActual:   {observed}"
            )
        print(f"SST-2 {split}: {len(rows)} rows, verified content hash")

    load_metric("glue", "sst2")


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("prepare", "check"):
        raise SystemExit("Usage: python verify_setup.py {prepare|check}")
    os.environ["HF_SCRIPTS_VERSION"] = "1.11.0"
    check_versions()
    if sys.argv[1] == "prepare":
        prepare_model()
    check_model()
    check_dataset()
    print("Input and environment verification PASSED.")


if __name__ == "__main__":
    main()
