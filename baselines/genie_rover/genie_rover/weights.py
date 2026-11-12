"""The authors' SAM-TP checkpoint.

The GENIE-SAMTP README points to a Google Drive folder; the Hugging Face repo ``jamiewjm/sam-tp`` holds the
training dataset, not the weights. The file is downloaded once and checked against its SHA-256.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

DRIVE_FILE_ID = "18TB6FiUOTJgOBOTEfbbJdZYMKRRhyTwq"  # "Sam2 Traversability check point/checkpoint_2.pt"
SHA256 = "2607fd6049d37f17fe96132cf35459f7e0a895107632410637d812756e3f9adb"
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "weights" / "checkpoint_2.pt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_path(explicit: str | None = None) -> Path:
    """The checkpoint to load: ``explicit``, ``GENIE_ROVER_CHECKPOINT``, or the downloaded default."""
    path = Path(explicit or os.environ.get("GENIE_ROVER_CHECKPOINT") or DEFAULT_PATH)
    if not path.exists():
        raise FileNotFoundError(f"SAM-TP checkpoint not found at {path}; run genie-rover-weights to download it")
    return path


def download(path: Path = DEFAULT_PATH) -> Path:
    import gdown

    if path.exists() and sha256(path) == SHA256:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(id=DRIVE_FILE_ID, output=str(path), quiet=False)
    actual = sha256(path)
    if actual != SHA256:
        raise RuntimeError(f"Downloaded checkpoint has SHA-256 {actual}, expected {SHA256}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download GeNIE's SAM-TP checkpoint (409 MB)")
    parser.add_argument("--output", type=Path, default=DEFAULT_PATH)
    print(f"SAM-TP checkpoint ready: {download(parser.parse_args().output)}")


if __name__ == "__main__":
    main()
