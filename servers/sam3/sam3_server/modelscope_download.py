"""Download a model file from ModelScope (stdlib only, resumable, size-checked).

Usage::

    python -m sam3_server.modelscope_download                     # SAM3 sam3.pt
    python -m sam3_server.modelscope_download --model-id facebook/sam3 --file sam3.pt

The file lands in ``$MODELSCOPE_CACHE`` (default ``~/.cache/modelscope``) under
``models/<model-id>/<file>``; re-running resumes a partial download and is a
no-op when the file is already complete.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

SAM3_MODEL_ID = "facebook/sam3"
SAM3_CKPT_NAME = "sam3.pt"
_API = "https://modelscope.cn/api/v1/models/{model_id}/repo/files?Revision={revision}&Recursive=true"
_FILE = "https://modelscope.cn/models/{model_id}/resolve/{revision}/{name}"
_CHUNK = 8 * 1024 * 1024


def cache_path(model_id: str, name: str) -> Path:
    """Return where ``name`` from ``model_id`` is stored locally."""
    root = Path(os.environ.get("MODELSCOPE_CACHE", Path.home() / ".cache" / "modelscope"))
    return root / "models" / model_id / name


def remote_size(model_id: str, name: str, revision: str = "master") -> int | None:
    """Return the published file size in bytes, or None when the listing lacks it."""
    url = _API.format(model_id=model_id, revision=revision)
    with urllib.request.urlopen(url, timeout=60) as response:
        data = json.load(response)
    for entry in (data.get("Data") or {}).get("Files") or []:
        if entry.get("Path") == name:
            return int(entry["Size"]) if entry.get("Size") else None
    raise FileNotFoundError(f"{name} is not in ModelScope repo {model_id} ({revision})")


def download(model_id: str = SAM3_MODEL_ID, name: str = SAM3_CKPT_NAME,
             revision: str = "master", quiet: bool = False) -> Path:
    """Download (or resume) one file and return its local path."""
    target = cache_path(model_id, name)
    expected = remote_size(model_id, name, revision)
    if target.is_file() and (expected is None or target.stat().st_size == expected):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    have = partial.stat().st_size if partial.is_file() else 0
    request = urllib.request.Request(_FILE.format(model_id=model_id, revision=revision, name=name))
    if have:
        request.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(request, timeout=120) as response:
        if have and response.status != 206:  # server ignored the range: start over
            have = 0
        with partial.open("ab" if have else "wb") as stream:
            done, last = have, -1
            while chunk := response.read(_CHUNK):
                stream.write(chunk)
                done += len(chunk)
                if not quiet and expected:
                    percent = int(done * 100 / expected)
                    if percent != last:
                        print(f"\r{name}: {done / 1e9:.2f}/{expected / 1e9:.2f} GB ({percent}%)",
                              end="", file=sys.stderr, flush=True)
                        last = percent
    if not quiet:
        print(file=sys.stderr)
    size = partial.stat().st_size
    if expected is not None and size != expected:
        raise OSError(f"{name}: downloaded {size} bytes, expected {expected}; re-run to resume")
    shutil.move(partial, target)
    return target


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-id", default=SAM3_MODEL_ID)
    parser.add_argument("--file", default=SAM3_CKPT_NAME)
    parser.add_argument("--revision", default="master")
    args = parser.parse_args()
    print(download(args.model_id, args.file, args.revision))


if __name__ == "__main__":
    main()
