# SAM3 segmentation server

The `inspect_capx` backend (SO-101 CaP-X models such as `claude-opus-5-5-capx`,
`gpt-6-astra-capx` and `grok-4.7-capx`) calls this HTTP server for SAM3 segmentation. See
[Inspect CaP-X setup](../../docs/inspect-capx.md). The code is copied unchanged
from Inspect Robots `servers/sam3` (commit `0bbdf73`), which is itself adapted
from [CaP-X](https://github.com/capgym/cap-x) (MIT, see `LICENSE-CAPX`).

SAM3 needs its own environment, separate from the Policy Server's `.venv`,
because it pins its own `torch` and SAM3 model code. It also needs a GPU.

## Install

Run these commands from the repository root:

```bash
uv venv servers/sam3/.venv --python 3.10
uv pip install --python servers/sam3/.venv/bin/python -e servers/sam3
```

Installing `sam3` and downloading its weights means accepting Meta's SAM
License.

## Run

```bash
servers/sam3/.venv/bin/python -m sam3_server --port 8114
```

The server is ready when it logs `Uvicorn running on http://127.0.0.1:8114`.
Keep the terminal open while the Policy Server runs. Port 8114 is the default
`capx.sam3_url` in the runtime configuration.

On its first start the server downloads `sam3.pt` (3.45 GB) from
[ModelScope](https://modelscope.cn/models/facebook/sam3) into
`~/.cache/modelscope`. Set `MODELSCOPE_CACHE` to use another directory. No
Hugging Face access request is needed. Other weight sources:

- `--source hf` downloads from the gated `facebook/sam3` Hugging Face
  repository. Request access first, then run `hf auth login`.
- `--checkpoint /path/to/sam3.pt` loads a local file.
