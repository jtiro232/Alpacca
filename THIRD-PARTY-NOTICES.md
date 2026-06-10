# Third-party notices

Alpacca stands on the shoulders of excellent free software. This file
records what we use and on what terms — everything here is properly
licensed for this use, free of charge.

## llama.cpp (vendored, unmodified)

- What: all model loading and inference — `llama-cli`, `llama-server`,
  `llama-quantize` and the other tools Alpacca drives.
- Where: `vendor/llama.cpp` (git submodule pinned to an upstream release
  tag; the code is not modified).
- Upstream: <https://github.com/ggml-org/llama.cpp>
- License: MIT — Copyright (c) 2023-2024 The ggml authors. The full
  license text ships with the submodule at `vendor/llama.cpp/LICENSE`.

## Ollama (ideas and protocol, no code)

- What: Alpacca's model-management UX (`pull` / `run` / `list` / `rm`,
  `name:tag` references, auto-pull on first run) is modeled on Ollama, and
  `alpacca pull` speaks the public Ollama registry protocol
  (`registry.ollama.ai`) so Ollama-published models work here. The
  client in `src/pull.cpp` is an independent implementation; no Ollama
  source code is included in this repository.
- Upstream: <https://github.com/ollama/ollama>
- License: MIT — Copyright (c) Ollama Inc.

## Models

Model weights are not part of Alpacca. Each model you `alpacca pull`
carries its own license from its publisher; when a model ships a license
file, Alpacca stores it next to the weights (`license.txt`, see
`alpacca show <model>`) — review it before redistribution or commercial
use.
