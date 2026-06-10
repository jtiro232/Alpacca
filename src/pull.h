// Alpacca — model downloads from the Ollama registry and Hugging Face.
// Downloads happen before inference starts; nothing here touches the
// llama.cpp inference path. MIT License. See LICENSE.
#pragma once

#include "store.h"

namespace alpacca {

struct PullOptions {
    bool force  = false; // re-download even if already installed
    bool verify = true;  // check sha256 digests when known
};

// Download a model into the local store and return it ready to run.
LocalModel pull_model(const ModelRef & ref, const PullOptions & opts);

} // namespace alpacca
