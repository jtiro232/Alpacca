// Alpacca — local model store (~/.alpacca/models) and model references.
// MIT License. See LICENSE.
#pragma once

#include "json.h"
#include "util.h"

#include <optional>
#include <string>
#include <vector>

namespace alpacca {

// A parsed model reference, e.g.
//   "llama3.2:1b"            → Ollama registry, library/llama3.2:1b
//   "ollama:user/model:tag"  → Ollama registry, user namespace
//   "hf:org/repo:Q4_K_M"     → Hugging Face repo, quant/file selector
//   "org/repo"               → Hugging Face (anything with a slash)
//   "./model.gguf"           → direct GGUF file path
struct ModelRef {
    enum Source { OLLAMA, HF, FILE };

    Source      source = OLLAMA;
    std::string ns;       // ollama namespace ("library") or HF org
    std::string name;     // model / repo name
    std::string tag;      // ollama tag ("latest") or HF quant/file selector ("")
    fs::path    file;     // when source == FILE

    std::string display() const;                    // canonical human name
    fs::path    store_dir(const fs::path & models_root) const;
};

ModelRef parse_model_ref(const std::string & input);

// A locally installed model, resolved and ready to run.
struct LocalModel {
    fs::path    dir;          // store directory (empty for FILE refs)
    fs::path    model_path;   // GGUF to pass to llama.cpp
    fs::path    mmproj_path;  // multimodal projector, empty if none
    JValue      manifest;     // parsed manifest.json (NUL type for FILE refs)
};

fs::path models_root();

// Look up an installed model. Returns nullopt when not installed.
std::optional<LocalModel> find_local(const ModelRef & ref);

// Write manifest.json into a model dir.
void write_manifest(const fs::path & dir, const JValue & manifest);

// All installed models (directories containing manifest.json).
struct ListedModel {
    std::string name;
    std::string source;
    uint64_t    size = 0;     // total bytes of GGUF payload(s)
    std::string modified;     // ISO-8601
    fs::path    dir;
};
std::vector<ListedModel> list_models();

// Remove an installed model; returns false when it was not installed.
bool remove_model(const ModelRef & ref);

} // namespace alpacca
