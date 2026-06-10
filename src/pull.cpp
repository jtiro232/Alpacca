// Alpacca — model downloads from the Ollama registry and Hugging Face.
//
// Registry protocol notes:
//  * Ollama models are served from an OCI-style registry
//    (https://registry.ollama.ai). A manifest lists content-addressed
//    layers; the GGUF weights layer has media type
//    "application/vnd.ollama.image.model". This client speaks plain HTTPS
//    and is an independent implementation of that protocol.
//  * Hugging Face models are plain files under
//    https://huggingface.co/<org>/<repo>/resolve/main/<file>, listed via
//    the /api/models/<org>/<repo>/tree endpoint.
//
// Transfers are delegated to the `curl` binary (resume + progress bar);
// integrity is verified with SHA-256 when the source publishes digests.
// MIT License. See LICENSE.
#include "pull.h"

#include "sha256.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <regex>

namespace alpacca {

static std::string ollama_registry() {
    return env_or("ALPACCA_OLLAMA_REGISTRY", "https://registry.ollama.ai");
}

static std::string hf_endpoint() {
    return env_or("ALPACCA_HF_ENDPOINT", "https://huggingface.co");
}

static std::string hf_token() {
    std::string t = env_or("HF_TOKEN", "");
    if (t.empty()) {
        t = env_or("HUGGING_FACE_HUB_TOKEN", "");
    }
    return t;
}

// ---- curl helpers -------------------------------------------------------

// Headers are passed through a stdin config file so secrets never show up
// in the process list.
static std::string curl_header_config(const std::vector<std::string> & headers) {
    std::string cfg;
    for (const auto & h : headers) {
        std::string escaped;
        for (char c : h) {
            if (c == '"' || c == '\\') {
                escaped.push_back('\\');
            }
            escaped.push_back(c);
        }
        cfg += "header = \"" + escaped + "\"\n";
    }
    return cfg;
}

static std::vector<std::string> curl_base(const std::vector<std::string> & headers) {
    std::vector<std::string> args = {
        "curl", "--location", "--fail", "--silent", "--show-error",
        "--connect-timeout", "15", "--retry", "3", "--retry-delay", "2",
    };
    if (!headers.empty()) {
        args.push_back("--config");
        args.push_back("-");
    }
    return args;
}

static std::string http_get(const std::string & url, const std::vector<std::string> & headers) {
    auto args = curl_base(headers);
    args.push_back(url);
    std::string body;
    std::string cfg = curl_header_config(headers);
    int rc = run_process(args, headers.empty() ? nullptr : &cfg, &body);
    if (rc != 0) {
        fail("HTTP request failed (curl exit " + std::to_string(rc) + "): " + url);
    }
    return body;
}

static JValue http_get_json(const std::string & url, const std::vector<std::string> & headers) {
    std::string body = http_get(url, headers);
    JValue v;
    std::string err;
    if (!json_parse(body, v, err)) {
        fail("invalid JSON from " + url + ": " + err);
    }
    return v;
}

// Download `url` to `dest`, resuming a previous partial transfer when
// possible. `expected_size` of 0 means unknown; `digest_hex` empty means
// no integrity check is possible.
static void download_file(const std::string & url, const fs::path & dest,
                          uint64_t expected_size, const std::string & digest_hex,
                          const std::vector<std::string> & headers, bool verify) {
    std::error_code ec;

    auto verified_ok = [&](const fs::path & p) {
        if (!fs::exists(p, ec)) {
            return false;
        }
        if (expected_size > 0 && fs::file_size(p, ec) != expected_size) {
            return false;
        }
        if (verify && !digest_hex.empty()) {
            fprintf(stderr, "verifying sha256... ");
            fflush(stderr);
            bool ok = SHA256::file_hex(p.string()) == digest_hex;
            fprintf(stderr, ok ? "ok\n" : "MISMATCH\n");
            return ok;
        }
        return true;
    };

    if (verified_ok(dest)) {
        fprintf(stderr, "already present: %s\n", dest.filename().string().c_str());
        return;
    }

    fs::create_directories(dest.parent_path());
    fs::path partial = dest;
    partial += ".partial";

    auto args = curl_base(headers);
    // keep error reporting but swap --silent for a progress bar
    args.erase(std::remove(args.begin(), args.end(), std::string("--silent")), args.end());
    args.push_back("--progress-bar");
    args.push_back("--continue-at");
    args.push_back("-");
    args.push_back("--output");
    args.push_back(partial.string());
    args.push_back(url);

    std::string cfg = curl_header_config(headers);
    int rc = run_process(args, headers.empty() ? nullptr : &cfg, nullptr);

    // curl exits 33/22 when the partial file already covers the full range;
    // accept it if the size checks out.
    if (rc != 0) {
        bool complete = expected_size > 0 && fs::exists(partial, ec) &&
                        fs::file_size(partial, ec) == expected_size;
        if (!complete) {
            fail("download failed (curl exit " + std::to_string(rc) + "): " + url +
                 "\n  partial data kept at " + partial.string() + " — rerun to resume");
        }
    }

    if (expected_size > 0) {
        uint64_t got = fs::exists(partial, ec) ? (uint64_t) fs::file_size(partial, ec) : 0;
        if (got != expected_size) {
            fail("size mismatch for " + dest.filename().string() + ": expected " +
                 std::to_string(expected_size) + " bytes, got " + std::to_string(got) +
                 "\n  partial data kept at " + partial.string() + " — rerun to resume");
        }
    }
    if (verify && !digest_hex.empty()) {
        fprintf(stderr, "verifying sha256... ");
        fflush(stderr);
        std::string got = SHA256::file_hex(partial.string());
        if (got != digest_hex) {
            fprintf(stderr, "MISMATCH\n");
            fs::remove(partial, ec);
            fail("sha256 mismatch for " + dest.filename().string() +
                 " (expected " + digest_hex + ", got " + got + "); removed corrupt download");
        }
        fprintf(stderr, "ok\n");
    }

    fs::rename(partial, dest, ec);
    if (ec) {
        fail("failed to move " + partial.string() + " into place: " + ec.message());
    }
}

static std::string digest_to_hex(const std::string & digest) {
    return starts_with(digest, "sha256:") ? digest.substr(7) : std::string();
}

// ---- Ollama registry ----------------------------------------------------

static LocalModel pull_ollama(const ModelRef & ref, const PullOptions & opts) {
    const std::string registry = ollama_registry();
    const std::string base = registry + "/v2/" + ref.ns + "/" + ref.name;
    fs::path dir = ref.store_dir(models_root());

    fprintf(stderr, "pulling manifest %s from %s\n", ref.display().c_str(), registry.c_str());
    JValue man = http_get_json(base + "/manifests/" + ref.tag,
                               { "Accept: application/vnd.docker.distribution.manifest.v2+json" });

    const JValue * layers = man.find("layers");
    if (!layers || !layers->is_arr()) {
        fail("unexpected registry manifest for " + ref.display() + " (no layers)");
    }

    JValue out = JValue::make_obj();
    out.set("name", JValue::make_str(ref.display()));
    out.set("source", JValue::make_str("ollama"));
    out.set("registry_ref", JValue::make_str(registry + "/" + ref.ns + "/" + ref.name + ":" + ref.tag));

    struct Item {
        std::string url, file, digest_hex, kind;
        uint64_t    size;
    };
    std::vector<Item> blobs;
    uint64_t model_size = 0;

    for (const auto & layer : layers->arr) {
        std::string media = layer.get_str("mediaType");
        std::string digest = layer.get_str("digest");
        uint64_t size = (uint64_t) layer.get_num("size", 0);
        if (digest.empty()) {
            continue;
        }
        std::string url = base + "/blobs/" + digest;
        std::string hex = digest_to_hex(digest);

        if (media == "application/vnd.ollama.image.model") {
            blobs.push_back({ url, "model.gguf", hex, "model", size });
            out.set("digest", JValue::make_str(digest));
            model_size = size;
        } else if (media == "application/vnd.ollama.image.projector") {
            blobs.push_back({ url, "mmproj.gguf", hex, "projector", size });
            out.set("mmproj_file", JValue::make_str("mmproj.gguf"));
        } else if (media == "application/vnd.ollama.image.adapter") {
            blobs.push_back({ url, "adapter.gguf", hex, "adapter", size });
            out.set("adapter_file", JValue::make_str("adapter.gguf"));
        } else if (media == "application/vnd.ollama.image.params") {
            blobs.push_back({ url, "params.json", hex, "params", size });
        } else if (media == "application/vnd.ollama.image.system") {
            blobs.push_back({ url, "system.txt", hex, "system", size });
        } else if (media == "application/vnd.ollama.image.template") {
            blobs.push_back({ url, "template.txt", hex, "template", size });
        } else if (media == "application/vnd.ollama.image.license") {
            blobs.push_back({ url, "license.txt", hex, "license", size });
        }
    }

    bool has_model = false;
    for (const auto & b : blobs) {
        has_model |= b.kind == "model";
    }
    if (!has_model) {
        fail(ref.display() + " has no GGUF weights layer; alpacca cannot run it");
    }

    fs::create_directories(dir);
    for (const auto & b : blobs) {
        fprintf(stderr, "pulling %-9s %s (%s)\n", b.kind.c_str(),
                b.digest_hex.substr(0, 12).c_str(), human_size(b.size).c_str());
        fs::path target = dir / b.file;
        std::error_code ec;
        if (opts.force) {
            fs::remove(target, ec);
        }
        download_file(b.url, target, b.size, b.digest_hex, {}, opts.verify);
    }

    // Fold the params layer into our manifest so `run` can apply defaults.
    {
        std::ifstream f(dir / "params.json", std::ios::binary);
        if (f) {
            std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            JValue params;
            std::string err;
            if (json_parse(text, params, err) && params.is_obj()) {
                out.set("params", params);
            }
        }
    }
    {
        std::ifstream f(dir / "system.txt", std::ios::binary);
        if (f) {
            std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            if (!trim(text).empty()) {
                out.set("system", JValue::make_str(text));
            }
        }
    }

    out.set("model_file", JValue::make_str("model.gguf"));
    out.set("size", JValue::make_num((double) model_size));
    out.set("pulled_at", JValue::make_str(now_iso8601()));
    write_manifest(dir, out);

    auto local = find_local(ref);
    if (!local) {
        fail("internal error: model not found after pull");
    }
    fprintf(stderr, "success: %s ready (%s)\n", ref.display().c_str(),
            human_size(model_size).c_str());
    return *local;
}

// ---- Hugging Face -------------------------------------------------------

struct HfFile {
    std::string path;
    uint64_t    size = 0;
    std::string sha256;
};

static std::vector<HfFile> hf_list_gguf(const std::string & org, const std::string & repo,
                                        const std::vector<std::string> & headers) {
    std::string url = hf_endpoint() + "/api/models/" + org + "/" + repo + "/tree/main?recursive=true";
    JValue tree = http_get_json(url, headers);
    if (!tree.is_arr()) {
        fail("unexpected response listing " + org + "/" + repo);
    }
    std::vector<HfFile> files;
    for (const auto & e : tree.arr) {
        std::string path = e.get_str("path");
        if (!ends_with(to_lower(path), ".gguf")) {
            continue;
        }
        HfFile f;
        f.path = path;
        f.size = (uint64_t) e.get_num("size", 0);
        if (const JValue * lfs = e.find("lfs")) {
            f.sha256 = lfs->get_str("oid");
            if (f.size == 0) {
                f.size = (uint64_t) lfs->get_num("size", 0);
            }
        }
        files.push_back(std::move(f));
    }
    return files;
}

static std::string base_name(const std::string & path) {
    size_t slash = path.find_last_of('/');
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

static bool is_mmproj(const std::string & path) {
    return starts_with(to_lower(base_name(path)), "mmproj");
}

// Choose the GGUF to download. Empty selector → common quant preference.
static const HfFile * hf_choose(const std::vector<HfFile> & files, const std::string & selector) {
    std::vector<const HfFile *> weights;
    for (const auto & f : files) {
        if (!is_mmproj(f.path)) {
            weights.push_back(&f);
        }
    }
    if (weights.empty()) {
        return nullptr;
    }

    if (!selector.empty()) {
        for (const HfFile * f : weights) { // exact path or filename
            if (f->path == selector || base_name(f->path) == selector) {
                return f;
            }
        }
        std::string sel = to_lower(selector);
        const HfFile * best = nullptr;
        for (const HfFile * f : weights) { // substring (quant tags like q4_k_m)
            if (to_lower(base_name(f->path)).find(sel) != std::string::npos) {
                if (!best || f->path.size() < best->path.size()) {
                    best = f;
                }
            }
        }
        return best;
    }

    static const char * preference[] = {
        "q4_k_m", "q4_k_s", "q5_k_m", "q5_k_s", "q4_0", "q8_0", "q6_k", "f16", "bf16", "f32",
    };
    for (const char * q : preference) {
        for (const HfFile * f : weights) {
            if (to_lower(base_name(f->path)).find(q) != std::string::npos) {
                return f;
            }
        }
    }
    return weights.front();
}

// GGUFs above ~50 GB are usually split into "-00001-of-000NN.gguf" parts;
// llama.cpp wants all parts side by side and is pointed at the first one.
static std::vector<const HfFile *> hf_collect_parts(const std::vector<HfFile> & files,
                                                    const HfFile * chosen) {
    static const std::regex split_re("^(.*)-(\\d{5})-of-(\\d{5})\\.gguf$", std::regex::icase);
    std::smatch m;
    if (!std::regex_match(chosen->path, m, split_re)) {
        return { chosen };
    }
    std::string prefix = m[1].str();
    std::string total = m[3].str();
    std::vector<const HfFile *> parts;
    for (const auto & f : files) {
        std::smatch fm;
        if (std::regex_match(f.path, fm, split_re) && fm[1].str() == prefix && fm[3].str() == total) {
            parts.push_back(&f);
        }
    }
    std::sort(parts.begin(), parts.end(), [](const HfFile * a, const HfFile * b) {
        return a->path < b->path;
    });
    return parts;
}

static LocalModel pull_hf(const ModelRef & ref, const PullOptions & opts) {
    std::vector<std::string> headers;
    if (!hf_token().empty()) {
        headers.push_back("Authorization: Bearer " + hf_token());
    }

    fprintf(stderr, "fetching file list for %s/%s\n", ref.ns.c_str(), ref.name.c_str());
    std::string repo = ref.name; // actual repo to download from
    std::vector<HfFile> files;
    try {
        files = hf_list_gguf(ref.ns, repo, headers);
    } catch (const std::exception &) {
        // missing repo — maybe only the -GGUF sibling exists
    }
    if (files.empty() && !ends_with(to_lower(repo), "-gguf")) {
        // Many models publish weights as safetensors with GGUFs in a sibling
        // "<repo>-GGUF" repo (e.g. NousResearch/Hermes-3-Llama-3.1-8B-GGUF).
        std::string alt = repo + "-GGUF";
        fprintf(stderr, "no GGUF files in %s/%s — trying %s/%s\n",
                ref.ns.c_str(), repo.c_str(), ref.ns.c_str(), alt.c_str());
        try {
            files = hf_list_gguf(ref.ns, alt, headers);
            if (!files.empty()) {
                repo = alt;
            }
        } catch (const std::exception &) {
            // fall through to the error below
        }
    }
    if (files.empty()) {
        fail("no .gguf files in " + ref.ns + "/" + ref.name +
             " — alpacca runs GGUF models (try a -GGUF repo, e.g. from ggml-org or bartowski)");
    }

    const HfFile * chosen = hf_choose(files, ref.tag);
    if (!chosen) {
        fail("no GGUF in " + ref.ns + "/" + repo + " matches '" + ref.tag + "'");
    }
    auto parts = hf_collect_parts(files, chosen);

    // Vision repos usually ship a separate mmproj GGUF; grab the best one.
    const HfFile * mmproj = nullptr;
    for (const auto & f : files) {
        if (!is_mmproj(f.path)) {
            continue;
        }
        bool better = !mmproj ||
                      (to_lower(f.path).find("f16") != std::string::npos &&
                       to_lower(mmproj->path).find("f16") == std::string::npos);
        if (better) {
            mmproj = &f;
        }
    }

    fs::path dir = ref.store_dir(models_root());
    fs::create_directories(dir);

    uint64_t total_size = 0;
    for (const HfFile * p : parts) {
        total_size += p->size;
    }
    fprintf(stderr, "selected %s (%s%s)\n", chosen->path.c_str(), human_size(total_size).c_str(),
            parts.size() > 1 ? (", " + std::to_string(parts.size()) + " parts").c_str() : "");

    std::string resolve = hf_endpoint() + "/" + ref.ns + "/" + repo + "/resolve/main/";
    size_t idx = 0;
    for (const HfFile * p : parts) {
        idx++;
        fprintf(stderr, "downloading %zu/%zu %s (%s)\n", idx, parts.size(),
                base_name(p->path).c_str(), human_size(p->size).c_str());
        fs::path target = dir / base_name(p->path);
        std::error_code ec;
        if (opts.force) {
            fs::remove(target, ec);
        }
        download_file(resolve + p->path, target, p->size, p->sha256, headers, opts.verify);
    }
    if (mmproj) {
        fprintf(stderr, "downloading projector %s (%s)\n", base_name(mmproj->path).c_str(),
                human_size(mmproj->size).c_str());
        download_file(resolve + mmproj->path, dir / base_name(mmproj->path),
                      mmproj->size, mmproj->sha256, headers, opts.verify);
    }

    JValue out = JValue::make_obj();
    out.set("name", JValue::make_str(ref.display()));
    out.set("source", JValue::make_str("hf"));
    out.set("registry_ref", JValue::make_str(hf_endpoint() + "/" + ref.ns + "/" + repo));
    out.set("model_file", JValue::make_str(base_name(parts.front()->path)));
    if (mmproj) {
        out.set("mmproj_file", JValue::make_str(base_name(mmproj->path)));
    }
    if (!chosen->sha256.empty()) {
        out.set("digest", JValue::make_str("sha256:" + chosen->sha256));
    }
    out.set("size", JValue::make_num((double) total_size));
    out.set("pulled_at", JValue::make_str(now_iso8601()));
    write_manifest(dir, out);

    auto local = find_local(ref);
    if (!local) {
        fail("internal error: model not found after pull");
    }
    fprintf(stderr, "success: %s ready (%s)\n", ref.display().c_str(),
            human_size(total_size).c_str());
    return *local;
}

// ---- entry point --------------------------------------------------------

LocalModel pull_model(const ModelRef & ref, const PullOptions & opts) {
    if (ref.source == ModelRef::FILE) {
        fail("'" + ref.file.string() + "' is a file path; nothing to pull");
    }
    if (!opts.force) {
        if (auto existing = find_local(ref)) {
            fprintf(stderr, "%s is already installed (use --force to re-pull)\n",
                    ref.display().c_str());
            return *existing;
        }
    }
    if (!on_path("curl")) {
        fail("pulling models requires `curl` on PATH");
    }
    return ref.source == ModelRef::OLLAMA ? pull_ollama(ref, opts) : pull_hf(ref, opts);
}

} // namespace alpacca
