// Alpacca — locating and launching the bundled llama.cpp binaries.
//
// alpacca execs the stock llama.cpp tools (llama-cli, llama-server, ...)
// after resolving model names from the local store. The wrapper process is
// replaced via exec, so inference performance is exactly upstream
// llama.cpp. MIT License. See LICENSE.
#include "runner.h"

#include "pull.h"

#include <cstdio>
#ifndef _WIN32
    #include <unistd.h>
#endif

namespace alpacca {

#ifndef ALPACCA_VERSION
#define ALPACCA_VERSION "dev"
#endif

#ifdef _WIN32
static const char * EXE_SUFFIX = ".exe";
#else
static const char * EXE_SUFFIX = "";
#endif

static bool is_executable(const fs::path & p) {
    std::error_code ec;
    if (!fs::exists(p, ec)) {
        return false;
    }
#ifdef _WIN32
    return true;
#else
    return access(p.c_str(), X_OK) == 0;
#endif
}

std::string find_llama_tool(const std::string & name) {
    std::vector<fs::path> dirs;
    std::string env_dir = env_or("ALPACCA_LLAMA_BIN_DIR", "");
    if (!env_dir.empty()) {
        dirs.push_back(env_dir);
    }
    dirs.push_back(self_exe_dir());

    for (const auto & d : dirs) {
        fs::path candidate = d / (name + EXE_SUFFIX);
        if (is_executable(candidate)) {
            return candidate.string();
        }
    }
    if (on_path(name)) {
        return name;
    }
    return "";
}

static std::string require_tool(const std::string & name) {
    std::string tool = find_llama_tool(name);
    if (tool.empty()) {
        fail(name + " not found. Build it (cmake --build build) or set $ALPACCA_LLAMA_BIN_DIR "
                    "to a directory containing the llama.cpp binaries.");
    }
    return tool;
}

// Resolve a model reference, pulling it on first use (Ollama-style).
static LocalModel resolve_or_pull(const std::string & name) {
    ModelRef ref = parse_model_ref(name);
    if (auto local = find_local(ref)) {
        return *local;
    }
    if (ref.source == ModelRef::FILE) {
        fail("model file not found: " + ref.file.string());
    }
    fprintf(stderr, "%s is not installed yet — pulling it first\n", ref.display().c_str());
    return pull_model(ref, PullOptions{});
}

// Map Ollama-style parameters from the model manifest onto llama.cpp flags.
// These are start-up defaults only; flags given by the user come later on
// the command line and therefore win.
static void append_manifest_args(const LocalModel & model, bool for_server,
                                 std::vector<std::string> & argv) {
    if (!model.mmproj_path.empty()) {
        argv.push_back("--mmproj");
        argv.push_back(model.mmproj_path.string());
    }
    if (!model.dir.empty()) {
        std::error_code ec;
        fs::path adapter = model.dir / model.manifest.get_str("adapter_file");
        if (!model.manifest.get_str("adapter_file").empty() && fs::exists(adapter, ec)) {
            argv.push_back("--lora");
            argv.push_back(adapter.string());
        }
    }

    const JValue * params = model.manifest.find("params");
    if (params && params->is_obj()) {
        struct Map {
            const char * key;
            const char * flag;
            bool         server_ok;
        };
        static const Map maps[] = {
            { "temperature",    "--temp",           true  },
            { "top_k",          "--top-k",          true  },
            { "top_p",          "--top-p",          true  },
            { "min_p",          "--min-p",          true  },
            { "repeat_penalty", "--repeat-penalty", true  },
            { "num_ctx",        "--ctx-size",       true  },
            { "num_predict",    "--n-predict",      false },
            { "seed",           "--seed",           true  },
        };
        for (const auto & m : maps) {
            if (for_server && !m.server_ok) {
                continue;
            }
            const JValue * v = params->find(m.key);
            if (v && v->is_num()) {
                char buf[32];
                if (v->num == (double) (long long) v->num) {
                    snprintf(buf, sizeof(buf), "%lld", (long long) v->num);
                } else {
                    snprintf(buf, sizeof(buf), "%g", v->num);
                }
                argv.push_back(m.flag);
                argv.push_back(buf);
            }
        }
        const JValue * stop = params->find("stop");
        if (stop && stop->is_arr()) {
            for (const auto & s : stop->arr) {
                if (s.is_str() && !s.str.empty()) {
                    argv.push_back("--reverse-prompt");
                    argv.push_back(s.str);
                }
            }
        }
    }

    if (!for_server) {
        std::string system = model.manifest.get_str("system");
        if (!trim(system).empty()) {
            argv.push_back("--system-prompt");
            argv.push_back(trim(system));
        }
    }
}

static int exec_or_fail(std::vector<std::string> & argv) {
    exec_replace(argv); // only returns on error
    return 127;
}

int cmd_run(const std::vector<std::string> & args) {
    if (args.empty()) {
        fail("usage: alpacca run <model> [prompt] [llama-cli flags...]\n"
             "       run `alpacca list` to see installed models");
    }

    std::string model_name = args[0];
    std::vector<std::string> prompt_words;
    std::vector<std::string> extra;
    bool verbatim = false;
    for (size_t i = 1; i < args.size(); i++) {
        if (!verbatim && args[i] == "--") {
            verbatim = true;
        } else if (verbatim || starts_with(args[i], "-")) {
            verbatim = true; // first flag: everything after goes to llama-cli
            extra.push_back(args[i]);
        } else {
            std::string w = args[i];
            prompt_words.push_back(w);
        }
    }

    LocalModel model = resolve_or_pull(model_name);
    std::string tool = require_tool("llama-cli");

    std::vector<std::string> argv = { tool, "-m", model.model_path.string() };
    append_manifest_args(model, /*for_server=*/false, argv);

    if (!prompt_words.empty()) {
        std::string prompt;
        for (size_t i = 0; i < prompt_words.size(); i++) {
            if (i) {
                prompt += " ";
            }
            prompt += prompt_words[i];
        }
        argv.push_back("--prompt");
        argv.push_back(prompt);
        argv.push_back("--single-turn");
    }

    argv.insert(argv.end(), extra.begin(), extra.end());
    return exec_or_fail(argv);
}

int cmd_serve(const std::vector<std::string> & args) {
    if (args.empty()) {
        fail("usage: alpacca serve <model> [llama-server flags...]\n"
             "       serves an OpenAI-compatible API (default 127.0.0.1:8080)");
    }

    LocalModel model = resolve_or_pull(args[0]);
    std::string tool = require_tool("llama-server");

    std::vector<std::string> argv = { tool, "-m", model.model_path.string() };
    append_manifest_args(model, /*for_server=*/true, argv);

    std::string host = env_or("ALPACCA_HOST", "");
    std::string port = env_or("ALPACCA_PORT", "");
    if (!host.empty()) {
        argv.push_back("--host");
        argv.push_back(host);
    }
    if (!port.empty()) {
        argv.push_back("--port");
        argv.push_back(port);
    }

    argv.insert(argv.end(), args.begin() + 1, args.end());
    return exec_or_fail(argv);
}

int cmd_tool(const std::string & tool_suffix, const std::vector<std::string> & args) {
    std::string tool = find_llama_tool("llama-" + tool_suffix);
    if (tool.empty()) {
        fail("unknown command '" + tool_suffix + "' (and no llama-" + tool_suffix +
             " binary found) — run `alpacca help`");
    }

    std::vector<std::string> argv = { tool };
    for (size_t i = 0; i < args.size(); i++) {
        // Convenience: let -m take an alpacca model name as well as a path.
        if ((args[i] == "-m" || args[i] == "--model") && i + 1 < args.size()) {
            argv.push_back(args[i]);
            std::error_code ec;
            const std::string & value = args[i + 1];
            if (!fs::exists(value, ec)) {
                try {
                    if (auto local = find_local(parse_model_ref(value))) {
                        argv.push_back(local->model_path.string());
                        i++;
                        continue;
                    }
                } catch (const std::exception &) {
                    // fall through: hand the raw value to the tool
                }
            }
            argv.push_back(value);
            i++;
            continue;
        }
        argv.push_back(args[i]);
    }
    return exec_or_fail(argv);
}

int cmd_doctor() {
    printf("alpacca %s\n\n", ALPACCA_VERSION);

    printf("data dir:    %s\n", alpacca_home().string().c_str());
    std::error_code ec;
    fs::create_directories(models_root(), ec);
    printf("models dir:  %s (%s)\n", models_root().string().c_str(), ec ? "NOT WRITABLE" : "ok");

    bool curl_ok = on_path("curl");
    printf("curl:        %s\n", curl_ok ? "found (needed for `alpacca pull`)" : "MISSING — `alpacca pull` will not work");

    static const char * tools[] = {
        "llama-cli", "llama-server", "llama-quantize", "llama-bench",
        "llama-tokenize", "llama-perplexity", "llama-gguf-split", "llama-imatrix",
    };
    printf("\nllama.cpp tools:\n");
    bool any_missing = false;
    for (const char * t : tools) {
        std::string p = find_llama_tool(t);
        printf("  %-18s %s\n", t, p.empty() ? "missing" : p.c_str());
        any_missing |= p.empty();
    }
    if (any_missing) {
        printf("\nsome tools are missing — build them with:\n"
               "  cmake -B build && cmake --build build --parallel\n"
               "or point $ALPACCA_LLAMA_BIN_DIR at an existing llama.cpp build.\n");
    }

    std::string cli = find_llama_tool("llama-cli");
    if (!cli.empty()) {
        printf("\nllama.cpp version: ");
        fflush(stdout);
        std::vector<std::string> v = { cli, "--version" };
        run_process(v); // --version prints to stderr
    }
    return 0;
}

int cmd_version() {
    printf("alpacca %s\n", ALPACCA_VERSION);
    std::string cli = find_llama_tool("llama-cli");
    if (!cli.empty()) {
        std::vector<std::string> v = { cli, "--version" };
        run_process(v);
    } else {
        printf("(llama.cpp binaries not found — run `alpacca doctor`)\n");
    }
    return 0;
}

} // namespace alpacca
