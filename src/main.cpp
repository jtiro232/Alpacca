// Alpacca — a friendly terminal front-end for llama.cpp.
//
// Model management (pull/list/rm/show) is Alpacca's own; inference is
// delegated to the stock llama.cpp binaries via exec, so it runs at full
// upstream speed. Built on llama.cpp (MIT, © The ggml authors); model
// distribution ideas borrowed from the Ollama project (MIT).
// MIT License. See LICENSE and THIRD-PARTY-NOTICES.md.
#include "pull.h"
#include "runner.h"
#include "store.h"

#include <algorithm>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
#endif

using namespace alpacca;

static void print_help() {
    printf(R"(alpacca — llama.cpp in your terminal, with Ollama-style model management

usage: alpacca <command> [arguments]

model management:
  pull <model>           download a model into ~/.alpacca/models
  list                   list installed models             (alias: ls)
  rm <model>             remove an installed model         (alias: remove)
  show <model>           show a model's manifest and files

inference (stock llama.cpp, zero overhead — alpacca execs the real binaries):
  run <model> [prompt]   chat with a model (one-shot when a prompt is given)
  serve <model>          OpenAI-compatible API server (llama-server)
  cli [args]             raw llama-cli passthrough
  server [args]          raw llama-server passthrough
  <tool> [args]          any other llama.cpp tool: quantize, bench, tokenize,
                         perplexity, gguf-split, imatrix, ...

other:
  doctor                 check the installation
  version                print alpacca + llama.cpp versions
  help                   this help

model references:
  llama3.2:1b                      Ollama registry (registry.ollama.ai)
  ollama:user/model:tag            Ollama registry, user namespace
  hf:org/repo  |  org/repo         Hugging Face repo (best GGUF quant)
  hf:org/repo:Q5_K_M               Hugging Face repo, specific quant or file
  ./path/to/model.gguf             local GGUF file

examples:
  alpacca pull llama3.2:1b
  alpacca run llama3.2:1b                       # interactive chat
  alpacca run llama3.2:1b "why is the sky blue" # one-shot
  alpacca run hf:ggml-org/gemma-3-4b-it-GGUF -- --temp 0.2
  alpacca serve llama3.2:1b --port 8080
  alpacca quantize -m in.gguf out.gguf Q4_K_M

flags after the model name are passed straight to llama.cpp, so every
llama.cpp option works here. environment: $ALPACCA_HOME, $ALPACCA_HOST,
$ALPACCA_PORT, $ALPACCA_LLAMA_BIN_DIR, $HF_TOKEN.
)");
}

static int cmd_pull_front(const std::vector<std::string> & args) {
    PullOptions opts;
    std::string name;
    for (const auto & a : args) {
        if (a == "--force" || a == "-f") {
            opts.force = true;
        } else if (a == "--no-verify") {
            opts.verify = false;
        } else if (!starts_with(a, "-") && name.empty()) {
            name = a;
        } else {
            fail("unknown pull argument: " + a);
        }
    }
    if (name.empty()) {
        fail("usage: alpacca pull <model> [--force] [--no-verify]");
    }
    pull_model(parse_model_ref(name), opts);
    return 0;
}

static int cmd_list_front() {
    auto models = list_models();
    if (models.empty()) {
        printf("no models installed — try: alpacca pull llama3.2:1b\n");
        return 0;
    }
    size_t name_w = 4;
    for (const auto & m : models) {
        name_w = std::max(name_w, m.name.size());
    }
    printf("%-*s  %-8s  %-10s  %s\n", (int) name_w, "NAME", "SOURCE", "SIZE", "PULLED");
    for (const auto & m : models) {
        printf("%-*s  %-8s  %-10s  %s\n", (int) name_w, m.name.c_str(), m.source.c_str(),
               human_size(m.size).c_str(), m.modified.c_str());
    }
    return 0;
}

static int cmd_rm_front(const std::vector<std::string> & args) {
    if (args.empty()) {
        fail("usage: alpacca rm <model> [<model>...]");
    }
    int rc = 0;
    for (const auto & name : args) {
        ModelRef ref = parse_model_ref(name);
        if (remove_model(ref)) {
            printf("removed %s\n", ref.display().c_str());
        } else {
            fprintf(stderr, "alpacca: %s is not installed\n", ref.display().c_str());
            rc = 1;
        }
    }
    return rc;
}

static int cmd_show_front(const std::vector<std::string> & args) {
    if (args.empty()) {
        fail("usage: alpacca show <model>");
    }
    ModelRef ref = parse_model_ref(args[0]);
    auto local = find_local(ref);
    if (!local) {
        fail(ref.display() + " is not installed (try `alpacca pull " + args[0] + "`)");
    }
    if (local->manifest.is_obj()) {
        printf("%s\n", local->manifest.dump(2).c_str());
    } else {
        printf("{ \"model_file\": \"%s\" }\n", local->model_path.string().c_str());
    }
    if (!local->dir.empty()) {
        printf("\nfiles in %s:\n", local->dir.string().c_str());
        std::error_code ec;
        for (const auto & f : fs::directory_iterator(local->dir, ec)) {
            if (f.is_regular_file(ec)) {
                printf("  %-24s %s\n", f.path().filename().string().c_str(),
                       human_size((uint64_t) f.file_size(ec)).c_str());
            }
        }
    }
    return 0;
}

int main(int argc, char ** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8); // model names and chat output are UTF-8
#endif
    std::vector<std::string> args(argv + 1, argv + argc);
    if (args.empty() || args[0] == "help" || args[0] == "--help" || args[0] == "-h") {
        print_help();
        return 0;
    }

    std::string cmd = args[0];
    std::vector<std::string> rest(args.begin() + 1, args.end());

    try {
        if (cmd == "pull") {
            return cmd_pull_front(rest);
        }
        if (cmd == "list" || cmd == "ls") {
            return cmd_list_front();
        }
        if (cmd == "rm" || cmd == "remove" || cmd == "delete") {
            return cmd_rm_front(rest);
        }
        if (cmd == "show") {
            return cmd_show_front(rest);
        }
        if (cmd == "run") {
            return cmd_run(rest);
        }
        if (cmd == "serve") {
            return cmd_serve(rest);
        }
        if (cmd == "doctor") {
            return cmd_doctor();
        }
        if (cmd == "version" || cmd == "--version" || cmd == "-v") {
            return cmd_version();
        }
        // anything else: llama.cpp tool passthrough (cli, server, quantize,
        // bench, tokenize, perplexity, ...)
        return cmd_tool(cmd, rest);
    } catch (const std::exception & e) {
        fprintf(stderr, "alpacca: error: %s\n", e.what());
        return 1;
    }
}
