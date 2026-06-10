// Alpacca — locating and launching the bundled llama.cpp binaries.
// MIT License. See LICENSE.
#pragma once

#include "store.h"

#include <string>
#include <vector>

namespace alpacca {

// Locate a llama.cpp tool ("llama-cli", "llama-server", ...). Search order:
// $ALPACCA_LLAMA_BIN_DIR, the directory of the alpacca executable, PATH.
// Returns the path/name to exec, or "" when not found anywhere.
std::string find_llama_tool(const std::string & name);

// `alpacca run <model> [prompt] [llama-cli flags...]`
int cmd_run(const std::vector<std::string> & args);

// `alpacca serve <model> [llama-server flags...]`
int cmd_serve(const std::vector<std::string> & args);

// `alpacca <tool> [args...]` → exec llama-<tool> with model-name resolution
// applied to `-m/--model` values.
int cmd_tool(const std::string & tool, const std::vector<std::string> & args);

int cmd_doctor();
int cmd_version();

} // namespace alpacca
