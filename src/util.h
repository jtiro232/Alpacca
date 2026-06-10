// Alpacca — small shared utilities (strings, env, filesystem, processes).
// MIT License. See LICENSE.
#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace alpacca {

namespace fs = std::filesystem;

// ---- strings ----------------------------------------------------------

bool starts_with(const std::string & s, const std::string & prefix);
bool ends_with(const std::string & s, const std::string & suffix);
std::string to_lower(std::string s);
std::string trim(const std::string & s);
std::vector<std::string> split(const std::string & s, char sep);

// "1.2 GB" style rendering of a byte count.
std::string human_size(uint64_t bytes);

// Current UTC time as ISO-8601 ("2026-06-10T12:34:56Z").
std::string now_iso8601();

// ---- environment / paths ----------------------------------------------

std::string env_or(const char * name, const std::string & def);

// Root of the Alpacca data dir: $ALPACCA_HOME or ~/.alpacca
fs::path alpacca_home();

// Directory containing the running alpacca executable (best effort).
fs::path self_exe_dir();

// Make a string safe to use as a single path component.
std::string sanitize_component(const std::string & s);

// ---- processes ---------------------------------------------------------

// Run a command (argv style, no shell). Optionally feed stdin_data and/or
// capture stdout into *out. stderr passes through to the terminal.
// Returns the exit code, or -1 if the process could not be started.
int run_process(const std::vector<std::string> & args,
                const std::string * stdin_data = nullptr,
                std::string * out = nullptr);

// Replace the current process (execvp). Only returns on failure.
void exec_replace(const std::vector<std::string> & args);

// True if `name` resolves to an executable on PATH.
bool on_path(const std::string & name);

// ---- errors ------------------------------------------------------------

[[noreturn]] void fail(const std::string & msg);

} // namespace alpacca
