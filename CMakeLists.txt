# Alpacca — llama.cpp in your terminal, with Ollama-style model management.
#
# Builds two things:
#   1. the vendored llama.cpp (unmodified, pinned via git submodule) — all
#      inference binaries: llama-cli, llama-server, llama-quantize, ...
#   2. the `alpacca` front-end, which manages models and execs those
#      binaries. Everything lands together in <build>/bin.
#
# Backend flags pass straight through to llama.cpp/ggml, e.g.:
#   cmake -B build -DGGML_CUDA=ON        # NVIDIA
#   cmake -B build -DGGML_VULKAN=ON      # Vulkan
#   cmake -B build -DGGML_METAL=ON       # Apple (default on macOS)
cmake_minimum_required(VERSION 3.14)
project(alpacca VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(GNUInstallDirs)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "build type" FORCE)
endif()

# Keep alpacca and the llama.cpp tools side by side; alpacca looks for them
# next to its own executable first. The per-config variables keep
# multi-config generators (Visual Studio) from nesting Release/ subdirs.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
foreach(cfg IN ITEMS DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${cfg} ${CMAKE_BINARY_DIR}/bin)
endforeach()

option(ALPACCA_BUILD_LLAMA "alpacca: build the vendored llama.cpp"           ON)
option(ALPACCA_STATIC      "alpacca: build llama.cpp statically (no .so)"    ON)

if (ALPACCA_BUILD_LLAMA)
    if (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp/CMakeLists.txt)
        message(FATAL_ERROR
            "vendor/llama.cpp is empty — fetch the pinned submodule first:\n"
            "  git submodule update --init --depth 1")
    endif()

    if (ALPACCA_STATIC)
        set(BUILD_SHARED_LIBS OFF CACHE BOOL "build shared libraries")
    endif()

    # llama.cpp build shape: the inference tools, nothing else. The web UI
    # needs network/npm at build time, so it is off by default; flip it on
    # with -DLLAMA_BUILD_UI=ON if you want the browser UI in llama-server.
    set(LLAMA_BUILD_COMMON   ON  CACHE BOOL "llama: build common utils")
    set(LLAMA_BUILD_TOOLS    ON  CACHE BOOL "llama: build tools")
    set(LLAMA_BUILD_SERVER   ON  CACHE BOOL "llama: build server")
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "llama: build examples")
    set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "llama: build tests")
    set(LLAMA_BUILD_APP      OFF CACHE BOOL "llama: build unified binary")
    set(LLAMA_BUILD_UI       OFF CACHE BOOL "llama: build server web UI")
    set(LLAMA_TOOLS_INSTALL  ON  CACHE BOOL "llama: install tools")

    add_subdirectory(vendor/llama.cpp)
endif()

add_executable(alpacca
    src/main.cpp
    src/json.cpp
    src/pull.cpp
    src/runner.cpp
    src/sha256.cpp
    src/store.cpp
    src/util.cpp
)
target_compile_definitions(alpacca PRIVATE ALPACCA_VERSION="${PROJECT_VERSION}")
if (MSVC)
    target_compile_options(alpacca PRIVATE /W3 /utf-8)
else()
    target_compile_options(alpacca PRIVATE -Wall -Wextra)
endif()

install(TARGETS alpacca RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
