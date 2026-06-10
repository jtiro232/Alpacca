# Alpacca installer (Windows / PowerShell).
#
# Run from a clone:        .\scripts\install.ps1
# Or bootstrap directly:   irm https://raw.githubusercontent.com/jtiro232/Alpacca/main/scripts/install.ps1 | iex
#
# What it does: fetch sources (when bootstrapping), build llama.cpp +
# alpacca with CMake, install to %LOCALAPPDATA%\Alpacca\bin, and add that
# directory to your user PATH — so a NEW terminal can just run `alpacca`.
#
# Requirements (one-time, via winget or the installers):
#   winget install Git.Git Kitware.CMake
#   winget install Microsoft.VisualStudio.2022.BuildTools --override `
#       "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --passive"
#
# knobs:
#   $env:ALPACCA_PREFIX   install location (default %LOCALAPPDATA%\Alpacca)
#   $env:CMAKE_FLAGS      extra build flags, e.g. "-DGGML_CUDA=ON"

$ErrorActionPreference = "Stop"

$RepoUrl = "https://github.com/jtiro232/Alpacca"
$Prefix  = if ($env:ALPACCA_PREFIX) { $env:ALPACCA_PREFIX } else { Join-Path $env:LOCALAPPDATA "Alpacca" }

foreach ($tool in @("git", "cmake")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        Write-Error ("'{0}' is required. Install it first, e.g.:  winget install Git.Git Kitware.CMake" -f $tool)
    }
}

# Find the sources: next to this script when run from a clone, otherwise
# clone into <prefix>\src (bootstrap mode, e.g. irm | iex).
$src = $null
if ($PSScriptRoot -and (Test-Path (Join-Path $PSScriptRoot "..\CMakeLists.txt"))) {
    $src = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
} else {
    $src = Join-Path $Prefix "src"
    if (Test-Path (Join-Path $src ".git")) {
        Write-Host "==> updating sources in $src"
        git -C $src pull --ff-only
        if ($LASTEXITCODE -ne 0) { throw "git pull failed" }
    } else {
        Write-Host "==> cloning $RepoUrl into $src"
        New-Item -ItemType Directory -Force -Path (Split-Path $src) | Out-Null
        git clone --depth 1 $RepoUrl $src
        if ($LASTEXITCODE -ne 0) { throw "git clone failed" }
    }
}

Write-Host "==> fetching pinned llama.cpp submodule"
git -C $src submodule update --init --depth 1
if ($LASTEXITCODE -ne 0) { throw "git submodule update failed" }

Write-Host "==> configuring (this needs the Visual Studio Build Tools / C++ workload)"
$extra = if ($env:CMAKE_FLAGS) { $env:CMAKE_FLAGS -split " " } else { @() }
cmake -S $src -B (Join-Path $src "build") @extra
if ($LASTEXITCODE -ne 0) {
    throw "CMake configure failed. If no C++ compiler was found, install the Build Tools:`n" +
          "  winget install Microsoft.VisualStudio.2022.BuildTools --override `"--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --passive`""
}

Write-Host "==> building (llama.cpp takes a few minutes)"
cmake --build (Join-Path $src "build") --config Release --parallel
if ($LASTEXITCODE -ne 0) { throw "build failed" }

$binSrc = Join-Path $src "build\bin"
$binDst = Join-Path $Prefix "bin"
Write-Host "==> installing to $binDst"
New-Item -ItemType Directory -Force -Path $binDst | Out-Null
Copy-Item (Join-Path $binSrc "*") $binDst -Force -Recurse

# Put alpacca on the user PATH so any new terminal finds it.
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if (-not $userPath) { $userPath = "" }
if (($userPath -split ";") -notcontains $binDst) {
    [Environment]::SetEnvironmentVariable("Path", ($userPath.TrimEnd(";") + ";" + $binDst), "User")
    Write-Host "==> added $binDst to your user PATH"
}

Write-Host ""
Write-Host "done. open a NEW terminal (PowerShell or cmd) and try:"
Write-Host "  alpacca doctor"
Write-Host "  alpacca run llama3.2:1b `"hello!`""
