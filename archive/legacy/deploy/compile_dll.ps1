# compile_dll.ps1 — Compile bridge_client.dll using MinGW-w64
#
# Usage: .\deploy\compile_dll.ps1
#
# Searches for a MinGW-w64 gcc installation in common locations.
# Output: bridge\bridge_client.dll

param(
    [string]$CompilerPath = "",   # Specify gcc path manually (optional)
    [string]$OutputDir  = ""      # Override output directory (optional)
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

$SourceFile = Join-Path $ProjectRoot "bridge\bridge_client.c"
$HeaderFile = Join-Path $ProjectRoot "bridge\bridge_client.h"

if (-not $OutputDir) {
    $OutputDir = Join-Path $ProjectRoot "bridge"
}
if (-not (Test-Path -LiteralPath $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

# ── Find gcc ────────────────────────────────────────────────────────────────

function Find-GCC {
    if ($CompilerPath -and (Test-Path -LiteralPath $CompilerPath)) {
        return $CompilerPath
    }

    # 1. Check PATH
    $fromPath = (Get-Command "gcc.exe" -ErrorAction SilentlyContinue)
    if ($fromPath) {
        Write-Host "[OK] Found gcc on PATH: $($fromPath.Source)"
        return $fromPath.Source
    }

    # 2. LLVM-MinGW via winget
    $llvmPaths = @(
        "C:\Program Files\LLVM-MinGW\bin\gcc.exe",
        "C:\Program Files (x86)\LLVM-MinGW\bin\gcc.exe",
        "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\MartinStorsjo.LLVM-MinGW.UCRT_*\bin\gcc.exe",
        "$env:LOCALAPPDATA\Microsoft\WinGet\Links\llvm-mingw-ucrt-*.exe\bin\gcc.exe"
    )
    foreach ($p in $llvmPaths) {
        $resolved = Resolve-Path -LiteralPath $p -ErrorAction SilentlyContinue
        if ($resolved) {
            Write-Host "[OK] Found gcc via winget: $resolved"
            return $resolved.Path
        }
    }

    # 3. MSYS2
    $msys2Paths = @(
        "C:\msys64\mingw64\bin\gcc.exe",
        "C:\msys64\ucrt64\bin\gcc.exe",
        "C:\msys64\clang64\bin\gcc.exe"
    )
    foreach ($p in $msys2Paths) {
        if (Test-Path -LiteralPath $p) {
            Write-Host "[OK] Found gcc via MSYS2: $p"
            return $p
        }
    }

    # 4. WinLibs / manual MinGW
    $manualPaths = @(
        "C:\mingw64\bin\gcc.exe",
        "C:\MinGW\bin\gcc.exe",
        "$env:USERPROFILE\mingw64\bin\gcc.exe"
    )
    foreach ($p in $manualPaths) {
        if (Test-Path -LiteralPath $p) {
            Write-Host "[OK] Found gcc at: $p"
            return $p
        }
    }

    throw @"
MinGW-w64 gcc not found.

Install one of these:
  winget install MartinStorsjo.LLVM-MinGW.UCRT
  winget install BrechtSanders.WinLibs.POSIX.UCRT

Or specify the path manually: .\compile_dll.ps1 -CompilerPath "C:\path\to\gcc.exe"
"@
}

# ── Compile ──────────────────────────────────────────────────────────────────

$gcc = Find-GCC
$outDll = Join-Path $OutputDir "bridge_client.dll"
$cFlags = @("-shared", "-O2", "-o", $outDll, "-lws2_32", "-s")
$src = @($SourceFile)

Write-Host "Compiling bridge_client.dll ..."
Write-Host "  Source:   $SourceFile"
Write-Host "  Output:   $outDll"
Write-Host "  Compiler: $gcc"

$args = $cFlags + $src
& $gcc @args

if ($LASTEXITCODE -ne 0) {
    throw "Compilation failed with exit code $LASTEXITCODE"
}

if (Test-Path -LiteralPath $outDll) {
    $size = (Get-Item -LiteralPath $outDll).Length
    Write-Host "[OK] Compilation successful: $outDll ($size bytes)"
} else {
    throw "DLL not created — unknown error"
}
