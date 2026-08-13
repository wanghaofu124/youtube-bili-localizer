param(
    [switch]$Clean,
    [string]$AppName = "YouTubeBiliLocalizerWorkbench"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Test-Path "frontend\dist\index.html")) {
    Push-Location frontend
    try {
        npm.cmd run build
        if ($LASTEXITCODE -ne 0) { throw "Frontend build failed." }
    }
    finally {
        Pop-Location
    }
}

if ($Clean) {
    Remove-Item -Recurse -Force -LiteralPath (Join-Path $ProjectRoot "build\$AppName") -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force -LiteralPath (Join-Path $ProjectRoot "dist\$AppName") -ErrorAction SilentlyContinue
    # Remove the legacy single-file build so users do not accidentally launch the slow version.
    Remove-Item -Force -LiteralPath (Join-Path $ProjectRoot "dist\$AppName.exe") -ErrorAction SilentlyContinue
}

$ExcludedModules = @("torch", "torchvision", "torchaudio", "onnxruntime", "pyarrow", "pandas", "scipy", "pytest", "matplotlib", "IPython", "jupyter", "notebook", "tensorboard")
$Arguments = @("-m", "PyInstaller", "--noconfirm", "--clean", "--onedir", "--windowed", "--icon", "assets\\app-icon.ico", "--name", $AppName, "--paths", "src")
foreach ($module in $ExcludedModules) { $Arguments += @("--exclude-module", $module) }
$Arguments += @(
    "--collect-submodules", "faster_whisper",
    "--collect-data", "faster_whisper",
    "--collect-binaries", "ctranslate2",
    "--collect-submodules", "webview",
    "--hidden-import", "webview.platforms.edgechromium",
    "--hidden-import", "openai",
    "--add-data", "frontend\dist;frontend\dist",
    "--add-data", "demo\authorized-demo-10s.mp4;demo",
    "--add-data", "demo\artifacts;demo\artifacts",
    "scripts\launch_workbench.py"
)

& python @Arguments
if ($LASTEXITCODE -ne 0) { throw "Workbench EXE build failed with exit code $LASTEXITCODE." }
Write-Host "Built: $ProjectRoot\dist\$AppName\$AppName.exe"
