param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
$ExcludedModules = @(
    "torch", "pandas", "scipy", "pytest", "matplotlib", "IPython", "jupyter", "notebook", "tensorboard"
)

if ($Clean) {
    Remove-Item -Recurse -Force -LiteralPath (Join-Path $ProjectRoot "build") -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force -LiteralPath (Join-Path $ProjectRoot "dist") -ErrorAction SilentlyContinue
}

$GuiArguments = @("-m", "PyInstaller", "--noconfirm", "--clean", "--onefile", "--windowed", "--name", "YouTubeBiliLocalizer", "--paths", "src")
foreach ($module in $ExcludedModules) { $GuiArguments += @("--exclude-module", $module) }
$GuiArguments += @(
    "--collect-all", "faster_whisper",
    "--collect-all", "ctranslate2",
    "--collect-all", "yt_dlp",
    "--collect-all", "pytesseract",
    "--collect-all", "playwright",
    "--hidden-import", "openai",
    "scripts\launch_gui.py"
)
& python @GuiArguments
if ($LASTEXITCODE -ne 0) {
    throw "GUI EXE build failed with exit code $LASTEXITCODE."
}

$CliArguments = @("-m", "PyInstaller", "--noconfirm", "--clean", "--onefile", "--console", "--name", "yblocalizer", "--paths", "src")
foreach ($module in $ExcludedModules) { $CliArguments += @("--exclude-module", $module) }
$CliArguments += @(
    "--collect-all", "faster_whisper",
    "--collect-all", "ctranslate2",
    "--collect-all", "yt_dlp",
    "--collect-all", "pytesseract",
    "--collect-all", "playwright",
    "--hidden-import", "openai",
    "scripts\launch_cli.py"
)
& python @CliArguments
if ($LASTEXITCODE -ne 0) {
    throw "CLI EXE build failed with exit code $LASTEXITCODE."
}

Write-Host "Built: $ProjectRoot\dist\YouTubeBiliLocalizer.exe"
Write-Host "Built: $ProjectRoot\dist\yblocalizer.exe"
Write-Host "To build the browser workbench EXE: scripts\build_workbench_exe.ps1"
