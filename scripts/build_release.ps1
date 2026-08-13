param(
    [switch]$Clean,
    [string]$Version = "0.2.0",
    [string]$AppName = "YouTubeBiliLocalizer"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$BuildRoot = Join-Path $ProjectRoot ".release-build"
$VenvRoot = Join-Path $BuildRoot "venv"
$ReleaseRoot = Join-Path $ProjectRoot "releases"
$DistRoot = Join-Path $ProjectRoot "dist"
$AppRoot = Join-Path $DistRoot $AppName
$ZipPath = Join-Path $ReleaseRoot "$AppName-v$Version-windows-x64.zip"

function Assert-WithinProject([string]$Target) {
    $full = [System.IO.Path]::GetFullPath($Target)
    if (-not $full.StartsWith($ProjectRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to modify a path outside the project: $full"
    }
}

foreach ($target in @($BuildRoot, $AppRoot, $ZipPath)) { Assert-WithinProject $target }

if ($Clean) {
    if (Test-Path -LiteralPath $BuildRoot) { Remove-Item -LiteralPath $BuildRoot -Recurse -Force }
    if (Test-Path -LiteralPath $AppRoot) { Remove-Item -LiteralPath $AppRoot -Recurse -Force }
    if (Test-Path -LiteralPath $ZipPath) { Remove-Item -LiteralPath $ZipPath -Force }
}

Push-Location (Join-Path $ProjectRoot "frontend")
try {
    if (-not (Test-Path -LiteralPath "node_modules")) {
        npm.cmd ci
        if ($LASTEXITCODE -ne 0) { throw "npm ci failed" }
    }
    npm.cmd run build
    if ($LASTEXITCODE -ne 0) { throw "frontend build failed" }
}
finally { Pop-Location }

New-Item -ItemType Directory -Path $BuildRoot -Force | Out-Null
if (-not (Test-Path -LiteralPath (Join-Path $VenvRoot "Scripts\python.exe"))) {
    python -m venv $VenvRoot
}
$BuildPython = Join-Path $VenvRoot "Scripts\python.exe"
& $BuildPython -m pip install --upgrade pip wheel setuptools pyinstaller
& $BuildPython -m pip install -e $ProjectRoot
if ($LASTEXITCODE -ne 0) { throw "isolated build dependency installation failed" }

$Excluded = @(
    "torch", "torchvision", "torchaudio", "onnxruntime", "pyarrow", "pandas", "scipy",
    "matplotlib", "pytest", "IPython", "jupyter", "notebook", "tensorboard"
)
$Arguments = @(
    "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir", "--windowed",
    "--icon", (Join-Path $ProjectRoot "assets\app-icon.ico"),
    "--name", $AppName, "--paths", (Join-Path $ProjectRoot "src")
)
foreach ($module in $Excluded) { $Arguments += @("--exclude-module", $module) }
$Arguments += @(
    "--collect-submodules", "faster_whisper",
    "--collect-data", "faster_whisper",
    "--collect-binaries", "ctranslate2",
    "--collect-submodules", "webview",
    "--hidden-import", "webview.platforms.edgechromium",
    "--hidden-import", "openai",
    "--add-data", "$ProjectRoot\frontend\dist;frontend\dist",
    "--add-data", "$ProjectRoot\demo\authorized-demo-10s.mp4;demo",
    "--add-data", "$ProjectRoot\demo\artifacts;demo\artifacts",
    "--distpath", $DistRoot,
    "--workpath", (Join-Path $BuildRoot "pyinstaller"),
    "--specpath", $BuildRoot,
    (Join-Path $ProjectRoot "scripts\launch_workbench.py")
)
& $BuildPython @Arguments
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

foreach ($forbidden in @("torch", "onnxruntime", "pyarrow")) {
    if (Get-ChildItem -LiteralPath $AppRoot -Recurse -Directory -ErrorAction SilentlyContinue | Where-Object Name -EQ $forbidden) {
        throw "Forbidden package leaked into release: $forbidden"
    }
}
$size = (Get-ChildItem -LiteralPath $AppRoot -Recurse -File | Measure-Object Length -Sum).Sum
$limit = 350MB
if ($size -gt $limit) { throw "Release is $([math]::Round($size / 1MB, 1)) MB; limit is 350 MB" }

New-Item -ItemType Directory -Path $ReleaseRoot -Force | Out-Null
Compress-Archive -LiteralPath $AppRoot -DestinationPath $ZipPath -CompressionLevel Optimal -Force
$hash = (Get-FileHash -LiteralPath $ZipPath -Algorithm SHA256).Hash.ToLowerInvariant()
Set-Content -LiteralPath "$ZipPath.sha256" -Value "$hash  $(Split-Path $ZipPath -Leaf)" -Encoding ascii
Write-Host "Release: $ZipPath"
Write-Host "Unpacked size: $([math]::Round($size / 1MB, 1)) MB"
