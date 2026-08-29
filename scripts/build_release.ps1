param(
    [switch]$Clean,
    [string]$Version = "0.2.6",
    [string]$AppName = "YouTubeBiliLocalizer",
    [string]$PythonExe = "",
    [string]$SigningThumbprint = "",
    [switch]$RequireSignature
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

if (-not $PythonExe) {
    $uv = Get-Command uv -ErrorAction SilentlyContinue
    if ($uv) {
        $PythonExe = (& $uv.Source python find 3.12).Trim()
    }
    else {
        $candidate = Get-Command python -ErrorAction SilentlyContinue
        if ($candidate) { $PythonExe = $candidate.Source }
    }
}
if (-not $PythonExe -or -not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python 3.12 was not found. Install it or pass -PythonExe C:\path\to\python.exe."
}
$PythonVersion = (& $PythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
if ($PythonVersion -ne "3.12") {
    throw "Release builds require Python 3.12; selected interpreter is $PythonVersion at $PythonExe."
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
$BuildPython = Join-Path $VenvRoot "Scripts\python.exe"
if (Test-Path -LiteralPath $BuildPython) {
    $ExistingVersion = (& $BuildPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
    if ($ExistingVersion -ne "3.12") {
        Remove-Item -LiteralPath $VenvRoot -Recurse -Force
    }
}
if (-not (Test-Path -LiteralPath $BuildPython)) {
    & $PythonExe -m venv $VenvRoot
}
& $BuildPython -m pip install "pip==26.2.1"
& $BuildPython -m pip install -r (Join-Path $ProjectRoot "requirements-build.lock")
& $BuildPython -m pip install --no-deps -e $ProjectRoot
if ($LASTEXITCODE -ne 0) { throw "isolated build dependency installation failed" }

$Excluded = @(
    "torch", "torchvision", "torchaudio", "pyarrow", "pandas", "scipy",
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
    "--collect-all", "playwright",
    "--collect-all", "nodriver",
    "--collect-submodules", "yt_dlp_plugins",
    "--hidden-import", "yt_dlp_plugins.extractor.getpot_wpc",
    "--copy-metadata", "yt-dlp-getpot-wpc",
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
$OriginalPath = $env:PATH
$PythonRoot = Split-Path -Parent $PythonExe
$env:PATH = @(
    (Join-Path $VenvRoot "Scripts"),
    $PythonRoot,
    (Join-Path $env:SystemRoot "System32"),
    $env:SystemRoot,
    (Join-Path $env:SystemRoot "System32\Wbem")
) -join [System.IO.Path]::PathSeparator
try {
    & $BuildPython @Arguments
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }
}
finally {
    $env:PATH = $OriginalPath
}

$CollectToc = Join-Path $BuildRoot "pyinstaller\$AppName\COLLECT-00.toc"
if (Test-Path -LiteralPath $CollectToc) {
    $TocText = Get-Content -Raw -LiteralPath $CollectToc
    if ($TocText -match "(?i)[\\/]Java[\\/].*MSVCP140\.dll") {
        throw "Unsafe MSVCP140.dll source detected in release build: Java runtime"
    }
}

$AppExe = Join-Path $AppRoot "$AppName.exe"
$OnnxRuntimeDll = Join-Path $AppRoot "_internal\onnxruntime\capi\onnxruntime_providers_shared.dll"
if (-not (Test-Path -LiteralPath $OnnxRuntimeDll)) {
    throw "ONNX Runtime is missing from the release; Whisper VAD cannot run."
}
$NativeSmoke = Start-Process -FilePath $AppExe -ArgumentList @("--whisper-worker", "--native-smoke-test") -WindowStyle Hidden -Wait -PassThru
if ($NativeSmoke.ExitCode -ne 0) {
    throw "Frozen Whisper native dependency smoke test failed with exit code $($NativeSmoke.ExitCode)."
}
if ($SigningThumbprint) {
    $SignTool = Get-Command signtool.exe -ErrorAction SilentlyContinue
    if (-not $SignTool) {
        $SignTool = Get-ChildItem "${env:ProgramFiles(x86)}\Windows Kits\10\bin" -Filter signtool.exe -Recurse -ErrorAction SilentlyContinue | Sort-Object FullName -Descending | Select-Object -First 1
    }
    if (-not $SignTool) { throw "signtool.exe was not found; install the Windows SDK before signing." }
    $SignToolPath = if ($SignTool.Source) { $SignTool.Source } else { $SignTool.FullName }
    & $SignToolPath sign /sha1 $SigningThumbprint /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 $AppExe
    if ($LASTEXITCODE -ne 0) { throw "Code signing failed." }
}
$Signature = Get-AuthenticodeSignature -LiteralPath $AppExe
if ($RequireSignature -and $Signature.Status -ne "Valid") {
    throw "Release signature is $($Signature.Status). Provide -SigningThumbprint or remove -RequireSignature for local test builds."
}
if ($Signature.Status -ne "Valid") {
    Write-Warning "Release is unsigned. Public releases should use -SigningThumbprint and -RequireSignature."
}

foreach ($forbidden in @("torch", "pyarrow")) {
    if (Get-ChildItem -LiteralPath $AppRoot -Recurse -Directory -ErrorAction SilentlyContinue | Where-Object Name -EQ $forbidden) {
        throw "Forbidden package leaked into release: $forbidden"
    }
}
$size = (Get-ChildItem -LiteralPath $AppRoot -Recurse -File | Measure-Object Length -Sum).Sum
$limit = 400MB
if ($size -gt $limit) { throw "Release is $([math]::Round($size / 1MB, 1)) MB; limit is 400 MB" }

New-Item -ItemType Directory -Path $ReleaseRoot -Force | Out-Null
Compress-Archive -LiteralPath $AppRoot -DestinationPath $ZipPath -CompressionLevel Optimal -Force
$hash = (Get-FileHash -LiteralPath $ZipPath -Algorithm SHA256).Hash.ToLowerInvariant()
Set-Content -LiteralPath "$ZipPath.sha256" -Value "$hash  $(Split-Path $ZipPath -Leaf)" -Encoding ascii
Write-Host "Release: $ZipPath"
Write-Host "Unpacked size: $([math]::Round($size / 1MB, 1)) MB"
