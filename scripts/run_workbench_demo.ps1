param(
    [int]$Port = 8765
)

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$FrontendDir = Join-Path $ProjectRoot "frontend"
$FrontendBuild = Join-Path $FrontendDir "dist\index.html"

Set-Location $ProjectRoot
if (-not (Test-Path $FrontendBuild)) {
    Write-Host "Building the workbench frontend..."
    Push-Location $FrontendDir
    try {
        npm.cmd run build
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    finally {
        Pop-Location
    }
}

$env:PYTHONPATH = "$ProjectRoot\src"
python -m yblocalizer.workbench_api --frontend "$FrontendDir\dist" --port $Port
