param(
    [string]$Version = "0.2.6",
    [string]$SigningThumbprint = "",
    [switch]$RequireSignature
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$AppRoot = Join-Path $ProjectRoot "dist\YouTubeBiliLocalizer"
$LanguageRoot = Join-Path $ProjectRoot ".release-build\installer-languages"
$ChineseLanguage = Join-Path $LanguageRoot "ChineseSimplified.isl"
$ChineseLanguageUrl = "https://raw.githubusercontent.com/jrsoftware/issrc/3cfb0e5632828e0dd9b49400a185834e8f1ab570/Files/Languages/ChineseSimplified.isl"
$ChineseLanguageSha256 = "e0b0b350e2245f3c5e65586dfe43d574f6e7f06f2261149aba284954b3fc9a8d"
$Compiler = Get-Command iscc.exe -ErrorAction SilentlyContinue
if (-not $Compiler) {
    $known = @(
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
    )
    $Compiler = $known | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}
if (-not $Compiler) {
    throw "Inno Setup 6 was not found. Install JRSoftware.InnoSetup, then run this script again."
}
if (-not (Test-Path -LiteralPath (Join-Path $AppRoot "YouTubeBiliLocalizer.exe"))) {
    throw "Build the onedir application first with scripts\build_release.ps1."
}
New-Item -ItemType Directory -Path $LanguageRoot -Force | Out-Null
if (-not (Test-Path -LiteralPath $ChineseLanguage) -or
    (Get-FileHash -LiteralPath $ChineseLanguage -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ChineseLanguageSha256) {
    Invoke-WebRequest -Uri $ChineseLanguageUrl -OutFile $ChineseLanguage -UseBasicParsing
}
$ActualLanguageHash = (Get-FileHash -LiteralPath $ChineseLanguage -Algorithm SHA256).Hash.ToLowerInvariant()
if ($ActualLanguageHash -ne $ChineseLanguageSha256) {
    throw "Chinese installer language file checksum mismatch: $ActualLanguageHash"
}
$CompilerPath = if ($Compiler.Source) { $Compiler.Source } else { [string]$Compiler }
& $CompilerPath "/DMyAppVersion=$Version" "/DChineseLanguageFile=$ChineseLanguage" (Join-Path $ProjectRoot "installer\YouTubeBiliLocalizer.iss")
if ($LASTEXITCODE -ne 0) { throw "Inno Setup build failed." }

$Installer = Join-Path $ProjectRoot "releases\YouTubeBiliLocalizer-v$Version-setup.exe"
if ($SigningThumbprint) {
    $SignTool = Get-Command signtool.exe -ErrorAction SilentlyContinue
    if (-not $SignTool) { throw "signtool.exe was not found; install the Windows SDK before signing." }
    & $SignTool.Source sign /sha1 $SigningThumbprint /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 $Installer
    if ($LASTEXITCODE -ne 0) { throw "Installer signing failed." }
}
$Signature = Get-AuthenticodeSignature -LiteralPath $Installer
if ($RequireSignature -and $Signature.Status -ne "Valid") {
    throw "Installer signature is $($Signature.Status). A signed public release is required."
}
$Hash = (Get-FileHash -LiteralPath $Installer -Algorithm SHA256).Hash.ToLowerInvariant()
$HashFile = "$Installer.sha256"
"$Hash  $(Split-Path -Leaf $Installer)" | Set-Content -LiteralPath $HashFile -Encoding ascii
if ($Signature.Status -ne "Valid") {
    Write-Warning "Installer signature is $($Signature.Status). Use -SigningThumbprint and -RequireSignature for a public release."
}
Write-Host "Installer: $Installer"
Write-Host "SHA256:    $HashFile"
