param(
    [string]$IdentityConfig = "packaging\msix\store-identity.json",
    [string]$SdkMakeAppx = "",
    [string]$Version = "",
    [switch]$PreviewIdentity
)
$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$DistRoot = Join-Path $ProjectRoot "dist\YouTubeBiliLocalizer"
$Template = Join-Path $ProjectRoot "packaging\msix\AppxManifest.template.xml"
$OutputRoot = Join-Path $ProjectRoot "releases\msix"
$RunId = [guid]::NewGuid().ToString('N')
$StageRoot = Join-Path $ProjectRoot ".msix-staging\$RunId\stage"
$UnpackRoot = Join-Path $ProjectRoot ".msix-staging\$RunId\unpack"

if (-not (Test-Path -LiteralPath $DistRoot)) { throw "Missing PyInstaller input: $DistRoot" }
if (-not (Test-Path -LiteralPath (Join-Path $DistRoot "YouTubeBiliLocalizer.exe"))) { throw "Missing packaged EXE." }
if (-not $SdkMakeAppx) {
    $candidate = Get-Command makeappx.exe -ErrorAction SilentlyContinue
    if ($candidate) { $SdkMakeAppx = $candidate.Source }
    else {
        $candidate = Get-ChildItem "${env:ProgramFiles(x86)}\Windows Kits\10\bin" -Recurse -Filter makeappx.exe -ErrorAction SilentlyContinue | Where-Object FullName -match '\\x64\\makeappx\.exe$' | Sort-Object FullName -Descending | Select-Object -First 1
        if ($candidate) { $SdkMakeAppx = $candidate.FullName }
    }
}
if (-not $SdkMakeAppx -or -not (Test-Path -LiteralPath $SdkMakeAppx)) { throw "makeappx.exe was not found. Pass -SdkMakeAppx." }
$versionText = Get-Content -Raw -LiteralPath (Join-Path $ProjectRoot 'src\yblocalizer\__init__.py')
$match = [regex]::Match($versionText, '__version__\s*=\s*["'']([0-9]+\.[0-9]+\.[0-9]+)["'']')
if (-not $match.Success) { throw "Could not read application version." }
$SourceVersion = $match.Groups[1].Value
if ([string]::IsNullOrWhiteSpace($Version)) { $Version = $SourceVersion }
if ($Version -ne $SourceVersion) { throw "Version $Version does not match source application version $SourceVersion." }
if ($Version -notmatch '^([0-9]+)\.([0-9]+)\.([0-9]+)$') { throw "Version must be major.minor.patch." }
$v = $Version.Split('.') | ForEach-Object {[int]$_}
if (($v[0] -ge 65535) -or ($v[1] -gt 65535) -or ($v[2] -gt 65535)) { throw "Version components exceed the MSIX 65535 limit after major-version mapping." }
$PackageVersion = "$($v[0]+1).$($v[1]).$($v[2]).0"
$identity = if ($PreviewIdentity) {
    @{ identityName="YouTubeBiliLocalizer.Preview"; publisher="CN=UNSIGNED-PREVIEW"; publisherDisplayName="UNSIGNED-PREVIEW"; displayName="YouTube Bili Localizer (Preview)" }
} else {
    if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot $IdentityConfig))) { throw "Identity config missing: $IdentityConfig" }
    Get-Content -Raw -LiteralPath (Join-Path $ProjectRoot $IdentityConfig) | ConvertFrom-Json
}
foreach ($key in @('identityName','publisher','publisherDisplayName','displayName')) { if ([string]::IsNullOrWhiteSpace([string]$identity.$key) -or (!$PreviewIdentity -and [string]$identity.$key -match 'RESERVE_IN_PARTNER_CENTER|RESERVED_PUBLISHER|YOUR_PUBLISHER')) { throw "Missing real identity field: $key" } }
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
$PackageName = if ($PreviewIdentity) { "YouTubeBiliLocalizer-v$Version-${PackageVersion}-UNSIGNED-PREVIEW_x64.msix" } else { "YouTubeBiliLocalizer-v$Version-${PackageVersion}-UNSIGNED-SUBMISSION_x64.msix" }
$PackagePath = Join-Path $OutputRoot $PackageName
if (Test-Path -LiteralPath $PackagePath) { throw "Refusing to overwrite existing output: $PackagePath" }
New-Item -ItemType Directory -Path $StageRoot -Force | Out-Null
Get-ChildItem -LiteralPath $DistRoot -Force | Copy-Item -Destination $StageRoot -Recurse -Force
$distFiles = Get-ChildItem -LiteralPath $DistRoot -Recurse -File | ForEach-Object { $_.FullName.Substring($DistRoot.Length + 1) }
foreach ($relative in $distFiles) {
    $distHash = (Get-FileHash -LiteralPath (Join-Path $DistRoot $relative) -Algorithm SHA256).Hash
    $stageHash = (Get-FileHash -LiteralPath (Join-Path $StageRoot $relative) -Algorithm SHA256).Hash
    if ($distHash -ne $stageHash) { throw "Dist-to-stage hash mismatch: $relative" }
}
$sensitive = Get-ChildItem -LiteralPath $StageRoot -Recurse -File | Where-Object {
    $relative = $_.FullName.Substring($StageRoot.Length + 1) -replace '\\','/'
    $relative -match '(?i)(^|/)(\.env(\..*)?|Cookies|cookies\.(txt|json|cookie)|netscape-cookie|\.cookie|bilibili-profile|bilibili-edge-profile|douyin-profile|logs?|outputs?|profiles?)(/|$)'
}
if ($sensitive) { throw "Refusing to package user or credential material: $($sensitive[0].FullName)" }
New-Item -ItemType Directory -Path (Join-Path $StageRoot 'Assets') -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $ProjectRoot 'assets\app-icon-source.png') -Destination (Join-Path $StageRoot 'Assets\StoreLogo.png')
Copy-Item -LiteralPath (Join-Path $ProjectRoot 'assets\app-icon-source.png') -Destination (Join-Path $StageRoot 'Assets\Square44x44Logo.png')
Copy-Item -LiteralPath (Join-Path $ProjectRoot 'assets\app-icon-source.png') -Destination (Join-Path $StageRoot 'Assets\Square150x150Logo.png')
Add-Type -AssemblyName System.Drawing
function Resize-Png([string]$Source, [string]$Destination, [int]$Size) {
    $sourceImage = [System.Drawing.Image]::FromFile($Source)
    try {
        $bitmap = New-Object System.Drawing.Bitmap($Size, $Size)
        try {
            $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
            try {
                $graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
                $graphics.DrawImage($sourceImage, 0, 0, $Size, $Size)
            } finally { $graphics.Dispose() }
            $bitmap.Save($Destination, [System.Drawing.Imaging.ImageFormat]::Png)
        } finally { $bitmap.Dispose() }
    } finally { $sourceImage.Dispose() }
}
$iconSource = Join-Path $ProjectRoot 'assets\app-icon-source.png'
Resize-Png $iconSource (Join-Path $StageRoot 'Assets\StoreLogo.png') 50
Resize-Png $iconSource (Join-Path $StageRoot 'Assets\Square44x44Logo.png') 44
Resize-Png $iconSource (Join-Path $StageRoot 'Assets\Square150x150Logo.png') 150
$xml = New-Object System.Xml.XmlDocument
$xml.PreserveWhitespace = $true
$xml.Load($Template)
$identityNode = $xml.SelectSingleNode("/*[local-name()='Package']/*[local-name()='Identity']")
$identityNode.SetAttribute('Name',[string]$identity.identityName)
$identityNode.SetAttribute('Publisher',[string]$identity.publisher)
$identityNode.SetAttribute('Version',$PackageVersion)
$properties = $xml.SelectSingleNode("/*[local-name()='Package']/*[local-name()='Properties']")
$properties.SelectSingleNode("*[local-name()='DisplayName']").InnerText = [string]$identity.displayName
$properties.SelectSingleNode("*[local-name()='PublisherDisplayName']").InnerText = [string]$identity.publisherDisplayName
$visual = $xml.SelectSingleNode("/*[local-name()='Package']/*[local-name()='Applications']/*[local-name()='Application']/*[local-name()='VisualElements']")
$visual.SetAttribute('DisplayName',[string]$identity.displayName)
$xml.Save((Join-Path $StageRoot 'AppxManifest.xml'))
& $SdkMakeAppx pack /d $StageRoot /p $PackagePath /o
if ($LASTEXITCODE -ne 0) { throw "MakeAppx pack failed." }
$unpack = $UnpackRoot
New-Item -ItemType Directory -Path $unpack -Force | Out-Null
& $SdkMakeAppx unpack /p $PackagePath /d $unpack /o
if ($LASTEXITCODE -ne 0) { throw "MakeAppx unpack validation failed." }
$manifest = Get-Content -Raw -LiteralPath (Join-Path $unpack 'AppxManifest.xml')
$manifestXml = New-Object System.Xml.XmlDocument
$manifestXml.Load((Join-Path $unpack 'AppxManifest.xml'))
$manifestIdentity = $manifestXml.SelectSingleNode("/*[local-name()='Package']/*[local-name()='Identity']")
$manifestProperties = $manifestXml.SelectSingleNode("/*[local-name()='Package']/*[local-name()='Properties']")
$manifestApp = $manifestXml.SelectSingleNode("/*[local-name()='Package']/*[local-name()='Applications']/*[local-name()='Application']")
$manifestVisual = $manifestApp.SelectSingleNode("*[local-name()='VisualElements']")
$manifestCapability = $manifestXml.SelectSingleNode("/*[local-name()='Package']/*[local-name()='Capabilities']/*[local-name()='Capability']")
if ($manifestIdentity.GetAttribute('Name') -ne [string]$identity.identityName -or $manifestIdentity.GetAttribute('Publisher') -ne [string]$identity.publisher -or $manifestIdentity.GetAttribute('Version') -ne $PackageVersion -or $manifestIdentity.GetAttribute('ProcessorArchitecture') -ne 'x64') { throw "Manifest identity mismatch." }
if ($manifestProperties.SelectSingleNode("*[local-name()='DisplayName']").InnerText -ne [string]$identity.displayName -or $manifestProperties.SelectSingleNode("*[local-name()='PublisherDisplayName']").InnerText -ne [string]$identity.publisherDisplayName -or $manifestVisual.GetAttribute('DisplayName') -ne [string]$identity.displayName) { throw "Manifest display fields mismatch." }
if ($manifestApp.GetAttribute('Executable') -ne 'YouTubeBiliLocalizer.exe' -or $manifestApp.GetAttribute('EntryPoint') -ne 'Windows.FullTrustApplication' -or $manifestCapability.GetAttribute('Name') -ne 'runFullTrust') { throw "Manifest executable or capability mismatch." }
$sourceFiles = Get-ChildItem -LiteralPath $StageRoot -Recurse -File | Where-Object Name -ne 'AppxManifest.xml' | ForEach-Object { $_.FullName.Substring($StageRoot.Length + 1) }
$unpackedFiles = Get-ChildItem -LiteralPath $unpack -Recurse -File | Where-Object { $_.Name -notin @('AppxManifest.xml','AppxBlockMap.xml','[Content_Types].xml') } | ForEach-Object { $_.FullName.Substring($unpack.Length + 1) }
if ((@($sourceFiles | Sort-Object) -join "`n") -ne (@($unpackedFiles | Sort-Object) -join "`n")) { throw "Payload file set changed during pack/unpack." }
foreach ($relative in $sourceFiles) {
    $sourceHash = (Get-FileHash -LiteralPath (Join-Path $StageRoot $relative) -Algorithm SHA256).Hash
    $unpackedHash = (Get-FileHash -LiteralPath (Join-Path $unpack $relative) -Algorithm SHA256).Hash
    if ($sourceHash -ne $unpackedHash) { throw "Payload hash mismatch: $relative" }
}
foreach ($logo in @(@('StoreLogo.png',50),@('Square44x44Logo.png',44),@('Square150x150Logo.png',150))) {
    $image = [System.Drawing.Image]::FromFile((Join-Path $unpack "Assets\$($logo[0])"))
    try { if ($image.Width -ne $logo[1] -or $image.Height -ne $logo[1]) { throw "Logo dimensions invalid: $($logo[0])" } } finally { $image.Dispose() }
}
$hash = (Get-FileHash -LiteralPath $PackagePath -Algorithm SHA256).Hash.ToLowerInvariant()
$summary = [ordered]@{ package=$PackagePath; sourceVersion=$SourceVersion; version=$PackageVersion; sha256=$hash; identityName=[string]$identity.identityName; publisher=[string]$identity.publisher; preview=[bool]$PreviewIdentity; makeappx=$SdkMakeAppx; validated=@('dist-to-stage SHA256','pack','unpack','manifest identity/executable/capability','payload file set and SHA256','logo dimensions','sensitive filename scan') }
$summary | ConvertTo-Json | Set-Content -LiteralPath "$PackagePath.json" -Encoding UTF8
Write-Host "MSIX: $PackagePath"
Write-Host "SHA256: $hash"
