# MSIX / Microsoft Store preparation

This directory is an independent Store packaging layer. The existing Inno Setup, ZIP release, PyInstaller build, and runtime dependency behavior remain unchanged.

After the product is reserved in Partner Center, copy `store-identity.example.json` to `store-identity.json` (the latter is ignored) and replace all four values exactly as shown under **Product management → Product identity**. The package identity values are case-sensitive. PFN is derived by the Store and is not written to the manifest.

Build from the repository root:

```powershell
scripts\build_msix.ps1 -IdentityConfig packaging\msix\store-identity.json
```

Before reservation, create only an explicit local preview:

```powershell
scripts\build_msix.ps1 -PreviewIdentity
```

The preview is deliberately named `UNSIGNED-PREVIEW`; it is not a Store submission and is not installed automatically. The formal build refuses placeholder identity values, existing output files, and missing package input. The script packs and unpacks with the Windows SDK `makeappx.exe`, checks the manifest, payload hashes, and logo dimensions, and writes a JSON summary beside the package.

Store packages are re-signed by Microsoft after certification. A local unsigned package is suitable for structure validation only.
