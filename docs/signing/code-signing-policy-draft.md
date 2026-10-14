# Code signing policy (proposal)

**Status:** preparing application; not approved, sponsored, or currently providing signing.

**Public privacy policy:** [privacy-policy.md](../privacy-policy.md)

## Scope

This proposal applies to Windows release executables and installers produced from the public repository. Local development builds may remain unsigned. If approved, each public release would require manual approval after the release checks pass.

## Current state

`scripts/build_release.ps1` and `scripts/build_installer.ps1` currently accept a local certificate thumbprint and invoke Windows `signtool.exe` with SHA-256 and a DigiCert timestamp. The GitHub Actions release workflow currently builds and uploads artifacts without a SignPath step. This is an implementation description, not a claim that any current public artifact is signed.

## Proposed release controls

1. Build from the tagged commit in GitHub Actions on a pinned Windows runner.
2. Run the agreed build and artifact checks before requesting signing.
3. Submit only the exact generated installer and executable to the approved signing service.
4. Verify Authenticode status, signer identity, timestamp, and SHA-256 after signing.
5. Upload the verified signed artifacts and checksums; retain unsigned intermediates only as private CI artifacts where appropriate.
6. Never store private keys or certificates in the repository, ZIP, release notes, or application package.

## Proposed roles and safeguards

- **Author:** wanghaofu124; **reviewer:** wanghaofu124; **release approver:** wanghaofu124.
- GitHub and SignPath MFA would be required before enabling signing; this is a future requirement, not a claim that it has already been verified.
- Only artifacts built from this repository would be submitted. Vendor DLLs would remain untouched and would require separate license review.
- Any planned Foundation attribution is subject to the Foundation's approval; there is no existing sponsorship or endorsement claim.

## Pending security and eligibility decisions

- SignPath approval and project eligibility are not yet granted.
- The shipped cuDNN DLL is proprietary NVIDIA software and must be cleared before adopting this workflow.
- The current public CI history has failed release/test runs and must be repaired or the signing gate must reject those commits.
- Any workflow change requires review before publication. This draft does not authorize signing, upload, release, or terms acceptance.
