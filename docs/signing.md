# Code signing

**Status: off.** Releases are built unsigned. The release workflow
(`.github/workflows/release.yml`) signs only after the steps below are done,
and it switches on by itself once the repository variable `AS_ACCOUNT` is
set. Until then every run is the job `build`, which builds, tests and
publishes the unsigned zip and setup.exe.

Signing uses [Azure Artifact Signing](https://learn.microsoft.com/azure/artifact-signing/)
(formerly Trusted Signing). GitHub proves to Azure which workflow is asking
(OpenID Connect), so no password or key is stored in GitHub.

Marks used below: **[verified]** checked against Microsoft's or GitHub's
documentation or by a dry run on 2026-09-18; **[unverified]** not yet
checked, or depends on things only a real signed release can show.

## What gets signed

With signing on, Tauri signs every file it bundles, through the sign command
the workflow gives it, and the workflow signs one more for the zip. A dry
run with a stand-in sign command confirmed the list **[verified]**:

| File | Signed where |
|---|---|
| `fidim.exe`, `fidim-dg.exe` | in place in `target\release`, so the zip gets signed copies too |
| `llama-fidim.exe` | inside the installer, after Tauri patches it |
| `NSISdl.dll`, `StartMenu.dll`, `System.dll`, `nsDialogs.dll`, `nsis_tauri_utils.dll` | the installer's NSIS plug-ins |
| `uninstall.exe` | while the installer is compiled (NSIS `!uninstfinalize`) |
| `Llama FIDIM_<version>_x64-setup.exe` | last |
| `llama-fidim.exe` for the zip | by the workflow: Tauri puts the unsigned GUI back in `target\release` after bundling |

That is 11 signatures per release.

## Turning it on

In this order. Steps 1 to 5 happen in the Azure portal and cannot be
scripted; identity validation takes days.

1. **Check eligibility.** Artifact Signing issues Public Trust certificates
   to organizations in the US, Canada, the EU, the UK, Australia, New
   Zealand, Japan, South Korea, Singapore, Switzerland, Norway and Israel,
   and to individual developers in the US and Canada only
   **[verified: Microsoft quickstart, 2026-09-18]**. Elsewhere, see
   [Other routes](#other-routes).
2. **A paid Azure subscription** (Pay-As-You-Go; free and trial
   subscriptions are refused). For an individual identity the billing
   account must have Account Type = Individual, and its legal name and
   sold-to address must match your government ID exactly; fix them under
   Billing first **[verified]**.
3. **An Artifact Signing account.** Register the resource provider
   `Microsoft.CodeSigning`, then create an Artifact Signing account in one
   region. The Basic tier is listed at $9.99 per month for 5,000 signatures,
   billed without proration **[unverified: price]**. Note the account name
   and the region's endpoint, for example `https://eus.codesigning.azure.net`
   for East US; the endpoint must match the account's region or signing
   fails with 403 **[verified]**.
4. **Identity validation.** Give yourself the role *Artifact Signing
   Identity Verifier* on the account, then validate in the portal: for an
   individual, a government photo ID and a selfie through AU10TIX and
   Microsoft Authenticator (Verified ID), sometimes a proof of address. It
   takes 1 to 20 business days, and has to be renewed each cycle (renewal
   opens 60 days before expiry) **[verified]**.
5. **A certificate profile** of type *Public Trust*. Its subject is your
   validated legal name, with city, state and country; street and postal
   code are optional. Every signed file shows this name publicly.
6. **An app registration for GitHub** in Microsoft Entra ID (it becomes a
   service principal). Under *Certificates & secrets* > *Federated
   credentials*, add one for GitHub Actions:
   - Issuer: `https://token.actions.githubusercontent.com`
   - Subject: `repo:Dixon-Cider/llama-fidim:environment:release`
   - Audience: `api://AzureADTokenExchange`

   Because the subject names the `release` environment, pull requests,
   forks and `ci.yml` can never obtain a signing token. Then give the
   service principal (not your own user) the role *Artifact Signing
   Certificate Profile Signer* on the signing account.
7. **The `release` environment on GitHub** (repository Settings >
   Environments > New environment, named exactly `release`):
   - Required reviewers: yourself. Every signed release then waits for your
     approval.
   - Deployment branches and tags: *Selected branches and tags*, with the
     tag rule `v*`.
   - Environment secrets (repository secrets work as well):

     | Secret | Value |
     |---|---|
     | `AZURE_CLIENT_ID` | the app registration's Application (client) ID |
     | `AZURE_TENANT_ID` | your Directory (tenant) ID |
     | `AZURE_SUBSCRIPTION_ID` | the subscription holding the signing account |

8. **Repository variables** (Settings > Secrets and variables > Actions >
   Variables). They must be repository variables, not environment ones:
   `AS_ACCOUNT` decides which job runs before any environment applies.

   | Variable | Value |
   |---|---|
   | `AS_ENDPOINT` | the account's endpoint, e.g. `https://eus.codesigning.azure.net` |
   | `AS_ACCOUNT` | the Artifact Signing account name; setting it turns signing on |
   | `AS_PROFILE` | the certificate profile name |
   | `AS_SIGNER` | optional: your certificate's CN (the validated legal name); the workflow then rejects a signature by anyone else |

9. **Two-factor authentication** on the GitHub account.
10. **Cut a release** as usual (`scripts\release.ps1`, then push the tag),
    and approve the `release` deployment when GitHub asks. The run is the
    job `build-signed`; check its steps *Verify signatures* and *Test the
    installer* (which requires every installed `.exe` and `.dll` to be
    signed).
11. **Update the README's install section**, which says releases are
    unsigned.

To turn signing off again, delete the `AS_ACCOUNT` variable.

## How the workflow signs

- A tag push runs the job `build-signed` when `AS_ACCOUNT` is set, and the
  job `build` otherwise; a manual run from a branch is always `build`. Both
  run the same steps; `build-signed` adds the `release` environment and the
  `id-token: write` permission that OpenID Connect needs.
- *Prepare signing* downloads signtool (NuGet package
  `Microsoft.Windows.SDK.BuildTools` 10.0.26100.4188) and Artifact Signing's
  signtool plug-in (`Microsoft.ArtifactSigning.Client` 1.0.128, file
  `bin\x64\Azure.CodeSigning.Dlib.dll`) from nuget.org and checks both
  against pinned SHA-512 hashes. These are the versions
  `azure/artifact-signing-action` v2 uses. The plug-in needs the .NET 8
  runtime; GitHub's Windows Server 2025 image lists .NET 8.0 runtimes
  (image of 2026-09-07) **[verified]**, and `windows-latest` is taken to
  be that image **[unverified]**. It writes `metadata.json` (endpoint, account, profile)
  with every Azure credential type excluded except the Azure CLI, and a
  Tauri config whose `bundle.windows.signCommand` runs
  `signtool sign /fd SHA256 /tr http://timestamp.acs.microsoft.com /td SHA256 /dlib ... /dmdf metadata.json`.
  Artifact Signing certificates are valid for about three days, so the
  RFC 3161 timestamp is what keeps a signature valid afterwards.
- *Sign in to Azure* is `azure/login@v3` with OpenID Connect, placed right
  before the bundle step.
- *Bundle the installer* runs `pnpm tauri bundle --bundles nsis` with
  `tauri.release.conf.json` and that sign config; *Sign the GUI for the zip*
  signs `target\release\llama-fidim.exe`.
- *Verify signatures* requires a valid, timestamped signature (and the
  `AS_SIGNER` name when set) on the zip's three executables and on
  setup.exe. *Test the installer* installs, upgrades and uninstalls
  silently and, when signing, checks every installed file.

## Not verified yet

- **The signed path has never run.** A dry run with a stand-in sign command
  confirmed which files Tauri passes and that the uninstaller's sign
  command (with the backslash-doubled paths Tauri writes) runs inside
  `makensis`; the config the workflow writes was checked to load in Tauri.
  Nothing has been signed by Artifact Signing yet.
- **Token lifetime.** The Azure sign-in happens right before bundling, and
  each signature asks the Azure CLI for a token. How long that sign-in
  stays usable was not checked; it only has to last through the bundle
  step and the one signature after it.
- **SmartScreen.** Microsoft says reputation for a new signing identity
  accumulates over time; some developers report it is immediate. Only real
  downloads will tell whether the first signed release still shows a
  warning.
- **Smart App Control** checks every executable and DLL. Signing Llama
  FIDIM does not help the llama.cpp builds it downloads: upstream's ROCm
  builds are unsigned.

## Other routes

- **SignPath Foundation**: free for open-source projects, but Windows shows
  "SignPath Foundation" as the publisher, every release needs a manual
  approval on their side, and signing happens after the build, so
  `uninstall.exe` (signed while the installer is compiled) would stay
  unsigned.
- **Certum Open Source Code Signing**: about USD 58 a year
  **[unverified]**, subject "Open Source Developer, <name>", individuals
  only; it has no official way to sign from CI.

EV certificates no longer skip SmartScreen's reputation check, so they are
not worth their cost here.
