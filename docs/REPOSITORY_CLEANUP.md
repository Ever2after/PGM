# Repository preparation (2026-11-03)

- Rewrote the root README for the current ReASC/follow-up study; added a documentation index and reproduction guide.
- Added a public follow-up entry point, aggregate-result exporter, package metadata, the recorded ReASC environment pins, and CPU test CI.
- Replaced the MATH-only summary script with the shared all-model/dataset exporter; moved the one-off runtime benchmark into `scripts/diagnostics/`.
- Made the synthetic reference-collection and legacy-transfer tests independent of local experiment artifacts. Their parity receipts, checkpoints and transfer boundary are now test fixtures.
- Published all 10 completed evaluations, including the earlier exploratory Gemma fresh subset, under `results/followup/`; raw banks remain local.
- Kept historical method/config/source paths unchanged wherever experiment locks register their hashes. Historical runners remain available for provenance and existing locks.
- Moved the original research proposal into `docs/references/research_proposal.pdf`.
- Removed the redundant user ZIP and extracted reference copy listed below. The pinned official ReASC checkout remains the executable reference; original audit metadata retains the removed archive checksums.
- Ignored environments, caches, weights, responses, logs, credentials, editor state and downloaded upstream checkouts. Model/cache storage used by existing environments remains local.

## Removed duplicate inputs

| Path | Files | Bytes | Reason |
|---|---:|---:|---|
| `pgm_reasc.zip` | 1 | 3,791,646 | Redundant source archive/extraction; pinned official upstream is the executable reference |
| `third_party/reasc_reference` | 21 | 13,151,029 | Redundant source archive/extraction; pinned official upstream is the executable reference |

No benchmark was rerun or retuned during this cleanup. Aggregate exports were generated from existing evaluation JSON files. The GitHub repository target is `ever2after/PGM`; creating or publishing the remote is a separate action.

## Validation

- 82 tests pass in a separate source tree without any original experiment artifacts or virtual environments in that tree. Pinned external reference checkouts were supplied; the existing Python environment ran the tests.
- The package wheel builds successfully and contains only `pgm_asc` and package metadata.
- All 10 aggregate exports match their source evaluation metrics and SHA256 values.
- All 10 checked policy/model locks pass: three historical V2/V5 locks, the initial ReASC comparison lock, and all six follow-up cell locks.
- Both external reference commits and their tracked working trees are verified.
- New documentation links resolve; candidate publication files contain no detected private-key, Hugging Face token, GitHub token or AWS access-key patterns, and no file exceeds 10 MiB. Pattern scanning is a limited check, not a guarantee that every possible secret format is covered.
- `.gitignore` excludes all three local virtual environments, artifact storage, upstream checkouts, representative credentials, model files and raw data files.

Initial local preparation initialized branch `main` with origin `https://github.com/ever2after/PGM.git`. Repository creation and initial push were subsequently requested as a separate publication step.
