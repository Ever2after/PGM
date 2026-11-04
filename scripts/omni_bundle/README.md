# Omni-MATH bundle runners (archived)

These files came from the private artifact bundle `Ever2after/PGM-omni-resume`
(snapshot of commit `9ec705a`), which was used to run the Omni-MATH and
Llama-3.1-8B cells on KISTI Hangang before being merged into this repository.
They are kept as a record of how those cells were launched. Use
`scripts/run_cell.py` (and `scripts/slurm/cell.sbatch`) for new work.

| File | Purpose at the time |
|---|---|
| `OMNI_RESUME.md` | Restore and resume instructions shipped with the bundle |
| `resume_omni.py` | Verify the bundle and run MATH → Omni-MATH transfer cells |
| `run_omni_indomain.py` | Omni in-domain runner; its collector now lives in `scripts/run_cell.py` |
| `omni.sbatch`, `omni_indomain.sbatch` | Whole-node Slurm jobs for the two runners |

`resume_omni.py` resolves paths relative to its own directory and reads
`omni_bundle_manifest.json` from there, so it runs only from the root of an
extracted bundle. The sbatch files hard-code the original `/scratch` paths. The
bundle manifest (file hashes, Git commit, cohort) is in
`results/followup/provenance/omni_bundle_manifest.json`.
