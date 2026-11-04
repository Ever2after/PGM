# Resume frozen PGM transfer on Omni-MATH

This private artifact bundle accompanies `https://github.com/Ever2after/PGM`.
It contains the frozen MATH heads and all their registered inputs for Gemma-3-4B,
Qwen2.5-3B and Qwen2.5-7B, plus the previously selected Omni-MATH cohort.
No Omni generation or evaluation was performed when preparing this bundle.

## Restore

Download the archive and `.sha256` sidecar from the Hugging Face dataset repo
shown in its README. In the download directory:

```bash
sha256sum -c pgm-omni-resume.tar.gz.sha256
mkdir -p PGM-omni
tar -xzf pgm-omni-resume.tar.gz -C PGM-omni
cd PGM-omni
python resume_omni.py verify
```

The archive contains repository-relative paths, regular files only, and no
credentials or pretrained base-model weights. The Git source snapshot is
included, so a separate GitHub clone is not required. Extract into a new
directory rather than mixing it with a different revision of the project.

## Install

The recorded generation environment used Python 3.10.12, PyTorch 2.8.0,
vLLM 0.11.0, and Transformers 4.56.2. Use a machine with a compatible NVIDIA
driver/CUDA environment, enough GPU memory for the selected generator, and
enough disk space for the downloaded weights and the growing response banks.
The collection implementation reserves up to 25% of each visible GPU's memory;
larger generators may require higher-memory devices. This bundle does not
change the recorded memory settings or certify compatibility with every GPU.

```bash
python3.10 -m venv .venv-reasc
source .venv-reasc/bin/activate
python -m pip install -r requirements-reasc.txt
python -m pip install --no-deps -e .
```

Authenticate to Hugging Face locally if needed. Gemma requires access accepted
through the model provider. Each generator and the common xVerify judge are
downloaded at the exact pinned revision; base weights are not redistributed.

## Run Qwen7B first

```bash
# Optional explicit asset download; the run command also downloads by default.
python resume_omni.py download --model qwen7b

# Optional 2-question collection smoke; this does not evaluate/tune a policy.
python resume_omni.py run --model qwen7b --gpus 0,1,2,3 --offline --limit 2

# Full frozen-policy transfer, resumable with the same command.
python resume_omni.py run --model qwen7b --gpus 0,1,2,3 --offline

# Or run all three models sequentially, downloading pinned assets as needed.
python resume_omni.py run --model all --gpus 0,1,2,3
```

GPU order is: generation shard0, generation shard1, exact-token TF features,
common xVerify judge. Pass `--gpus 4,5,6,7` when that is the physical allocation.
GPU remapping is a runtime setting; the locked config bytes stay unchanged.
Run without an enclosing `CUDA_VISIBLE_DEVICES` mask or scheduler remapping
that makes those indices ambiguous. Use your assigned devices only.

The runner verifies every bundled file and both original/transfer locks before
work, detects worker exceptions even during CUDA teardown, and cleans up its
worker groups on interruption. A fixed two-shard layout is used for resume.
It does not refit heads, calibrators, similarity thresholds or stopping rules.

## Protocol

- 999 distinct questions from the previously audited 1,000-row archive; one
  duplicate removed, no exact normalized overlap with fitting cohorts.
- Frozen MATH-to-Omni transfer for all arms and baseline offline calibration.
- 16 responses per question, body cap 2,048 tokens, both answer-extraction
  passes retained. The same bank is shared by methods within each model.
- No Omni labels for fitting/calibration/threshold selection. Gold sidecars are
  used only for the common judge and final scoring.
- 11 independently fitted arms x training seeds42/43/44, plus the three
  same-head single-partition interventions; no continuation policy.
- Source protocols, cohort manifests and original selection locks are included.
  Additional historical MATH fitting inputs are present to satisfy the original
  lock closure, not to authorize new Omni fitting.

The exact xVerify prompt snapshots and upstream license are supplied, so GPU
judging needs no separate upstream Git clone. These are minimal source
snapshots, not complete upstream Git checkouts; the root repository's full
reference-setup tests are separate from this continuation workflow.

## Outputs

```text
artifacts/runs/pgm_followup/<model>_omnimath/test/study_report.md
artifacts/runs/pgm_followup/<model>_omnimath/test/study_evaluation.json
artifacts/runs/pgm_followup/<model>_omnimath/test/study_decisions/
artifacts/runs/pgm_followup/<model>_omnimath/logs/test_portable_*.log
```

This is bank collection plus offline replay, not measured causal serving
latency. Output tokens exclude input processing and additional TF feature
costs. Primary structural inference uses four per-cell Holm-adjusted contrasts;
the whole model/dataset matrix is not globally corrected. Training seeds do
not increase the number of independent test questions.
