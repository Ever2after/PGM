# External references

Upstream code and dataset files are downloaded locally rather than vendored in
this repository. Keep upstream notices and licenses in each checkout.

```bash
python scripts/setup_references.py
```

| Local directory | Upstream | Pinned commit |
|---|---|---|
| `reasc_official/` | [junseokkim00/ReASC](https://github.com/junseokkim00/ReASC) | `8ee1c76ea0c04bb3cdb81d09bc2655dbc3a8cecb` |
| `xverify_official/` | [IAAR-Shanghai/xVerify](https://github.com/IAAR-Shanghai/xVerify) | `3f50a37912992ab89c51bdc8a83a655c85714079` |

ReASC supplies source notebooks and the reference MATH/Omni-MATH files. xVerify
supplies the original prompt literals. The adapter reads these files without
executing notebook cells. Model weights are downloaded separately from
Hugging Face at the revisions in `configs/`.

`reasc_sources.json` is the original static audit manifest. Its status and archive
paths describe the audit at that time; current execution status is documented in
`results/followup/`. The redundant user ZIP and extracted `reasc_reference/` copy
were removed during repository cleanup; their checksums remain in that manifest.
