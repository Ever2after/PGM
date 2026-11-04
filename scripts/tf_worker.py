"""Run pgm_asc.collection.study for tokenizers that ship without a pad token (e.g. Llama 3.1).

score_batch right-pads with tokenizer.pad_token_id under an attention mask and reads hidden
states only at real positions, so the pad value never reaches a feature. Filling a missing
pad token with EOS is therefore output-neutral; it is a no-op for Gemma/Qwen. Done here, not
in pgm_asc/, so that source hashes registered in existing locks stay valid.
"""
import runpy
import sys

from transformers import AutoTokenizer

_from_pretrained = AutoTokenizer.from_pretrained


def from_pretrained(*args, **kwargs):
    tokenizer = _from_pretrained(*args, **kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


AutoTokenizer.from_pretrained = from_pretrained
sys.argv = ["pgm_asc.collection.study", *sys.argv[1:]]
runpy.run_module("pgm_asc.collection.study", run_name="__main__", alter_sys=True)
