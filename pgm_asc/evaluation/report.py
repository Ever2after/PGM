"""Integrated results for the two retained experimental stages."""
from __future__ import annotations

from pgm_asc.core.config import RUN, SELECTOR, STATE
from pgm_asc.core.io import read_json


def report():
    selector = read_json(SELECTOR / 'evaluation.json')
    state = read_json(STATE / 'evaluation.json')
    lines = ['# PGM adaptive self-consistency: integrated results', '',
             'Selector stage (historical V2) and state/value stage (historical V5) share a frozen answer selector. Their holdouts differ; compare methods within each table.', '',
             '## Selector stage — 500 questions', '',
             '| Method | Accuracy | Mean responses | Mean generated tokens |', '|---|---:|---:|---:|']
    for key, row in selector['stages'][selector['final_stage']].items():
        lines.append(f"| {key} | {row['accuracy']:.4f} | {row['mean_samples']:.3f} | {row['mean_tokens']:.3f} |")
    lines += ['', '## State/value stage — fresh 1,000 questions', '',
              '| Method | Accuracy | Mean responses | Mean generated tokens |', '|---|---:|---:|---:|']
    for key in ['frozen_v2', 'selector_M', 'selector_S', 'S_masked_seed42', 'S_mlp_seed42', 'S_unmasked_seed42', 'cges', 'asc', 'reasc', 'sc8']:
        row = state['policy'][key]
        lines.append(f"| {key} | {row['accuracy']:.4f} | {row['mean_samples']:.3f} | {row['mean_tokens']:.3f} |")
    lines += ['', 'The primary temporal-state method reduces generated tokens but fails the accuracy constraint. Frozen-v2 remains the retained reference; integration does not promote V5.', '',
              'The spent selector holdout500 is part of state dev700. The state holdout1000 excludes all prior cohorts.',
              'Generated-token figures exclude extraction/head costs; online latency is unverified.', '',
              '[Selector details](./selector/report.md) · [State details](./state/report.md) · [Method](../../../docs/METHOD.md) · [History](../../../docs/EXPERIMENT_HISTORY.md)']
    target = RUN / 'result_summary.md'; target.write_text('\n'.join(lines) + '\n')
    return str(target)
