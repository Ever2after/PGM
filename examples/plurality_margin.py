"""Example of adding a new baseline to the v2 evaluation (bank replay).

Rule: sample until the leading answer is ahead of the runner-up by `margin` votes (at least 2 responses), then
answer the leader. Evaluate it, then draw it on the same figures as the PGM and the built-in baselines:

    pgm v2 method --plugin examples/plurality_margin.py
    pgm v2 report                       # tables, figures/accuracy_vs_tokens.png, report.html now include it
    pgm v2 plot --methods M1 asc plurality_margin --out results/v2/figures/margin_vs_m1

A method receives one split at a time with labels removed (see pgm_asc.v2.methods.SPLIT_INPUTS) and returns, per
question, the stop index (responses consumed - 1) and the answer slot it returns. It may only use responses up to
its stop index; the harness checks that the returned answer was among them.
"""
import numpy as np

from pgm_asc.v2.methods import register


@register("plurality_margin", thresholds=[1, 2, 3, 4, 5, 6, 8], default=3, label="Margin stop")
def plurality_margin(arrays, margin, context):
    ids, valid = arrays["ids"].astype(int), arrays["valid"]
    q, n = ids.shape
    stop, chosen = np.full(q, n - 1), np.zeros(q, dtype=int)
    for i in range(q):
        counts = np.zeros(n)
        for t in range(n):
            if valid[i, t]:
                counts[ids[i, t]] += 1
            top = np.sort(counts)[::-1]
            if (t + 1 >= 2 and top[0] - top[1] >= margin) or t == n - 1:
                stop[i], chosen[i] = t, int(np.argmax(counts))    # ties: first observed answer
                break
    return stop, chosen
