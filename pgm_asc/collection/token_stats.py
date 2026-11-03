"""Token-span alignment and unmodified model probability statistics."""
from __future__ import annotations


def token_span(tokenizer, token_ids, char_start, char_end):
    """Map decoded character spans onto the exact sampled token sequence."""
    def prefix_len(n):
        return len(tokenizer.decode(token_ids[:n], skip_special_tokens=True, clean_up_tokenization_spaces=False))
    def first_at_least(pos):
        lo, hi = 0, len(token_ids)
        while lo < hi:
            mid = (lo + hi) // 2
            if prefix_len(mid) >= pos:
                hi = mid
            else:
                lo = mid + 1
        return lo
    # A token intersecting the boundary belongs to the segment.
    return max(0, first_at_least(char_start + 1) - 1), first_at_least(char_end)


def raw_statistics(logits, tokens):
    """Original model probabilities, before temperature or top-p filtering."""
    lp = logits.float().log_softmax(dim=-1)
    chosen = lp.gather(-1, tokens[:, None]).squeeze(-1)
    entropy = -(lp.exp() * lp).sum(dim=-1)
    return chosen, entropy
