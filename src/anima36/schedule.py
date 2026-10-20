"""Deterministic attention schedule."""
from typing import List, Literal
AttentionKind = Literal["ls_linear", "standard"]

def build_attention_schedule(depth: int = 36) -> List[AttentionKind]:
    """Three LS-Linear blocks followed by one Standard block per group."""
    if depth <= 0 or depth % 4:
        raise ValueError("depth must be a positive multiple of four")
    return [kind for _ in range(depth // 4)
            for kind in ("ls_linear", "ls_linear", "ls_linear", "standard")]

def attention_kind(block_index: int, depth: int = 36) -> AttentionKind:
    schedule = build_attention_schedule(depth)
    if not 0 <= block_index < depth:
        raise IndexError(f"block_index must be in [0, {depth})")
    return schedule[block_index]
