"""Balanced, interleaved attention schedule for Anima-36."""
from typing import List, Literal

AttentionKind = Literal["ls_linear", "standard"]

def build_attention_schedule(depth: int = 36) -> List[AttentionKind]:
    """Alternate LS-Linear and Standard attention for a 1:1 schedule.

    Even depths produce equal counts. Interleaving keeps global attention
    available throughout the network rather than concentrating it in one stage.
    """
    if depth <= 0 or depth % 2:
        raise ValueError("depth must be a positive even number")
    return [kind for _ in range(depth // 2)
            for kind in ("ls_linear", "standard")]

def attention_kind(block_index: int, depth: int = 36) -> AttentionKind:
    schedule = build_attention_schedule(depth)
    if not 0 <= block_index < depth:
        raise IndexError(f"block_index must be in [0, {depth})")
    return schedule[block_index]
