from __future__ import annotations

from itertools import combinations
from typing import Iterable, List, Sequence

from deuces import Card, Evaluator


_evaluator = Evaluator()


def parse_card(code: str) -> int:
    """
    Parse a two-character card code like 'As' or 'Td' into the internal
    Deuces integer representation.
    """
    if len(code) != 2:
        raise ValueError(f"Invalid card code '{code}', expected length 2.")
    try:
        return Card.new(code)
    except Exception as exc:
        raise ValueError(f"Failed to parse card code '{code}'.") from exc


def parse_cards(codes: Iterable[str]) -> List[int]:
    """
    Parse an iterable of 2-character card codes into Deuces card integers.
    """
    return [parse_card(code) for code in codes]


def evaluate_5card_hand(cards_5: Sequence[int]) -> int:
    """
    Evaluate a 5-card poker hand and return a rank in [1, 7462],
    where 1 is best (royal flush) and 7462 is worst.
    """
    if len(cards_5) != 5:
        raise ValueError("evaluate_5card_hand requires exactly 5 cards.")
    # Deuces expects separate board and hand; we can pass 5 cards as board.
    return _evaluator.evaluate(list(cards_5), [])


def evaluate_omaha_hand(
    hole_cards: Sequence[int],
    board_cards: Sequence[int],
) -> int:
    """
    Evaluate a 4-card Omaha hand on a 5-card board.

    Omaha rule: the final hand must use exactly 2 hole cards and 3 board cards.
    This function enumerates all C(4,2) * C(5,3) = 60 combinations and returns
    the best (minimum) 5-card rank.
    """
    if len(hole_cards) != 4:
        raise ValueError("Omaha hand must have exactly 4 hole cards.")
    if len(board_cards) != 5:
        raise ValueError("Board must have exactly 5 cards on the river.")

    best_rank: int | None = None

    for hole_pair in combinations(hole_cards, 2):
        for board_triplet in combinations(board_cards, 3):
            rank = evaluate_5card_hand((*hole_pair, *board_triplet))
            if best_rank is None or rank < best_rank:
                best_rank = rank

    if best_rank is None:
        raise RuntimeError("Failed to evaluate Omaha hand: no combinations generated.")

    return best_rank


def compare_omaha_hands(
    oop_hole: Sequence[int],
    ip_hole: Sequence[int],
    board_cards: Sequence[int],
) -> int:
    """
    Compare two 4-card Omaha hands on a fixed 5-card board.

    Returns:
    - 1 if OOP wins
    - -1 if IP wins
    - 0 if tie
    """
    oop_rank = evaluate_omaha_hand(oop_hole, board_cards)
    ip_rank = evaluate_omaha_hand(ip_hole, board_cards)

    if oop_rank < ip_rank:
        return 1
    if oop_rank > ip_rank:
        return -1
    return 0


__all__ = [
    "parse_card",
    "parse_cards",
    "evaluate_5card_hand",
    "evaluate_omaha_hand",
    "compare_omaha_hands",
]

