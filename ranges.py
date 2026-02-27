from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from evaluator import parse_card


CardInt = int


@dataclass
class HandRange:
    """
    Weighted distribution over 4-card PLO starting hands.

    Each hand is represented as a tuple of 4 card integers (in Deuces format),
    paired with a non-negative weight.
    """

    hands: List[Tuple[Tuple[CardInt, CardInt, CardInt, CardInt], float]]

    def total_weight(self) -> float:
        return sum(weight for _, weight in self.hands)

    def filtered_for_board(
        self,
        board: Sequence[CardInt],
    ) -> "HandRange":
        board_set = set(board)
        filtered: List[Tuple[Tuple[CardInt, CardInt, CardInt, CardInt], float]] = []
        for hand, weight in self.hands:
            if not board_set.intersection(hand):
                filtered.append((hand, weight))
        return HandRange(filtered)

    def exclude_cards(self, cards: Iterable[CardInt]) -> "HandRange":
        exclude_set = set(cards)
        filtered: List[Tuple[Tuple[CardInt, CardInt, CardInt, CardInt], float]] = []
        for hand, weight in self.hands:
            if not exclude_set.intersection(hand):
                filtered.append((hand, weight))
        return HandRange(filtered)


def parse_range_lines(lines: Iterable[str]) -> HandRange:
    """
    Parse a simple text representation of a 4-card Omaha range.

    Each non-empty, non-comment line has the form:
        HAND [weight]
    where:
        HAND   = 8-character string, e.g. 'AsKdQcJh'
        weight = optional float (default = 1.0)
    """
    hands: List[Tuple[Tuple[CardInt, CardInt, CardInt, CardInt], float]] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if not parts:
            continue
        hand_str = parts[0]
        if len(hand_str) != 8:
            raise ValueError(f"Invalid hand string '{hand_str}', expected 8 characters.")
        card_codes = [hand_str[i : i + 2] for i in range(0, 8, 2)]
        cards = tuple(parse_card(code) for code in card_codes)
        if len(set(cards)) != 4:
            raise ValueError(f"Hand '{hand_str}' contains duplicate cards.")
        weight = 1.0
        if len(parts) > 1:
            try:
                weight = float(parts[1])
            except ValueError as exc:
                raise ValueError(f"Invalid weight '{parts[1]}' for hand '{hand_str}'.") from exc
        if weight < 0.0:
            raise ValueError(f"Negative weight '{weight}' for hand '{hand_str}'.")
        hands.append((cards, weight))
    return HandRange(hands)


def sample_from_range(
    hand_range: HandRange,
    rng,
    exclude_cards: Sequence[CardInt] | None = None,
) -> Tuple[CardInt, CardInt, CardInt, CardInt]:
    """
    Sample a 4-card hand from a weighted range, optionally excluding cards.
    """
    if exclude_cards:
        hand_range = hand_range.exclude_cards(exclude_cards)
    total = hand_range.total_weight()
    if total <= 0.0:
        raise ValueError("Cannot sample from an empty or zero-weight range.")
    threshold = rng.random() * total
    cumulative = 0.0
    for hand, weight in hand_range.hands:
        cumulative += weight
        if cumulative >= threshold:
            return hand
    # Fallback in case of floating point rounding.
    return hand_range.hands[-1][0]


def sample_hand_pair(
    oop_range: HandRange,
    ip_range: HandRange,
    board: Sequence[CardInt],
    rng,
) -> Tuple[
    Tuple[CardInt, CardInt, CardInt, CardInt],
    Tuple[CardInt, CardInt, CardInt, CardInt],
]:
    """
    Sample (OOP hand, IP hand) from weighted ranges ensuring no card conflicts
    with each other or with the board.
    """
    board_set = set(board)

    # Sample OOP first, respecting board.
    oop_hand = sample_from_range(oop_range.filtered_for_board(board), rng)
    oop_set = set(oop_hand)

    # Sample IP, respecting both board and OOP hand.
    ip_exclude = board_set.union(oop_set)
    ip_hand = sample_from_range(ip_range, rng, exclude_cards=tuple(ip_exclude))
    return oop_hand, ip_hand


__all__ = [
    "CardInt",
    "HandRange",
    "parse_range_lines",
    "sample_from_range",
    "sample_hand_pair",
]

