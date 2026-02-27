from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

from evaluator import parse_cards
from engine import MCCFRTrainer
from ranges import HandRange, parse_range_lines


def _parse_board(board_str: str) -> Tuple[int, int, int, int, int]:
    if len(board_str) != 10:
        raise ValueError("Board string must have exactly 10 characters (5 cards).")
    codes = [board_str[i : i + 2] for i in range(0, 10, 2)]
    cards = parse_cards(codes)
    if len(set(cards)) != 5:
        raise ValueError("Board contains duplicate cards.")
    return (cards[0], cards[1], cards[2], cards[3], cards[4])


def _load_range_file(path: Path) -> HandRange:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to read range file '{path}': {exc}") from exc
    return parse_range_lines(text.splitlines())


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="River-only MCCFR solver for 4-card PLO (heads-up).",
    )
    parser.add_argument(
        "--board",
        type=str,
        required=True,
        help="Board cards as 10-character string, e.g. AsKh7d4c2h.",
    )
    parser.add_argument(
        "--oop-range",
        type=str,
        required=True,
        help="Path to OOP range file.",
    )
    parser.add_argument(
        "--ip-range",
        type=str,
        required=True,
        help="Path to IP range file.",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=30.0,
        help="Time budget for MCCFR iterations (default: 30).",
    )
    parser.add_argument(
        "--min-iterations",
        type=int,
        default=1000,
        help="Minimum number of iterations to run (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (optional).",
    )

    args = parser.parse_args(argv)

    board = _parse_board(args.board)
    oop_range = _load_range_file(Path(args.oop_range))
    ip_range = _load_range_file(Path(args.ip_range))

    rng = np.random.default_rng(args.seed)
    trainer = MCCFRTrainer(
        board=board,
        oop_range=oop_range,
        ip_range=ip_range,
        rng=rng,
    )

    start = time.time()
    trainer.run(seconds=args.seconds, min_iterations=args.min_iterations)
    elapsed = time.time() - start

    # At this stage we expose only basic meta-information. Further reporting
    # (EV estimation, volatility, per-node strategies) can be built on top.
    avg_strategy = trainer.average_strategy()

    print(f"Training completed in {elapsed:.3f} seconds.")
    print(f"Information sets visited: {len(avg_strategy)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

