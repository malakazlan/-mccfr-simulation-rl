from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, List, Tuple

from evaluator import parse_cards
from engine import MCCFRTrainer, RunResult
from game import action_to_str
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


def _infoset_label(key: Any) -> str:
    """Human-readable infoset identifier for output."""
    player_idx, board, hand, history = key
    player = "OOP" if player_idx == 0 else "IP"
    hist_str = ",".join(str(h) for h in history)
    return f"{player} board={len(board)}cards hand=4cards history=[{hist_str}]"


def _print_results(result: RunResult) -> None:
    """Print EV, volatility, iterations, and per-node strategy/regret."""
    print("=" * 60)
    print("GLOBAL METRICS")
    print("=" * 60)
    print(f"  Elapsed (s):     {result.elapsed_seconds:.3f}")
    print(f"  Iterations:      {result.iterations}")
    print(f"  Volatility:      {result.volatility:.6f}")
    print(f"  EV OOP (BB/100): {result.ev_oop_bb100:.4f}")
    print(f"  EV IP (BB/100):  {result.ev_ip_bb100:.4f}")
    print()

    print("=" * 60)
    print("PER NODE: Strategy Frequency and Cumulative Regret")
    print("=" * 60)
    for key in sorted(result.avg_strategy.keys(), key=lambda k: (k[0], k[3])):
        strategies = result.avg_strategy[key]
        regrets = result.regrets.get(key, [])
        actions = result.infoset_actions.get(key, [])
        if len(actions) != len(strategies):
            continue
        print(f"\n  Infoset: {_infoset_label(key)}")
        for i, (action, freq) in enumerate(zip(actions, strategies)):
            reg = regrets[i] if i < len(regrets) else 0.0
            print(f"    {action_to_str(action):12s}  strategy={freq:.4f}  regret={reg:.4f}")
    print()


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
        "--volatility-target",
        type=float,
        default=0.1,
        help="Stop when volatility below this (default: 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (optional).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only global metrics, skip per-node output.",
    )

    args = parser.parse_args(argv)

    board = _parse_board(args.board)
    oop_range = _load_range_file(Path(args.oop_range))
    ip_range = _load_range_file(Path(args.ip_range))

    import numpy as np

    rng = np.random.default_rng(args.seed)
    trainer = MCCFRTrainer(
        board=board,
        oop_range=oop_range,
        ip_range=ip_range,
        rng=rng,
    )

    result = trainer.run(
        seconds=args.seconds,
        min_iterations=args.min_iterations,
        volatility_target=args.volatility_target,
    )

    if args.quiet:
        print(f"Elapsed: {result.elapsed_seconds:.3f}s  Iterations: {result.iterations}")
        print(f"Volatility: {result.volatility:.6f}")
        print(f"EV OOP (BB/100): {result.ev_oop_bb100:.4f}  EV IP (BB/100): {result.ev_ip_bb100:.4f}")
    else:
        _print_results(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

