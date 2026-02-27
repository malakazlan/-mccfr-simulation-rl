from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, MutableMapping, Tuple

import numpy as np

from game import Action, GameState, Player, apply_action, initial_state, legal_actions, terminal_payoff
from ranges import HandRange, sample_hand_pair


InfosetKey = Tuple[int, Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]


def _infoset_key(
    player: Player,
    state: GameState,
    my_hand: Tuple[int, int, int, int],
) -> InfosetKey:
    """
    Information set key:
    (player_index, sorted_board, sorted_my_hand, history_actions)
    """
    player_index = 0 if player is Player.OOP else 1
    board_sorted = tuple(sorted(state.board))
    hand_sorted = tuple(sorted(my_hand))
    history = tuple(action.value for action in state.history)
    return player_index, board_sorted, hand_sorted, history


def _regret_matching(regrets: np.ndarray) -> np.ndarray:
    positive = np.maximum(regrets, 0.0)
    total = positive.sum()
    if total > 0.0:
        return positive / total
    n_actions = regrets.shape[0]
    return np.full(n_actions, 1.0 / float(n_actions), dtype=float)


@dataclass
class MCCFRTrainer:
    """
    MCCFR trainer for river-only 4-card PLO with chance sampling over hands.

    The game is assumed to be zero-sum, and utilities are from OOP's perspective.
    """

    board: Tuple[int, int, int, int, int]
    oop_range: HandRange
    ip_range: HandRange
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())
    regrets: MutableMapping[InfosetKey, np.ndarray] = field(default_factory=dict)
    strategy_sum: MutableMapping[InfosetKey, np.ndarray] = field(default_factory=dict)

    def run(
        self,
        seconds: float = 30.0,
        min_iterations: int = 1_000,
    ) -> None:
        """
        Run MCCFR for a given wall-clock time budget.
        """
        start = time.time()
        iterations = 0
        while True:
            oop_hand, ip_hand = sample_hand_pair(
                self.oop_range,
                self.ip_range,
                self.board,
                self.rng,
            )
            state = initial_state(self.board)
            self._cfr(
                state=state,
                oop_hand=oop_hand,
                ip_hand=ip_hand,
                reach_probs=(1.0, 1.0),
            )
            iterations += 1
            elapsed = time.time() - start
            if elapsed >= seconds and iterations >= min_iterations:
                break

    def _cfr(
        self,
        state: GameState,
        oop_hand: Tuple[int, int, int, int],
        ip_hand: Tuple[int, int, int, int],
        reach_probs: Tuple[float, float],
    ) -> float:
        """
        Chance-sampled CFR recursion.

        Returns utility from OOP's perspective.
        """
        if state.is_terminal:
            return terminal_payoff(state, oop_hand, ip_hand)

        current_player = state.current_player
        player_index = 0 if current_player is Player.OOP else 1
        opponent_index = 1 - player_index

        actions: List[Action] = legal_actions(state)
        if not actions:
            # Safety: if no actions but not flagged terminal, treat as showdown.
            return terminal_payoff(state, oop_hand, ip_hand)

        my_hand = oop_hand if current_player is Player.OOP else ip_hand
        key = _infoset_key(current_player, state, my_hand)

        if key not in self.regrets:
            self.regrets[key] = np.zeros(len(actions), dtype=float)
            self.strategy_sum[key] = np.zeros(len(actions), dtype=float)

        regrets = self.regrets[key]
        strategy = _regret_matching(regrets)

        # Accumulate strategy for average strategy computation.
        self.strategy_sum[key] += reach_probs[player_index] * strategy

        action_utils = np.zeros(len(actions), dtype=float)
        node_util = 0.0

        for i, action in enumerate(actions):
            next_state = apply_action(state, action)
            next_reach = list(reach_probs)
            next_reach[player_index] *= strategy[i]
            util_oop = self._cfr(
                state=next_state,
                oop_hand=oop_hand,
                ip_hand=ip_hand,
                reach_probs=(next_reach[0], next_reach[1]),
            )
            # Convert to current player's utility.
            my_util = util_oop if current_player is Player.OOP else -util_oop
            action_utils[i] = my_util
            node_util += strategy[i] * my_util

        # Regret update scaled by opponent reach probability.
        for i in range(len(actions)):
            regret = action_utils[i] - node_util
            self.regrets[key][i] += reach_probs[opponent_index] * regret

        # Return utility from OOP's perspective.
        return node_util if current_player is Player.OOP else -node_util

    def average_strategy(self) -> Dict[InfosetKey, List[float]]:
        """
        Compute the average strategy at each information set as a list
        of action probabilities in the same order as generated by legal_actions.
        """
        avg: Dict[InfosetKey, List[float]] = {}
        for key, strategy_sum in self.strategy_sum.items():
            total = float(strategy_sum.sum())
            if total > 0.0:
                avg[key] = list((strategy_sum / total).tolist())
            else:
                n_actions = strategy_sum.shape[0]
                avg[key] = [1.0 / float(n_actions)] * n_actions
        return avg


__all__ = [
    "MCCFRTrainer",
]

