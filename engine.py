from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, MutableMapping, Tuple

import numpy as np

from game import (
    Action,
    GameState,
    Player,
    apply_action,
    initial_state,
    legal_actions,
    terminal_payoff,
)
from ranges import HandRange, sample_hand_pair

# 1 BB = 50 chips for EV conversion to BB/100
BB_CHIPS = 50.0


def chips_to_bb100(chip_ev: float) -> float:
    """Convert chip EV to BB/100 (1 BB = 50)."""
    return (chip_ev / BB_CHIPS) * 100.0


InfosetKey = Tuple[int, Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]


@dataclass
class RunResult:
    """Result of an MCCFR run: iterations, EVs, volatility, and strategy data."""

    iterations: int
    ev_oop_bb100: float
    ev_ip_bb100: float
    volatility: float
    elapsed_seconds: float
    avg_strategy: Dict[InfosetKey, List[float]]
    regrets: Dict[InfosetKey, List[float]]
    infoset_actions: Dict[InfosetKey, List[Action]]


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
    infoset_actions: MutableMapping[InfosetKey, List[Action]] = field(default_factory=dict)

    def _playout_with_strategy(
        self,
        oop_hand: Tuple[int, int, int, int],
        ip_hand: Tuple[int, int, int, int],
        avg_strategy: Dict[InfosetKey, List[float]],
    ) -> float:
        """
        Sample one trajectory using the given strategy; return OOP chip payoff.
        """
        state = initial_state(self.board)
        while not state.is_terminal:
            current = state.current_player
            my_hand = oop_hand if current is Player.OOP else ip_hand
            actions = legal_actions(state)
            if not actions:
                break
            key = _infoset_key(current, state, my_hand)
            probs = avg_strategy.get(key)
            if probs is None or len(probs) != len(actions):
                probs = [1.0 / len(actions)] * len(actions)
            idx = int(self.rng.choice(len(actions), p=probs))
            action = actions[idx]
            state = apply_action(state, action)
        return terminal_payoff(state, oop_hand, ip_hand)

    def _estimate_ev_bb100(
        self,
        avg_strategy: Dict[InfosetKey, List[float]],
        n_samples: int = 200,
    ) -> float:
        """Estimate OOP EV in BB/100 by sampling playouts with the given strategy."""
        payoffs: List[float] = []
        for _ in range(n_samples):
            oop_hand, ip_hand = sample_hand_pair(
                self.oop_range,
                self.ip_range,
                self.board,
                self.rng,
            )
            payoffs.append(self._playout_with_strategy(oop_hand, ip_hand, avg_strategy))
        return chips_to_bb100(float(np.mean(payoffs)))

    def run(
        self,
        seconds: float = 30.0,
        min_iterations: int = 1_000,
        volatility_target: float = 0.1,
        ev_batch_interval: int = 500,
        ev_batch_samples: int = 200,
        volatility_window: int = 20,
    ) -> RunResult:
        """
        Run MCCFR until time or volatility target. Tracks batch EV estimates
        and stops when volatility (std of last batch EVs in BB/100) < volatility_target.
        """
        start = time.time()
        iterations = 0
        batch_evs_bb100: List[float] = []
        last_volatility = 1.0
        last_ev_oop_bb100 = 0.0

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

            if iterations % ev_batch_interval == 0 and iterations > 0:
                avg = self.average_strategy()
                ev_oop = self._estimate_ev_bb100(avg, n_samples=ev_batch_samples)
                batch_evs_bb100.append(ev_oop)
                if len(batch_evs_bb100) > volatility_window:
                    batch_evs_bb100.pop(0)
                last_ev_oop_bb100 = ev_oop
                if len(batch_evs_bb100) >= 2:
                    last_volatility = float(np.std(batch_evs_bb100))
                else:
                    last_volatility = 1.0

            if elapsed >= seconds and iterations >= min_iterations:
                break
            if (
                iterations >= min_iterations
                and last_volatility < volatility_target
                and len(batch_evs_bb100) >= 2
            ):
                break

        elapsed = time.time() - start
        avg_strategy = self.average_strategy()
        # Final EV estimate
        ev_oop_bb100 = (
            last_ev_oop_bb100
            if batch_evs_bb100
            else self._estimate_ev_bb100(avg_strategy, n_samples=500)
        )
        ev_ip_bb100 = -ev_oop_bb100  # zero-sum
        volatility = last_volatility if len(batch_evs_bb100) >= 2 else 0.0

        regrets_dict: Dict[InfosetKey, List[float]] = {}
        for k, arr in self.regrets.items():
            regrets_dict[k] = list(arr.tolist())
        actions_dict: Dict[InfosetKey, List[Action]] = dict(self.infoset_actions)

        return RunResult(
            iterations=iterations,
            ev_oop_bb100=ev_oop_bb100,
            ev_ip_bb100=ev_ip_bb100,
            volatility=volatility,
            elapsed_seconds=elapsed,
            avg_strategy=avg_strategy,
            regrets=regrets_dict,
            infoset_actions=actions_dict,
        )

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
            self.infoset_actions[key] = list(actions)

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
    "BB_CHIPS",
    "RunResult",
    "MCCFRTrainer",
    "chips_to_bb100",
]

