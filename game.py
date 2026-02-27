from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterable, List, Sequence, Tuple

from evaluator import compare_omaha_hands


class Player(Enum):
    OOP = 0
    IP = 1


class ActionType(Enum):
    CHECK = auto()
    BET_25 = auto()
    BET_100 = auto()
    FOLD = auto()
    CALL = auto()
    RAISE_50 = auto()
    RAISE_100 = auto()


Action = ActionType


@dataclass(frozen=True)
class GameState:
    """
    River-only PLO game state for a single betting round.

    All chip amounts are represented as integers (e.g. starting pot = 50,
    effective stacks = 50 for each player).
    """

    pot: int
    stacks: Tuple[int, int]
    history: Tuple[Action, ...]
    current_player: Player
    last_bet: int
    board: Tuple[int, int, int, int, int]

    @property
    def is_terminal(self) -> bool:
        if not self.history:
            return False
        last_action = self.history[-1]
        # Any fold ends the hand immediately.
        if last_action is ActionType.FOLD:
            return True
        # A call always ends the betting on the river.
        if last_action is ActionType.CALL:
            return True
        # After check-check, go to showdown.
        if len(self.history) >= 2 and self.history[-2:] == (
            ActionType.CHECK,
            ActionType.CHECK,
        ):
            return True
        return False


def initial_state(board: Sequence[int]) -> GameState:
    """
    Construct the initial river state.

    Pot = 50, effective stacks = 50 each.
    OOP acts first.
    """
    if len(board) != 5:
        raise ValueError("Initial board must have exactly 5 cards.")
    return GameState(
        pot=50,
        stacks=(50, 50),
        history=(),
        current_player=Player.OOP,
        last_bet=0,
        board=(board[0], board[1], board[2], board[3], board[4]),
    )


def _pot_after_bet(current_pot: int, bet_size: int) -> int:
    return current_pot + bet_size


def _max_pot_total_bet(pot_before: int, amount_to_call: int, stack: int) -> int:
    """
    Maximum total bet a player may put in under pot-limit:
    amount_to_call + pot_after_call, where pot_after_call = pot_before + amount_to_call.
    """
    pot_after_call = pot_before + amount_to_call
    max_total = amount_to_call + pot_after_call
    return min(max_total, stack)


def legal_actions(state: GameState) -> List[Action]:
    """
    Compute the list of legal actions from this state, constrained to the
    specified betting tree and strict pot-limit rules.
    """
    if state.is_terminal:
        return []

    pot = state.pot
    stacks = state.stacks
    current = state.current_player
    last_bet = state.last_bet

    idx = 0 if current is Player.OOP else 1
    my_stack = stacks[idx]

    actions: List[Action] = []

    # Root or after check on river: OOP or IP may check or bet 25% / 100% pot,
    # respecting pot-limit and stack sizes.
    if not state.history or (
        len(state.history) == 1 and state.history[0] is ActionType.CHECK
    ):
        actions.append(ActionType.CHECK)
        # 25% pot bet
        bet_25 = min(int(round(0.25 * pot)), my_stack)
        if bet_25 > 0:
            actions.append(ActionType.BET_25)
        # 100% pot bet (maximum allowed here)
        bet_pot = min(pot, my_stack)
        if bet_pot > 0 and bet_pot != bet_25:
            actions.append(ActionType.BET_100)
        return actions

    # Responding to an existing bet.
    if last_bet > 0:
        amount_to_call = min(last_bet, my_stack)
        # Fold and call are always available when facing a bet.
        actions.append(ActionType.FOLD)
        if amount_to_call > 0:
            actions.append(ActionType.CALL)

        # Pot-limit raise sizes: 50% and 100% of maximum allowed pot-sized raise.
        # Only IP has raises in the specified betting tree.
        if current is Player.IP and my_stack > amount_to_call:
            max_total = _max_pot_total_bet(pot, amount_to_call, my_stack)
            if max_total > amount_to_call:
                raise_component = max_total - amount_to_call
                half_raise_total = amount_to_call + max(
                    1, int(round(0.5 * raise_component))
                )
                full_raise_total = max_total

                if half_raise_total > amount_to_call and half_raise_total <= my_stack:
                    actions.append(ActionType.RAISE_50)
                if (
                    full_raise_total > amount_to_call
                    and full_raise_total <= my_stack
                    and full_raise_total != half_raise_total
                ):
                    actions.append(ActionType.RAISE_100)

    return actions


def apply_action(state: GameState, action: Action) -> GameState:
    """
    Apply an action and return the next game state.
    """
    if state.is_terminal:
        raise ValueError("Cannot act from a terminal state.")

    pot = state.pot
    oop_stack, ip_stack = state.stacks
    last_bet = state.last_bet
    history = list(state.history)
    current = state.current_player

    idx_current = 0 if current is Player.OOP else 1

    def update_stacks(
        stacks_in: Tuple[int, int],
        player_index: int,
        amount: int,
    ) -> Tuple[int, int]:
        oop, ip = stacks_in
        if player_index == 0:
            return (oop - amount, ip)
        return (oop, ip - amount)

    if action is ActionType.CHECK:
        history.append(ActionType.CHECK)
        next_player = Player.IP if current is Player.OOP else Player.OOP
        return GameState(
            pot=pot,
            stacks=(oop_stack, ip_stack),
            history=tuple(history),
            current_player=next_player,
            last_bet=last_bet,
            board=state.board,
        )

    if action in (ActionType.BET_25, ActionType.BET_100):
        # New bet from player with no outstanding bet they are facing.
        bet_size = _bet_amount_from_action(action, pot, state.stacks[idx_current])
        if bet_size <= 0:
            raise ValueError("Bet size computed as non-positive.")
        stacks_new = update_stacks(state.stacks, idx_current, bet_size)
        pot_new = _pot_after_bet(pot, bet_size)
        history.append(action)
        return GameState(
            pot=pot_new,
            stacks=stacks_new,
            history=tuple(history),
            current_player=Player.IP if current is Player.OOP else Player.OOP,
            last_bet=bet_size,
            board=state.board,
        )

    # At this point, the player is responding to a bet.
    if action is ActionType.FOLD:
        history.append(ActionType.FOLD)
        return GameState(
            pot=pot,
            stacks=(oop_stack, ip_stack),
            history=tuple(history),
            current_player=current,
            last_bet=last_bet,
            board=state.board,
        )

    if action is ActionType.CALL:
        amount_to_call = min(last_bet, state.stacks[idx_current])
        stacks_new = update_stacks(state.stacks, idx_current, amount_to_call)
        pot_new = _pot_after_bet(pot, amount_to_call)
        history.append(ActionType.CALL)
        return GameState(
            pot=pot_new,
            stacks=stacks_new,
            history=tuple(history),
            current_player=current,
            last_bet=last_bet,
            board=state.board,
        )

    if action in (ActionType.RAISE_50, ActionType.RAISE_100):
        if last_bet <= 0:
            raise ValueError("Cannot raise without an outstanding bet.")
        amount_to_call = min(last_bet, state.stacks[idx_current])
        max_total = _max_pot_total_bet(pot, amount_to_call, state.stacks[idx_current])
        raise_component = max_total - amount_to_call
        if raise_component <= 0:
            raise ValueError("No room to raise under pot-limit.")
        if action is ActionType.RAISE_50:
            total_bet = amount_to_call + max(1, int(round(0.5 * raise_component)))
        else:
            total_bet = max_total
        if total_bet > state.stacks[idx_current]:
            total_bet = state.stacks[idx_current]
        additional_amount = total_bet
        stacks_new = update_stacks(state.stacks, idx_current, additional_amount)
        pot_new = _pot_after_bet(pot, additional_amount)
        history.append(action)
        return GameState(
            pot=pot_new,
            stacks=stacks_new,
            history=tuple(history),
            current_player=Player.OOP,  # back to OOP after IP raises
            last_bet=total_bet,
            board=state.board,
        )

    raise ValueError(f"Unhandled action type: {action}")


def _bet_amount_from_action(action: Action, pot: int, stack: int) -> int:
    if action is ActionType.BET_25:
        size = int(round(0.25 * pot))
    elif action is ActionType.BET_100:
        size = pot
    else:
        raise ValueError(f"Unsupported bet action: {action}")
    return min(size, stack)


def showdown_payoff(
    state: GameState,
    oop_hole: Sequence[int],
    ip_hole: Sequence[int],
) -> float:
    """
    Compute the payoff from the perspective of OOP (player 0) at showdown.
    """
    result = compare_omaha_hands(oop_hole, ip_hole, state.board)
    if result == 0:
        return 0.0
    # Pot represents the total chips already in the middle.
    if result > 0:
        return float(state.pot / 2)
    return float(-state.pot / 2)


def terminal_payoff(
    state: GameState,
    oop_hole: Sequence[int],
    ip_hole: Sequence[int],
) -> float:
    """
    Compute terminal payoff for OOP (player 0) given a terminal state.
    """
    if not state.is_terminal:
        raise ValueError("terminal_payoff called on non-terminal state.")

    last_action = state.history[-1]
    if last_action is ActionType.FOLD:
        # The player who did not fold wins the pot.
        folding_player = state.current_player
        if folding_player is Player.OOP:
            return float(-state.pot / 2)
        return float(state.pot / 2)

    # Call or check-check leads to showdown.
    return showdown_payoff(state, oop_hole, ip_hole)


__all__ = [
    "Player",
    "ActionType",
    "Action",
    "GameState",
    "initial_state",
    "legal_actions",
    "apply_action",
    "terminal_payoff",
]

