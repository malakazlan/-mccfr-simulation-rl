# River-Only MCCFR Solver for 4-Card PLO

A Python-based **Monte Carlo Counterfactual Regret Minimization (MCCFR)** engine for the **river street** in **Heads-Up 4-card Pot-Limit Omaha (PLO)**.

## Overview

- **Game**: 4-card PLO — each hand uses exactly **2 hole cards** and **3 board cards**.
- **Situation**: Heads-up (OOP vs IP), **river only**.
- **Pot**: Starting pot = 50, effective stacks = 50.
- **Algorithm**: MCCFR with external sampling; regret matching; average strategy and EV/volatility tracking.

## Requirements

- Python 3.10+
- See [requirements.txt](requirements.txt): `deuces`, `numpy`, `numba` (optional for speed).

## Installation

```bash
git clone https://github.com/malakazlan/-mccfr-simulation-rl.git
cd -mccfr-simulation-rl
pip install -r requirements.txt
```

## Usage

### CLI

```bash
python main.py --board AsKh7d4c2h --oop-range sample_oop.txt --ip-range sample_ip.txt [options]
```

**Required**

| Argument        | Description                                      |
|----------------|--------------------------------------------------|
| `--board`      | 5 board cards as 10 characters, e.g. `AsKh7d4c2h` |
| `--oop-range` | Path to OOP 4-card range file                    |
| `--ip-range`  | Path to IP 4-card range file                     |

**Optional**

| Argument              | Default | Description                              |
|-----------------------|---------|------------------------------------------|
| `--seconds`           | 30      | Time budget (seconds)                    |
| `--min-iterations`   | 1000    | Minimum MCCFR iterations                 |
| `--volatility-target`| 0.1     | Stop when volatility below this         |
| `--seed`             | —       | Random seed for reproducibility          |
| `--quiet`            | false   | Print only global metrics                |

### Example

```bash
python main.py --board AsKh7d4c2h --oop-range sample_oop.txt --ip-range sample_ip.txt --seconds 10 --quiet
```

Output includes **EV per player (BB/100)**, **simulation volatility**, **iterations**, and (unless `--quiet`) **strategy frequency** and **cumulative regret** per infoset/action.

## Range File Format

Plain text, one 4-card hand per line. Each line:

- **HAND**: 8 characters (4 cards × 2 chars), e.g. `AsKdQcJh`.
- **Weight** (optional): space then a number (default 1.0).

Card codes: rank `2-9`, `T`, `J`, `Q`, `K`, `A` + suit `s`, `h`, `d`, `c`.

```
# comments allowed
AsKdQcJh 1.0
Th9s8h7h
```

Hands must not use any of the 5 board cards (removal is applied automatically).

## Betting Tree

- **OOP**: Check, Bet 25% pot, Bet 100% pot.
- **IP vs bet**: Fold, Call, Raise 50%, Raise 100%.

All raise sizes respect **strict pot-limit** mechanics.

## Project Structure

| File           | Role                                                |
|----------------|-----------------------------------------------------|
| `engine.py`    | MCCFR trainer, regret matching, EV/volatility logic  |
| `game.py`      | PLO rules, pot-limit math, `GameState`, payoffs     |
| `evaluator.py` | 5-card hand evaluation; Omaha 2-from-4 + 3-from-5  |
| `ranges.py`    | Parse and sample weighted 4-card distributions     |
| `main.py`      | CLI: board/ranges in, results and metrics out        |

## Validation (MonkerSolver)

To compare strategy and EV with **MonkerSolver**: use the same board string and equivalent ranges. Run this solver, then run MonkerSolver on the same inputs and compare EV and key strategy frequencies (manual comparison; no API). Strategy and EV should be within about 5% for the same ranges/board.

## License

Use and modify as needed for your project.
