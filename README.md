# ♠️♦️♣️♥️ Realistic CLI Texas Hold'em Trainer

This project provides a **command-line interface (CLI)** tool for practicing **4-handed No-Limit Texas Hold'em** cash game strategy.  
It simulates a realistic **micro-stakes online cash game** environment focused on positional awareness, opponent tendencies, and equity-based decision making.

---

## 🚀 Key Features

- **Realistic 4-Max Gameplay:**  
  Simulates a 4-handed table (UTG, BU, SB, BB).

- **Full Positional Logic:**  
  Correctly rotates the dealer button, posts blinds, and determines action order.

- **Stochastic Opponent AI:**  
  Opponents have selectable profiles (`tight`, `loose`, `maniac`):
  - Pre-flop range tables (open, call, 3-bet)
  - Post-flop frequencies (c-bet, fold to c-bet, check-raise)

- **Independent Stacks & Rebuys:**  
  Players manage stacks individually; busted players automatically rebuy.

- **Range-Based Equity Calculation:**  
  Hero's equity is calculated against **estimated opponent ranges**, not random hands.

- **Realistic Bet Sizing:**
  - Hero offered realistic bet options: **Min**, **½ Pot**, **¾ Pot**, **Pot**, **All-in**
  - Pot odds shown next to call
  - Custom raise amounts allowed

- **Fold Equity Estimation:**  
  Displays estimated **fold equity** based on raise size and opponent profiles.

- **Session Statistics Tracking:**
  - **VPIP** (% of hands played)
  - **PFR** (Pre-flop Raise %)
  - **AF** (Aggression Factor)
  - **Net winnings** graph at end of session

- **Colourful CLI Interface:**  
  Hearts/diamonds in red ♥♦, spades/clubs in blue/green ♠♣ using ANSI codes.

---

## 🎮 Demo Output

```text
Villain 1 (SB) posts 10
Villain 2 (BB) posts 20

--- Starting Hand #8 ---
Dealer is Hero

======================================================================
Hero(BU)D: 2000 | Villain 1(SB): 1990 | Villain 2(BB): 1980 | Villain 3(UTG): 2000
----------------------------------------------------------------------
Pot: 60  | Board: --
Your Hand: Q♣ J♣
Equity vs Range (3 opp): 22.1% Win / 0.4% Tie / 77.5% Lose (1000 sims, 0.14s)
----------------------------------------------------------------------
Action on Villain 3 (UTG) Stack: 2000 To Call: 20
Villain 3 raises to 60

======================================================================
Hero(BU)D: 2000 | Villain 1(SB): 1990 | Villain 2(BB): 1980 | Villain 3(UTG): 1940
----------------------------------------------------------------------
Pot: 150  | Board: --
Your Hand: Q♣ J♣
Equity vs Range (3 opp): 15.5% Win / 0.8% Tie / 83.7% Lose (1000 sims, 0.13s)
----------------------------------------------------------------------
Action on Hero (BU) Stack: 2000 To Call: 60
Your options:
  fold (f) | call 60 (c) [Odds: 29%] | raise 180 (min, r1) [FE: 25%] |
  raise 195 (½ pot, r2) [FE: 40%] | raise 251 (¾ pot, r3) [FE: 50%] |
  raise 310 (pot, r4) [FE: 60%] | raise 2000 (all-in, r5) [FE: 98%]

> Your move: _
```

---

## 📦 Requirements

- Python **≥ 3.8**
- `treys` library (for fast hand evaluation)

Install with:

```bash
pip install treys
```

---

## ⚙️ Installation

Clone the repository:

```bash
# Using Git
git clone https://github.com/<your-username>/poker_trainer.git
cd poker_trainer
```

Or simply download the script manually.

---

## 🛠 Command-Line Options

| Flag | Description |
|:--|:--|
| `--hands N` | Play exactly **N** hands then exit (with session stats). |
| `--bankroll N` | Set starting stack to **N** chips (default: 2000). |
| `--bb N` | Set Big Blind to **N** (default: 20). Small Blind = BB/2. |
| `--villain [tight\|loose\|maniac]` | Set opponent AI profile (default: tight). |
| `--sims N` | Number of Monte Carlo simulations per decision (default: 1000). |
| `--seed N` | Use a random seed for reproducible hands/shuffles. |
| `--help` | Display available options. |

Example usage:

```bash
python poker_trainer.py --hands 50 --villain loose --sims 1500
```

---

## 🎯 Gameplay Controls

When it's your turn ("Action on Hero..."), you'll be given move options.

Type one of:

| Move | Input |
|:--|:--|
| Fold | `f` or `fold` |
| Check (if no bet) | `x` or `check` |
| Call | `c` or `call` |
| Raise | `r1`, `r2`, `r3`, etc. (preset sizes) or `r(amount)` for custom |

**Notes:**
- Invalid inputs will re-prompt.
- Press **Ctrl+C** at any time to exit and view session stats.

---

## 📊 Session Summary

At session end, you'll see:

- Hands played
- Good Decisions %
- VPIP, PFR, AF
- Net winnings graph

---



