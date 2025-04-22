# Realistic CLI Texas Hold'em Trainer ♠️♦️♣️♥️

This project provides a command-line interface (CLI) tool for practicing 4-handed No-Limit Texas Hold'em cash game strategy. It aims to simulate a micro-stakes online cash game environment with realistic gameplay mechanics, focusing on positional awareness, opponent tendencies, and equity calculations.

This script is an evolution of a simpler poker simulator, incorporating several key enhancements for a more effective training experience.

## Key Features

*   **Realistic 4-Max Gameplay:** Simulates a 4-handed table (UTG, BU, SB, BB).
*   **Full Positional Logic:** Correctly rotates the dealer button, posts blinds (Small Blind, Big Blind), and determines the order of action on each street. Player positions are clearly displayed.
*   **Stochastic Opponent AI:** Features three distinct opponent profiles (`tight`, `loose`, `maniac`) selectable via command-line flag. Opponents make decisions based on:
    *   Basic pre-flop range tables (open, call, 3-bet ranges based on profile).
    *   Simple post-flop frequencies (c-bet, fold vs c-bet, check-raise based on profile).
*   **Independent Stacks & Rebuys:** Each player manages their stack independently. Busted players automatically rebuy to the starting bankroll.
*   **Range-Based Equity Calculation:** Utilizes the `treys` library for fast hand evaluation and Monte Carlo simulation. Crucially, Hero's equity is calculated against estimated opponent ranges (derived from their profile and pre-flop actions), not just random hands. Win/Tie/Loss percentages are displayed each street.
*   **Realistic Bet Sizing:**
    *   Hero is prompted with common bet/raise sizes: Minimum, ½ Pot, ¾ Pot, Pot size, and All-in.
    *   Pot odds required to call are displayed alongside the call action.
    *   Allows custom raise amounts.
*   **Fold Equity Estimation:** Displays an estimated fold equity percentage alongside Hero's raise options, based on opponent profiles and the size of the raise relative to the pot.
*   **Session Statistics:** Tracks key stats for Hero and each Villain throughout the session:
    *   **VPIP:** Voluntarily Put Money In Pot (%)
    *   **PFR:** Pre-Flop Raise (%)
    *   **AF:** Post-Flop Aggression Factor (Bets+Raises / Calls)
    *   **Net:** Total chips won or lost during the session.
    *   A summary report and simple ASCII winnings graph are shown at the end.
*   **Configurable Settings:** Control various parameters via command-line flags:
    *   Number of hands (`--hands`)
    *   Starting bankroll (`--bankroll`)
    *   Big Blind amount (`--bb`)
    *   Villain profile (`--villain`)
    *   Monte Carlo simulation count (`--sims`)
    *   Random seed for reproducibility (`--seed`)
*   **Colourful CLI:** Uses ANSI escape codes for coloured suits and key information, maintaining a clear and engaging interface.

## Demo Output

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
Equity vs Range (3 opp): 22.1% Win /  0.4% Tie / 77.5% Lose (1000 sims, 0.14s)
----------------------------------------------------------------------
Action on Villain 3 (UTG) Stack: 2000 To Call: 20
Villain 3 raises to 60

======================================================================
Hero(BU)D: 2000 | Villain 1(SB): 1990 | Villain 2(BB): 1980 | Villain 3(UTG): 1940
----------------------------------------------------------------------
Pot: 150  | Board: --
Your Hand: Q♣ J♣
Equity vs Range (3 opp): 15.5% Win /  0.8% Tie / 83.7% Lose (1000 sims, 0.13s)
----------------------------------------------------------------------
Action on Hero (BU) Stack: 2000 To Call: 60
Your options:
  fold (f) | call 60 (c) [Odds: 29%] | raise 180 (min, r1) [FE: 25%] | raise 195 (½ pot, r2) [FE: 40%] | raise 251 (¾ pot, r3) [FE: 50%] | raise 310 (pot, r4) [FE: 60%] | raise 2000 (all-in, r5) [FE: 98%]
> Your move: _
