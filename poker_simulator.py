#!/usr/bin/env python3
"""
Realistic 4-Hand Texas Hold'em CLI Trainer
==========================================
* Full positional logic (BU, SB, BB, UTG) with rotating button.
* Stochastic opponents (tight, loose, maniac profiles via --villain).
* Range-based Monte Carlo equity calculation.
* Realistic bet sizing (½-pot, ¾-pot, pot, all-in) and pot odds display.
* Fold equity estimates displayed.
* Fast hand evaluation using the 'treys' library.
* Session stats tracking (VPIP, PFR, AF) for all players.
* Configurable settings (--hands, --bankroll, --seed, --profile).
* Colourful CLI output preserved.
"""
from __future__ import annotations

import argparse
import random
import sys
import math
import time
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Set, Union, TypeAlias
import itertools

# Use treys for hand evaluation
try:
    from treys import Card, Deck as TreysDeck, Evaluator
except ImportError:
    print("Error: 'treys' library not found. Please install it: pip install treys")
    sys.exit(1)

# ────────── Constants ──────────
RANKS = "23456789TJQKA"
SUITS = "cdhs"  # clubs diamonds hearts spades
SUIT_SYM = {"c": "♣", "d": "♦", "h": "♥", "s": "♠"}
ANSI = {
    "♠": "\033[94m", # bright blue
    "♣": "\033[92m", # bright green
    "♥": "\033[91m", # bright red
    "♦": "\033[91m", # bright red
    "reset": "\033[0m",
    "grey": "\033[90m",
    "yellow": "\033[93m",
    "cyan": "\033[96m",
    "magenta": "\033[95m",
    "bold": "\033[1m",
    "underline": "\033[4m",
}
POSITIONS_4MAX = {0: "BU", 1: "SB", 2: "BB", 3: "UTG"}
NUM_PLAYERS = 4 # Fixed for 4-max as requested

# Type Aliases
CardTuple: TypeAlias = Tuple[str, str] # ('A', 's')
HoleCards: TypeAlias = Tuple[str, str] # ('As', 'Kc')
CardInt: TypeAlias = int              # treys Card representation
Range: TypeAlias = Set[str]           # Set of canonical hand strings like 'AKs', '77', 'T9o'

# ────────── Card & Hand Helpers ──────────

def pc(card: Union[str, CardInt]) -> str:
    """Pretty print card with ANSI color."""
    if isinstance(card, int):
        card_str = Card.int_to_str(card)
    else:
        card_str = card # Assume format like 'As', 'Td'
    rank = card_str[0]
    suit = card_str[1]
    sym = SUIT_SYM[suit]
    color = ANSI[sym]
    return f"{color}{rank}{sym}{ANSI['reset']}"

def join(cards: List[Union[str, CardInt]]) -> str:
    """Join a list of cards into a formatted string."""
    return " ".join(pc(c) for c in cards)

def str_to_treys(card_str: str) -> CardInt:
    """Convert 'As'/'Td' format to treys integer."""
    return Card.new(card_str)

def hole_cards_to_treys(cards: HoleCards) -> List[CardInt]:
    return [str_to_treys(c) for c in cards]

def hole_cards_to_str(cards: HoleCards) -> str:
    return "".join(sorted(cards, key=lambda c: (RANKS.index(c[0]), c[1]), reverse=True))

def canonical_form(card1: str, card2: str) -> str:
    """Return canonical hand form ('AKs', 'T9o', '77')."""
    r1, s1 = card1[0], card1[1]
    r2, s2 = card2[0], card2[1]
    idx1, idx2 = RANKS.index(r1), RANKS.index(r2)
    if idx1 < idx2:
        r1, r2 = r2, r1
        s1, s2 = s2, s1
    if r1 == r2:
        return r1 + r2 # Pair
    elif s1 == s2:
        return r1 + r2 + 's' # Suited
    else:
        return r1 + r2 + 'o' # Offsuit


# ────────── Hand Evaluation (using treys) ──────────
EVALUATOR = Evaluator()
HAND_RANKS = {
    1: "Straight Flush", 2: "Quads", 3: "Full House", 4: "Flush", 5: "Straight",
    6: "Trips", 7: "Two Pair", 8: "One Pair", 9: "High Card"
}

def evaluate_hand(hole_cards: List[CardInt], board: List[CardInt]) -> Tuple[int, str, int]:
    """Evaluate the best 5-card hand from 7 cards using treys."""
    if not hole_cards: return (9999, "Folded", 9999)
    hand_int = EVALUATOR.evaluate(hole_cards, board)
    rank_class = EVALUATOR.get_rank_class(hand_int)
    class_str = HAND_RANKS.get(rank_class, "Unknown")
    return hand_int, class_str, rank_class # Lower score is better

def get_rank_percentage(hand_strength: int) -> float:
    """Convert treys hand score to a percentile (0-1)."""
    # Max score for High Card is 7462. Lower is better.
    # Approximate percentile - lower score = higher percentile
    return max(0.0, min(1.0, 1.0 - (hand_strength / 7462.0)))

# ────────── Opponent Profiles & Ranges ──────────

# Simplified pre-flop ranges (expand these significantly for real strategy)
# Format: Set of canonical hands ('AKs', 'T9o', '77')
# TODO: These ranges are placeholders and need refinement for realistic play.
PREFLOP_RANGES = {
    "tight": {
        "UTG_open": {'AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', 'AKs', 'AQs', 'AJs', 'KQs', 'AKo', 'AQo'},
        "BU_open": {'AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', '66', '55', 'AKs', 'AQs', 'AJs', 'ATs', 'A9s', 'A8s', 'A7s', 'A6s', 'A5s', 'KQs', 'KJs', 'KTs', 'QJs', 'QTs', 'JTs', 'T9s', '98s', 'AKo', 'AQo', 'AJo', 'ATo', 'KQo'},
        "call_range": {'QQ', 'JJ', 'TT', '99', '88', '77', 'AKs', 'AQs', 'AJs', 'KQs', 'AKo', 'AQo'}, # Example call vs open
        "3bet_range": {'AA', 'KK', 'QQ', 'AKs', 'AKo'} # Example 3bet vs open
    },
    "loose": {
        "UTG_open": {'AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', '66', 'AKs', 'AQs', 'AJs', 'ATs', 'A9s', 'KQs', 'KJs', 'KTs', 'QJs', 'JTs', 'T9s', 'AKo', 'AQo', 'AJo', 'KQo'},
        "BU_open": {'AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', '66', '55', '44', '33', '22', 'Axs', 'K9s+', 'Q9s+', 'J9s+', 'T8s+', '97s+', '86s+', '76s', '65s', '54s', 'Axo', 'KTo+', 'QTo+', 'JTo'}, # '+' means and higher rank
        "call_range": {'JJ', 'TT', '99', '88', '77', '66', '55', 'AQs', 'AJs', 'ATs', 'KQs', 'KJs', 'QJs', 'JTs', 'T9s', '98s', 'AQo', 'AJo', 'KQo'},
        "3bet_range": {'AA', 'KK', 'QQ', 'JJ', 'AKs', 'AQs', 'A5s', 'A4s', 'KQs', 'AKo', 'AQo'} # Includes bluffs
    },
    "maniac": {
        "UTG_open": {'Any Pair', 'Axs', 'K8s+', 'Q9s+', 'J9s+', 'T8s+', '97s+', '87s', '76s', 'Axo', 'KTo+', 'QTo+', 'JTo'},
        "BU_open": {'Any Hand'}, # Simplified - nearly ATC
        "call_range": {'Any Pair', 'Any Suited Ace', 'K9s+', 'Q9s+', 'J9s+', 'T8s+', 'Any Suited Connector', 'ATo+', 'KJo+', 'QJo'},
        "3bet_range": {'AA', 'KK', 'QQ', 'JJ', 'TT', '99', 'AKs', 'AQs', 'AJs', 'KQs', 'KJs', 'QJs', 'A5s', 'A4s', 'K9s', 'T9s', '98s', '87s', '76s', 'AKo', 'AQo', 'AJo', 'KQo', 'KJo'} # Wide value + bluffs
    }
}

# Expand simplified range notation (like 'Axs', 'KTo+', '77+', 'Any Pair', 'Any Hand')
def expand_range_notation(simple_range: Set[str]) -> Range:
    """Expands shorthand notation like 'K9s+' into specific canonical hands."""
    expanded = set()
    ranks_list = list(RANKS)
    for item in simple_range:
        if item == 'Any Hand':
            for i in range(len(ranks_list)):
                for j in range(i, len(ranks_list)):
                    r1, r2 = ranks_list[i], ranks_list[j]
                    if i == j:
                        expanded.add(r1 + r2)
                    else:
                        expanded.add(r2 + r1 + 's')
                        expanded.add(r2 + r1 + 'o')
            return expanded # Optimization: Any Hand covers everything

        if item == 'Any Pair':
            for r in ranks_list: expanded.add(r + r)
            continue

        if item.endswith('+'):
            base = item[:-1]
            r1, r2 = base[0], base[1]
            suited = 's' if base.endswith('s') else ('o' if base.endswith('o') else '')
            if not suited and r1 == r2: # Pair like '77+'
                start_idx = ranks_list.index(r1)
                for i in range(start_idx, len(ranks_list)):
                    expanded.add(ranks_list[i] * 2)
            elif suited: # Suited or offsuit like K9s+ or KTo+
                r1_idx, r2_idx = ranks_list.index(r1), ranks_list.index(r2)
                if suited == 's':
                     for i in range(r2_idx, r1_idx): # K9s, KTs, KJs, KQs (r1 is higher rank)
                         expanded.add(r1 + ranks_list[i] + 's')
                else: # suited == 'o'
                    for i in range(r2_idx, r1_idx): # KTo, KJo, KQo (r1 is higher rank)
                         expanded.add(r1 + ranks_list[i] + 'o')
            # '+' without s/o is ambiguous, ignore for now
        elif item == 'Axs': # Any suited Ace
            for r in ranks_list[:-1]: expanded.add('A' + r + 's')
        elif item == 'Axo': # Any offsuit Ace
             for r in ranks_list[:-1]: expanded.add('A' + r + 'o')
        elif item == 'Any Suited Connector':
             for i in range(len(ranks_list) - 1):
                 expanded.add(ranks_list[i+1] + ranks_list[i] + 's')
        else: # Specific hand like 'AKs', '77', 'T9o'
            expanded.add(item)

    return expanded

# Precompute expanded ranges
EXPANDED_RANGES: Dict[str, Dict[str, Range]] = {}
for profile, ranges in PREFLOP_RANGES.items():
    EXPANDED_RANGES[profile] = {}
    for range_type, simple_range in ranges.items():
        # Handle special cases like BU open for maniac
        if profile == "maniac" and range_type == "BU_open":
             # Approximate 70% range for 'nearly ATC' for performance
             EXPANDED_RANGES[profile][range_type] = expand_range_notation({'77+', 'A2s+', 'K6s+', 'Q8s+', 'J8s+', 'T8s+', '97s+', '86s+', '75s+', '64s+', '54s', 'A8o+', 'K9o+', 'QTo+', 'JTo'})
        else:
             EXPANDED_RANGES[profile][range_type] = expand_range_notation(simple_range)


# Post-flop tendencies (simple frequencies)
POSTFLOP_TENDENCIES = {
    "tight": {"cbet": 0.5, "fold_vs_cbet": 0.6, "check_raise": 0.05, "fold_vs_raise_half_pot": 0.4, "fold_vs_raise_pot": 0.6},
    "loose": {"cbet": 0.7, "fold_vs_cbet": 0.4, "check_raise": 0.10, "fold_vs_raise_half_pot": 0.25, "fold_vs_raise_pot": 0.4},
    "maniac": {"cbet": 0.85, "fold_vs_cbet": 0.3, "check_raise": 0.20, "fold_vs_raise_half_pot": 0.15, "fold_vs_raise_pot": 0.25},
}

# ────────── Stats Tracking ──────────

class StatsTracker:
    def __init__(self):
        self.hands_played = 0
        # Per-player stats: [Hero, P1, P2, P3]
        self.vpip_opportunities = [0] * NUM_PLAYERS
        self.vpip_actions = [0] * NUM_PLAYERS
        self.pfr_opportunities = [0] * NUM_PLAYERS
        self.pfr_actions = [0] * NUM_PLAYERS
        self.aggression_bets_raises = [0] * NUM_PLAYERS # Post-flop B+R
        self.aggression_calls = [0] * NUM_PLAYERS       # Post-flop Calls
        self.winnings = [0.0] * NUM_PLAYERS # Track net win/loss per player

    def record_preflop_action(self, player_idx: int, action: str, voluntary: bool, can_raise: bool):
        if voluntary:
            self.vpip_opportunities[player_idx] += 1
            if action in ['call', 'raise', 'bet']: # bet implies open raise
                self.vpip_actions[player_idx] += 1
        if can_raise:
            self.pfr_opportunities[player_idx] += 1
            if action == 'raise':
                self.pfr_actions[player_idx] += 1

    def record_postflop_action(self, player_idx: int, action: str):
        if action in ['bet', 'raise']:
            self.aggression_bets_raises[player_idx] += 1
        elif action == 'call':
            self.aggression_calls[player_idx] += 1
        # Checks and folds don't count towards Aggression Factor

    def record_win(self, player_idx: int, amount: int):
         self.winnings[player_idx] += amount

    def get_stats(self, player_idx: int) -> Dict[str, float]:
        vpip = (self.vpip_actions[player_idx] / self.vpip_opportunities[player_idx] * 100
                if self.vpip_opportunities[player_idx] > 0 else 0)
        pfr = (self.pfr_actions[player_idx] / self.pfr_opportunities[player_idx] * 100
               if self.pfr_opportunities[player_idx] > 0 else 0)
        af = (self.aggression_bets_raises[player_idx] / self.aggression_calls[player_idx]
              if self.aggression_calls[player_idx] > 0 else float('inf') if self.aggression_bets_raises[player_idx] > 0 else 0)
        # Handle infinite AF for display
        af_display = "inf" if af == float('inf') else f"{af:.1f}"

        return {
            "VPIP": f"{vpip:.1f}",
            "PFR": f"{pfr:.1f}",
            "AF": af_display,
            "Net": self.winnings[player_idx]
        }

    def report(self, player_names: List[str]):
        print("\n" + "=" * 60)
        print(f"Session Stats ({self.hands_played} hands)")
        print("-" * 60)
        print(f"{'Player':<10} {'VPIP':>6} {'PFR':>6} {'AF':>6} {'Net':>10}")
        print("-" * 60)
        for i, name in enumerate(player_names):
            stats = self.get_stats(i)
            print(f"{name:<10} {stats['VPIP']:>6} {stats['PFR']:>6} {stats['AF']:>6} {stats['Net']:>10.0f}")
        print("=" * 60)

        # Simple ASCII Graph (Net Winnings)
        print("\nNet Winnings Over Session:")
        max_abs_win = max(abs(w) for w in self.winnings) if any(w != 0 for w in self.winnings) else 1
        scale = 40 / max_abs_win if max_abs_win > 0 else 1

        for i, name in enumerate(player_names):
             win = self.winnings[i]
             bar_len = int(abs(win) * scale)
             bar = ('#' * bar_len) if win >= 0 else ('-' * bar_len)
             print(f"{name:>8}: {win:>6.0f} |{bar}")
        print("-" * 60)


# ────────── Player Class ──────────

class Player:
    def __init__(self, name: str, stack: int, profile: str = "tight", is_hero: bool = False):
        self.name = name
        self.initial_stack = stack
        self.stack = stack
        self.profile = profile
        self.is_hero = is_hero
        self.hole_cards: Optional[HoleCards] = None
        self.treys_cards: List[CardInt] = []
        self.position: str = "" # BU, SB, BB, UTG
        self.in_hand: bool = True
        self.is_all_in: bool = False
        self.current_bet: int = 0 # Amount bet *in the current round*
        self.total_bet_in_pot: int = 0 # Total contributed to the pot this hand
        self.last_action: Optional[str] = None
        self.preflop_range: Range = set() # Estimated range based on preflop action

    def post_blind(self, amount: int) -> int:
        posted = min(amount, self.stack)
        self.stack -= posted
        self.current_bet = posted
        self.total_bet_in_pot = posted
        if self.stack == 0:
            self.is_all_in = True
        return posted

    def can_bet(self) -> bool:
        return self.in_hand and not self.is_all_in and self.stack > 0

    def make_decision(self, game_state: 'GameState', stats: StatsTracker) -> Tuple[str, int]:
        """AI decision logic for opponents."""
        if self.is_hero:
            raise RuntimeError("Hero should not use AI decision logic")

        street = game_state.street
        to_call = game_state.current_bet - self.current_bet
        pot_size = game_state.pot
        tendencies = POSTFLOP_TENDENCIES[self.profile]
        player_idx = game_state.players.index(self)

        # Pre-flop Logic
        if street == "Pre-flop":
            can_raise = any(p.can_bet() for p in game_state.players if p != self) # Can someone else raise? Crude check.
            hand_canon = canonical_form(self.hole_cards[0], self.hole_cards[1])
            ranges = EXPANDED_RANGES[self.profile]
            pos_simple = self.position if self.position in ['UTG', 'BU'] else 'BLIND' # Group SB/BB for simple ranges
            open_range = ranges.get(f"{pos_simple}_open", ranges["BU_open"]) # Default to BU range if specific pos missing
            call_range = ranges["call_range"]
            three_bet_range = ranges["3bet_range"]

            action = "fold"
            amount = 0

            # Determine preflop range based on action we *would* take if first to act cleanly
            initial_pf_range = set()
            if hand_canon in open_range:
                 initial_pf_range = open_range
            # Refine range based on actual actions taken
            # This is complex - simplifying: assume they continue if hand in *some* reasonable range

            if to_call == 0: # Option to check (BB) or open
                if hand_canon in open_range:
                    action = "raise" # Simple: always raise if in opening range
                    # Simple raise sizing: 3x BB
                    amount = game_state.bb_amount * 3
                    self.preflop_range = open_range # Assume opening range
                    stats.record_preflop_action(player_idx, action, True, True)
                else:
                    action = "check" if self.position == "BB" else "fold"
                    self.preflop_range = set() # Folded pre
                    stats.record_preflop_action(player_idx, action, action=='check', True)

            else: # Facing a bet/raise
                 # TODO: More nuanced logic based on who raised, size, position
                 should_call = hand_canon in call_range
                 should_3bet = hand_canon in three_bet_range

                 if should_3bet:
                     action = "raise"
                     # Simple 3-bet sizing: 3x the previous raise amount
                     amount = game_state.current_bet * 3 # Approximation
                     self.preflop_range = three_bet_range # Assume 3bet range
                 elif should_call:
                     action = "call"
                     amount = to_call
                     # Simplification: If they call, their range could be call_range OR slowplayed 3bet_range
                     self.preflop_range = call_range.union(three_bet_range)
                 else:
                     action = "fold"
                     amount = 0
                     self.preflop_range = set()

                 stats.record_preflop_action(player_idx, action, action != 'fold', True) # Can always raise facing bet

            # Ensure bet is valid
            amount = min(amount, self.stack + self.current_bet) # Max is all-in raise
            if action == "raise" and amount <= game_state.current_bet: # Not a legal raise
                 action = "call"
                 amount = to_call
            if action == "call" and amount > self.stack: # Cannot afford call
                action = "fold"
                amount = 0
            if action == "fold": self.preflop_range = set()


        # Post-flop Logic (Simplified frequency-based)
        else:
            # Estimate hand strength (simple equity vs random for now)
            # TODO: Replace with equity vs estimated opponent ranges for better decisions
            win_pct, _, _ = calculate_equity(self.treys_cards, game_state.board_cards,
                                             game_state.get_active_opponents(self),
                                             sims=500, # Fewer sims for AI decisions
                                             opponent_ranges=None) # Use None for random opponent hands in AI quick eval

            action = "check"
            amount = 0

            if to_call > 0: # Facing a bet
                pot_odds = to_call / (pot_size + to_call) if (pot_size + to_call) > 0 else 0
                required_equity = pot_odds

                fold_freq = tendencies["fold_vs_cbet"] # Simplification
                if random.random() < fold_freq and win_pct < 0.6: # Fold sometimes based on profile unless strong
                    action = "fold"
                elif win_pct >= required_equity * 0.9: # Call if equity is close to pot odds (with slight margin)
                    action = "call"
                    amount = to_call
                else:
                     action = "fold" # Default to fold if not meeting odds or profile fold freq

                # Consider check-raising?
                if action == "call" and random.random() < tendencies["check_raise"] and win_pct > 0.6:
                     action = "raise"
                     # Simple raise: Pot size
                     amount = pot_size + to_call # Bet the pot *after* calling
            else: # Option to check or bet
                cbet_freq = tendencies["cbet"]
                if random.random() < cbet_freq and win_pct > 0.4: # C-bet if profile dictates and hand has some potential
                    action = "bet"
                    # Simple bet sizing: 2/3 pot
                    amount = int(pot_size * 0.66)
                else:
                    action = "check"

            # Record postflop action for stats
            if action != 'fold': stats.record_postflop_action(player_idx, action)

            # Ensure amount is valid and capped at stack
            amount = max(0, amount) # Cannot bet negative
            if action in ["bet", "raise"]:
                 amount = max(game_state.bb_amount, amount) # Min bet is big blind postflop
                 effective_bet = amount + self.current_bet # Total amount for this player's bet
                 if effective_bet <= game_state.current_bet and action == "raise": # Check if raise is legal
                      action = "call"
                      amount = to_call # Revert to call if raise is too small
                 amount = min(amount, self.stack) # Cap bet/raise at available stack


            if action == "call":
                 amount = min(to_call, self.stack) # Cap call amount at stack

        # Final check for all-in
        if action == "call" and amount == self.stack:
            self.is_all_in = True
        elif action in ["bet", "raise"] and amount == self.stack:
             action = "raise" # Treat all-in bet as a raise action type
             self.is_all_in = True
             # Ensure the 'amount' reflects the total chips going in *beyond* the current bet level
             amount = self.stack # The amount *added* to the pot is the remaining stack


        return action, amount

    def reset_for_new_hand(self, stack: Optional[int] = None):
        if stack is not None: self.stack = stack
        if self.stack <= 0: # Handle bust out / rebuy
            print(f"{ANSI['yellow']}{self.name} busted! Rebuying to {self.initial_stack}{ANSI['reset']}")
            self.stack = self.initial_stack # Simple rebuy model
            # TODO: Add option to sit out

        self.hole_cards = None
        self.treys_cards = []
        self.in_hand = True
        self.is_all_in = False
        self.current_bet = 0
        self.total_bet_in_pot = 0
        self.last_action = None
        self.preflop_range = set()


# ────────── Game State & Logic ──────────

class GameState:
    def __init__(self, players: List[Player], bb_amount: int, seed: Optional[int] = None):
        self.players = players
        self.num_players = len(players)
        self.sb_amount = bb_amount // 2
        self.bb_amount = bb_amount
        self.dealer_pos_idx = random.randrange(self.num_players) if seed is None else random.Random(seed).randrange(self.num_players)
        self.deck = TreysDeck()
        self.board_cards: List[CardInt] = []
        self.pot = 0
        self.side_pots: List[Dict[str, Union[int, List[Player]]]] = [] # [{'amount': int, 'eligible': List[Player]}]
        self.current_bet = 0 # The highest bet amount placed in the current round
        self.last_raiser: Optional[Player] = None
        self.action_idx: int = 0 # Index of player whose turn it is
        self.street: str = "Pre-flop" # Pre-flop, Flop, Turn, River
        self.hand_over = False
        self.stats: Optional[StatsTracker] = None # Assigned externally

    def setup_new_hand(self):
        self.deck = TreysDeck()
        self.board_cards = []
        self.pot = 0
        self.side_pots = []
        self.current_bet = 0
        self.last_raiser = None
        self.street = "Pre-flop"
        self.hand_over = False

        # Rotate dealer button
        self.dealer_pos_idx = (self.dealer_pos_idx + 1) % self.num_players

        # Reset players
        for player in self.players:
            player.reset_for_new_hand() # Keep existing stack unless busted

        # Assign positions relative to dealer
        for i in range(self.num_players):
            player_idx = (self.dealer_pos_idx + 1 + i) % self.num_players
            pos_offset = i + 1 # 1=SB, 2=BB, 3=UTG (for 4-max)
            self.players[player_idx].position = POSITIONS_4MAX.get(pos_offset % self.num_players, f"P{pos_offset}")

        # Deal cards
        for player in self.players:
             cards = tuple(self.deck.draw(2)) # Draw treys CardInts
             player.treys_cards = list(cards)
             # Store string representation too for canonical form & display if needed
             player.hole_cards = (Card.int_to_str(cards[0]), Card.int_to_str(cards[1]))

        # Post blinds
        sb_player = self.get_player_at_position("SB")
        bb_player = self.get_player_at_position("BB")

        if sb_player:
             sb_posted = sb_player.post_blind(self.sb_amount)
             self.pot += sb_posted
             print(f"{sb_player.name} (SB) posts {sb_posted}")
        if bb_player:
             bb_posted = bb_player.post_blind(self.bb_amount)
             self.pot += bb_posted
             print(f"{bb_player.name} (BB) posts {bb_posted}")
             self.last_raiser = bb_player # BB is the initial 'raiser' pre-flop

        self.current_bet = self.bb_amount

        # Determine first player to act pre-flop (UTG)
        utg_player = self.get_player_at_position("UTG")
        self.action_idx = self.players.index(utg_player) if utg_player else (self.dealer_pos_idx + 3) % self.num_players


    def get_player_at_position(self, pos_abbr: str) -> Optional[Player]:
        for p in self.players:
            if p.position == pos_abbr:
                return p
        return None

    def get_active_players(self) -> List[Player]:
        return [p for p in self.players if p.in_hand]

    def get_active_opponents(self, hero: Player) -> List[Player]:
         return [p for p in self.players if p.in_hand and p != hero]

    def betting_round_over(self) -> bool:
        active_players = [p for p in self.get_active_players() if not p.is_all_in]
        if len(active_players) <= 1:
             # If only one active player left who isn't all-in, or all remaining are all-in
             all_in_players = [p for p in self.get_active_players() if p.is_all_in]
             if len(active_players) + len(all_in_players) <= 1: # Only 0 or 1 player left in hand total
                 return True
             # Check if all non-all-in players have acted and bet amounts match
             if not active_players: return True # All remaining are all-in

        # Normal check: has action returned to the last aggressor OR checked around?
        aggressor = self.last_raiser
        current_player = self.players[self.action_idx]

        # Condition 1: Everyone has matched the current bet or folded/all-in
        all_matched_or_done = True
        max_bet = self.current_bet
        for p in self.get_active_players():
            if not p.is_all_in and p.current_bet < max_bet and p.last_action is not None:
                 all_matched_or_done = False
                 break
            # Also ensure everyone who *can* act has had a turn since the last raise
            if not p.is_all_in and p.last_action is None and max_bet > (self.bb_amount if self.street=="Pre-flop" else 0):
                 all_matched_or_done = False # Someone hasn't acted yet facing a bet
                 break

        # Condition 2: Action is on the last aggressor and no raise occurred since their last action OR it checked around
        is_on_aggressor = (aggressor is not None and current_player == aggressor)
        checked_around = (aggressor is None and current_player == self.get_player_at_position("SB") and self.street != "Pre-flop") # Postflop checkaround check
        preflop_bb_option = (aggressor is not None and current_player == aggressor and current_player.position == "BB" and self.current_bet == self.bb_amount and self.street == "Pre-flop") # BB option check

        # Combine: Everyone matched OR (action is on aggressor who cannot act again OR it checked around postflop OR preflop BB option check)
        round_over = all_matched_or_done and (is_on_aggressor or checked_around or preflop_bb_option)


        # Simpler check: All active, non-all-in players have put in the same amount OR have folded
        active_non_all_in = [p for p in self.players if p.in_hand and not p.is_all_in]
        if not active_non_all_in: return True # Only all-in players left

        bets = {p.current_bet for p in active_non_all_in}
        acted = all(p.last_action is not None for p in active_non_all_in)

        # Special case: Big blind preflop option
        is_bb = self.players[self.action_idx].position == 'BB'
        is_preflop = self.street == 'Pre-flop'
        bet_is_bb = self.current_bet == self.bb_amount
        no_raiser_yet = self.last_raiser == self.get_player_at_position('BB') # If last raiser is still BB, means no raise occurred

        if is_preflop and is_bb and bet_is_bb and no_raiser_yet and acted:
             # If it gets back to BB preflop with no raise, they have option to check/raise
             # If they check/fold, round is over. If they raise, it continues.
             # This function checks if it *should* be over, so return False if BB still has option
             return False # BB still needs to act

        return len(bets) == 1 and acted


    def advance_action(self):
        if self.hand_over: return
        # Find the next player who is still in the hand and not all-in
        current_action_idx = self.action_idx
        while True:
            self.action_idx = (self.action_idx + 1) % self.num_players
            next_player = self.players[self.action_idx]
            if next_player.in_hand and not next_player.is_all_in:
                # Check if action has come back around to the start/last raiser
                # This requires careful checking if the round is actually over
                if self.betting_round_over():
                     self.end_betting_round()
                     break # Move to next street or showdown
                else:
                    break # Found next player to act

            # If we loop all the way around, the round/hand might be over
            if self.action_idx == current_action_idx:
                 # This can happen if only one player remains or all others are all-in
                 self.end_betting_round()
                 break


    def process_action(self, player: Player, action: str, amount: int = 0):
        player_idx = self.players.index(player)
        to_call = self.current_bet - player.current_bet
        player.last_action = action

        if action == "fold":
            player.in_hand = False
            print(f"{player.name} folds")
            # No change to pot unless walking over blinds (handled implicitly)

        elif action == "check":
             if to_call > 0:
                 print(f"{player.name} Error: Cannot check, call amount is {to_call}") # Should not happen with valid prompts
                 action = "fold" # Treat invalid check as fold
                 player.in_hand = False
                 print(f"{player.name} folds")
             else:
                 print(f"{player.name} checks")

        elif action == "call":
            call_amount = min(to_call, player.stack) # Can only call what they have
            player.stack -= call_amount
            player.current_bet += call_amount
            player.total_bet_in_pot += call_amount
            self.pot += call_amount
            print(f"{player.name} calls {call_amount}")
            if player.stack == 0:
                player.is_all_in = True
                print(f"{player.name} is all-in!")

        elif action in ["bet", "raise"]:
            # 'amount' is the total size of the bet/raise for this player this round
            # We need the amount *added* to the pot
            additional_amount = amount - player.current_bet # How much more are they putting in now?
            additional_amount = min(additional_amount, player.stack) # Cap at stack

            actual_total_bet = player.current_bet + additional_amount

            # Validate raise amount (must be at least min raise)
            min_raise_amount = self.bb_amount # Minimum raise increment
            if self.last_raiser:
                min_raise_amount = max(self.bb_amount, self.current_bet + (self.current_bet - (self.last_raiser.current_bet if self.last_raiser else 0)))

            if action == "raise" and actual_total_bet < min_raise_amount and actual_total_bet < player.stack + player.current_bet:
                # Illegal raise size (unless it's exactly all-in)
                # For simplicity in trainer, force call or fold if raise invalid? Or force min-raise?
                # Let's force call if possible, else fold
                print(f"{ANSI['yellow']}Illegal raise size by {player.name}. Min raise to {min_raise_amount}. Treating as call.{ANSI['reset']}")
                action = "call"
                call_amount = min(to_call, player.stack)
                player.stack -= call_amount
                player.current_bet += call_amount
                player.total_bet_in_pot += call_amount
                self.pot += call_amount
                print(f"{player.name} calls {call_amount}")
                if player.stack == 0: player.is_all_in = True

            else: # Bet or Valid Raise
                player.stack -= additional_amount
                player.current_bet = actual_total_bet
                player.total_bet_in_pot += additional_amount
                self.pot += additional_amount

                verb = "bets" if to_call == 0 else "raises to"
                print(f"{player.name} {verb} {actual_total_bet}")

                self.current_bet = player.current_bet # Update the current bet level
                self.last_raiser = player # This player is now the last aggressor

                if player.stack == 0:
                    player.is_all_in = True
                    print(f"{player.name} is all-in!")

        # After any action, record stats
        if self.stats:
             if self.street == "Pre-flop":
                 # Simplified: Assume voluntary if not SB/BB initial post
                 is_blind_post = (player.position in ["SB", "BB"] and player.total_bet_in_pot <= (self.sb_amount if player.position == "SB" else self.bb_amount))
                 voluntary = not is_blind_post and action != 'check' # Check by BB is not voluntary VPIP $
                 # TODO: Refine can_raise check
                 can_raise = True # Simplification: Assume raise was always possible if they didn't fold
                 self.stats.record_preflop_action(player_idx, action, voluntary, can_raise)
             else:
                 self.stats.record_postflop_action(player_idx, action)

        # Check if hand ends due to folds
        active_players = self.get_active_players()
        if len(active_players) == 1:
            self.hand_over = True
            self.award_pot(active_players[0])

    def end_betting_round(self):
        print("--- Betting round ends ---")
        # Create side pots if necessary (before adding board cards)
        self.create_side_pots()

        # Reset for next round
        self.current_bet = 0
        self.last_raiser = None
        for p in self.players:
            p.current_bet = 0 # Reset per-round bet amount
            p.last_action = None # Reset last action marker

        # Set action to first player after dealer (SB if active)
        next_actor_idx = (self.dealer_pos_idx + 1) % self.num_players
        while not self.players[next_actor_idx].in_hand or self.players[next_actor_idx].is_all_in:
             next_actor_idx = (next_actor_idx + 1) % self.num_players
             if next_actor_idx == (self.dealer_pos_idx + 1) % self.num_players: # Looped around
                 break # Should lead to next street deal or showdown
        self.action_idx = next_actor_idx

        # Check again if hand is over after creating side pots (e.g., one active player left)
        active_players = self.get_active_players()
        if len(active_players) == 1:
            self.hand_over = True
            # Pot awarding already handled by create_side_pots logic if needed
            if self.pot > 0: # Award main pot if side pots didn't cover it
                 self.award_pot(active_players[0])
            return


        # Deal next street or go to showdown
        if self.street == "Pre-flop":
            self.street = "Flop"
            self.board_cards.extend(self.deck.draw(3))
            print(f"Dealing Flop: {join(self.board_cards)}")
        elif self.street == "Flop":
            self.street = "Turn"
            self.board_cards.extend(self.deck.draw(1))
            print(f"Dealing Turn: {join(self.board_cards)}")
        elif self.street == "Turn":
            self.street = "River"
            self.board_cards.extend(self.deck.draw(1))
            print(f"Dealing River: {join(self.board_cards)}")
        elif self.street == "River":
            self.hand_over = True
            self.go_to_showdown()


    def create_side_pots(self):
        """ Resolve all-ins and create side pots. Complex logic."""
        players_in_pot = sorted([p for p in self.players if p.total_bet_in_pot > 0], key=lambda p: p.total_bet_in_pot)
        
        if not any(p.is_all_in for p in players_in_pot):
             return # No all-ins, no side pots needed

        self.side_pots = []
        last_all_in_level = 0
        
        processed_pot = 0

        all_in_players = sorted([p for p in players_in_pot if p.is_all_in], key=lambda p: p.total_bet_in_pot)
        
        for all_in_player in all_in_players:
            all_in_amount = all_in_player.total_bet_in_pot
            if all_in_amount <= last_all_in_level: continue # Already covered by a previous side pot

            side_pot_value = 0
            eligible_players = []

            for p in players_in_pot:
                contribution = min(p.total_bet_in_pot, all_in_amount) - last_all_in_level
                if contribution > 0:
                    side_pot_value += contribution
                    if p.in_hand: # Only players still in the hand are eligible
                         eligible_players.append(p)

            if side_pot_value > 0:
                 self.side_pots.append({'amount': side_pot_value, 'eligible': eligible_players})
                 processed_pot += side_pot_value
                 print(f"Side Pot {len(self.side_pots)} ({side_pot_value}) created for players: {', '.join(p.name for p in eligible_players)}")

            last_all_in_level = all_in_amount

        # Main pot (remaining amount after all side pots)
        main_pot_value = self.pot - processed_pot
        if main_pot_value < 0 :
             # This indicates an error in calculation, correct self.pot based on side pots
             print(f"Warning: Pot calculation mismatch. Adjusting pot. ({self.pot} vs {processed_pot})")
             self.pot = processed_pot
             main_pot_value = 0
             # Try to recover by assigning remainder to last pot? Or recalculate total pot?
             # Recalculate total pot from player contributions for safety
             self.pot = sum(p.total_bet_in_pot for p in self.players)
             main_pot_value = self.pot - processed_pot
             if main_pot_value < 0:
                 print("Error: Still negative main pot value after recalculation.")
                 main_pot_value = 0 # Failsafe

        if main_pot_value > 0:
            # Eligible for main pot are all players who contributed beyond the last all-in level (or all if no all-ins)
            eligible_main = [p for p in players_in_pot if p.in_hand and p.total_bet_in_pot > last_all_in_level]
            if not eligible_main: # If last all-in was the highest bettor
                 eligible_main = [p for p in players_in_pot if p.in_hand and p.total_bet_in_pot == last_all_in_level]

            print(f"Main Pot ({main_pot_value}) eligible players: {', '.join(p.name for p in eligible_main)}")
            # Add main pot info to side_pots structure for unified handling
            self.side_pots.append({'amount': main_pot_value, 'eligible': eligible_main})
            self.pot = 0 # All money is now accounted for in side_pots list

        # If after processing, only one player is eligible for the highest remaining pot, award it now
        if self.side_pots:
             last_pot = self.side_pots[-1]
             if len(last_pot['eligible']) == 1:
                 winner = last_pot['eligible'][0]
                 amount = last_pot['amount']
                 print(f"{winner.name} wins pot {len(self.side_pots)} ({amount}) uncontested.")
                 winner.stack += amount
                 if self.stats: self.stats.record_win(self.players.index(winner), amount)
                 self.side_pots.pop() # Remove awarded pot

        # Reset player total_bet_in_pot after pots are assigned (maybe?)
        # No, keep total_bet_in_pot for stat tracking? Or reset? Let's keep it for now.


    def award_pot(self, winner: Player):
        """Award the main pot when hand ends before showdown due to folds."""
        if self.pot > 0: # Should only be called if side pots haven't claimed all money
             print(f"{winner.name} wins the pot ({self.pot})")
             winner.stack += self.pot
             if self.stats: self.stats.record_win(self.players.index(winner), self.pot)
             self.pot = 0
        # Also award any remaining side pots they might be eligible for
        remaining_side_pots = []
        for pot_info in self.side_pots:
             if winner in pot_info['eligible']:
                  print(f"{winner.name} also wins side pot ({pot_info['amount']})")
                  winner.stack += pot_info['amount']
                  if self.stats: self.stats.record_win(self.players.index(winner), pot_info['amount'])
             else:
                 remaining_side_pots.append(pot_info) # Keep pots they weren't eligible for (shouldn't happen here)
        self.side_pots = remaining_side_pots


    def go_to_showdown(self):
        print("\n" + ANSI['bold'] + "*** Showdown ***" + ANSI['reset'])
        print(f"Board: {join(self.board_cards)}")

        showdown_players = [p for p in self.players if p.in_hand]
        if not showdown_players:
             print("Error: Showdown with no players in hand?")
             return
        if len(showdown_players) == 1:
             # This case should have been caught earlier, but handle defensively
             self.award_pot(showdown_players[0])
             return

        player_hands = []
        for p in showdown_players:
            if p.hole_cards:
                 hand_score, hand_name, _ = evaluate_hand(p.treys_cards, self.board_cards)
                 print(f"{p.name:<10} {join(p.hole_cards):<10} {hand_name:<15} (Score: {hand_score})")
                 player_hands.append({'player': p, 'score': hand_score, 'name': hand_name})
            else:
                 print(f"{p.name:<10} -- FOLDED --") # Should not happen if p.in_hand is true


        if not player_hands:
             print("No hands to evaluate at showdown.")
             return

        # Award side pots first, from smallest to largest
        pots_to_award = self.side_pots # Use the calculated side pots
        if not pots_to_award and self.pot > 0: # Handle case where no all-ins occurred
             pots_to_award = [{'amount': self.pot, 'eligible': showdown_players}]
             self.pot = 0


        for i, pot_info in enumerate(pots_to_award):
            pot_amount = pot_info['amount']
            eligible_players = pot_info['eligible']

            # Filter player_hands to only those eligible for *this* pot
            eligible_hands = [h for h in player_hands if h['player'] in eligible_players]

            if not eligible_hands:
                 print(f"Warning: No eligible hands found for Pot {i+1} ({pot_amount}). Skipping pot.")
                 continue # Skip this pot if no eligible hands

            # Find best score among eligible hands (lower is better in treys)
            best_score = min(h['score'] for h in eligible_hands)
            # Get the Player objects who achieved the best score
            winners = [h['player'] for h in eligible_hands if h['score'] == best_score]

            # Find the corresponding hand entry to get the descriptive hand name
            winning_hand_entry = next((h for h in eligible_hands if h['score'] == best_score), None)
            winning_hand_name = winning_hand_entry['name'] if winning_hand_entry else "Unknown Hand" # Get hand name like "Two Pair"

            pot_name = f"Side Pot {i+1}" if len(pots_to_award) > 1 and i < len(pots_to_award)-1 else "Main Pot"
            if len(pots_to_award) == 1: pot_name = "Pot"

            share = pot_amount // len(winners)
            remainder = pot_amount % len(winners) # Handle odd chips

            # --- CORRECTED PRINT STATEMENT ---
            print(f"\n{pot_name} ({pot_amount}) winners ({winners[0].name if len(winners)==1 else 'Split'}):")
            # --- END CORRECTION ---

            for winner in winners:
                 winner.stack += share
                 # --- CORRECTED HAND NAME USAGE ---
                 print(f"  {winner.name} wins {share} (Hand: {winning_hand_name})")
                 # --- END CORRECTION ---
                 if self.stats: self.stats.record_win(self.players.index(winner), share)

            # Award remainder to first winner (closest to dealer's left) - simple method
            if remainder > 0 and winners:
                 # Find winner closest to dealer's left among winners
                 min_dist = self.num_players
                 first_winner = winners[0]
                 dealer_idx = self.dealer_pos_idx
                 for w in winners:
                      w_idx = self.players.index(w)
                      dist = (w_idx - dealer_idx -1 + self.num_players) % self.num_players
                      if dist < min_dist:
                          min_dist = dist
                          first_winner = w
                 first_winner.stack += remainder
                 print(f"  {first_winner.name} gets remainder {remainder} chip(s)")
                 if self.stats: self.stats.record_win(self.players.index(first_winner), remainder)


# ────────── Equity Calculation ──────────

def calculate_equity(
    hero_cards: List[CardInt],
    board: List[CardInt],
    opponents: List[Player],
    sims: int,
    opponent_ranges: Optional[List[Range]] = None # List of ranges, one per opponent
) -> Tuple[float, float, float]:
    """Monte Carlo equity simulation against opponent ranges."""
    wins = 0
    ties = 0
    total_sims = sims

    # We don't need hero_hand_str right now, can be removed if not used later
    # hero_hand_str = [Card.int_to_str(c) for c in hero_cards]

    # Use a deck object only for simulations later, not for initial card list
    dead_cards = set(hero_cards) | set(board)

    # Prepare opponent range possibilities
    possible_opponent_hands: List[List[HoleCards]] = [] # Stores list of ('Ac', 'Ks') per opponent

    if opponent_ranges:
        if len(opponent_ranges) != len(opponents):
             raise ValueError("Mismatch between number of opponents and provided ranges.")

        # --- Generate possible combos ONCE before looping through opponents ---
        try:
            all_cards_ints = TreysDeck().cards # Get all 52 card integers
        except NameError: # Fallback if TreysDeck is somehow not defined (shouldn't happen)
             print("Error: TreysDeck not defined? Check imports.")
             # Create manually if needed (less efficient)
             all_cards_ints = [Card.new(r+s) for r in RANKS for s in SUITS]

        available_cards = [c for c in all_cards_ints if c not in dead_cards]

        # Generate all possible 2-card combos from available cards
        # Make sure itertools is imported at the top!
        all_possible_combos_int: List[Tuple[CardInt, CardInt]] = list(itertools.combinations(available_cards, 2))
        # --- End of combo generation ---

        for i, opp in enumerate(opponents):
             opp_range = opponent_ranges[i]
             if not opp_range: # If opponent folded or range is unknown, treat as random
                 possible_opponent_hands.append([]) # Sentinel for random
                 continue

             valid_hands_for_opp: List[HoleCards] = []
             # Iterate through the pre-calculated combos for this opponent's range
             for card1_int, card2_int in all_possible_combos_int:
                  # Convert combo to string form for range checking
                  card1_str, card2_str = Card.int_to_str(card1_int), Card.int_to_str(card2_int)
                  canon = canonical_form(card1_str, card2_str)
                  if canon in opp_range:
                      # Add the string tuple ('As', 'Kd') to the list for this opponent
                      valid_hands_for_opp.append((card1_str, card2_str))

             if not valid_hands_for_opp:
                 # Range is impossible with dead cards
                 # print(f"Warning: Opponent {opp.name}'s range {opp_range} has no combos left. Simulating random.")
                 possible_opponent_hands.append([]) # Sentinel for random
             else:
                 possible_opponent_hands.append(valid_hands_for_opp)

    else: # No ranges provided, simulate against random hands
        possible_opponent_hands = [[] for _ in opponents] # Sentinel for random for all


    # --- Simulation Loop ---
    sims_run = 0
    for i in range(total_sims):
        # Use a fresh deck instance for card availability checks INSIDE the simulation loop
        current_sim_deck_cards = list(TreysDeck().cards)
        current_dead_sim = set(dead_cards) # Start with hero + board dead cards

        # 1. Sample opponent hands
        opp_hands_sim: List[List[CardInt]] = []
        possible_this_sim = True
        
        # Keep track of cards available in this specific simulation run
        available_cards_sim = [c for c in current_sim_deck_cards if c not in current_dead_sim]
        random.shuffle(available_cards_sim) # Shuffle for random drawing


        for j, opp in enumerate(opponents):
             possible_hands_for_opp = possible_opponent_hands[j] # Get the list of ('As','Kd') etc.

             drawn_hand: List[CardInt] = []
             if not possible_hands_for_opp: # Sentinel for random or impossible range
                 # Draw 2 random cards from available_cards_sim
                 if len(available_cards_sim) < 2:
                     possible_this_sim = False; break
                 # Use pop() on the shuffled list for efficiency
                 card1 = available_cards_sim.pop()
                 card2 = available_cards_sim.pop()
                 drawn_hand = [card1, card2]
             else:
                 # Sample a hand from their valid range combos ('As', 'Kd')
                 attempts = 0
                 found_hand = False
                 range_copy = list(possible_hands_for_opp) # Copy to sample without replacement? No, sample with replacement is fine.
                 random.shuffle(range_copy) # Shuffle potential hands

                 for candidate_hand_str in range_copy: # Try shuffled candidates first
                     candidate_ints = hole_cards_to_treys(candidate_hand_str)

                     # Check if these specific card integers are still available in this sim run
                     card1_available = candidate_ints[0] in available_cards_sim
                     card2_available = candidate_ints[1] in available_cards_sim

                     if card1_available and card2_available:
                         drawn_hand = candidate_ints
                         # Remove chosen cards from available list for this sim
                         available_cards_sim.remove(candidate_ints[0])
                         available_cards_sim.remove(candidate_ints[1])
                         found_hand = True
                         break # Found a hand for this opponent

                 if not found_hand: # Failed to find a hand from range after trying all shuffled combos
                      possible_this_sim = False; break

             if not possible_this_sim: break # Break outer opponent loop if sim fails

             opp_hands_sim.append(drawn_hand)
             # No need to update current_dead_sim here, availability check handles it

        if not possible_this_sim:
             # print("Sim failed: Not enough cards for opponents or range conflict.")
             continue # Skip to next simulation attempt

        # 2. Deal remaining board cards
        board_runout = list(board)
        cards_to_deal = 5 - len(board)
        if len(available_cards_sim) < cards_to_deal:
            # print("Sim failed: Not enough cards for board.")
            continue # Skip to next simulation attempt

        # Draw board cards from the remaining available cards for this sim
        drawn_board = random.sample(available_cards_sim, cards_to_deal)
        board_runout.extend(drawn_board)

        # 3. Evaluate hands
        try:
            hero_score, _, _ = evaluate_hand(hero_cards, board_runout)
            opponent_scores = [evaluate_hand(opp_hand, board_runout)[0] for opp_hand in opp_hands_sim]
        except Exception as e:
            # Catch potential errors during evaluation (e.g., invalid card data)
            print(f"Error during hand evaluation in sim: {e}")
            print(f"Hero: {hero_cards}, Opp: {opp_hands_sim}, Board: {board_runout}")
            continue # Skip this sim

        best_opp_score = min(opponent_scores) if opponent_scores else 9999 # Lower score is better

        if hero_score < best_opp_score:
            wins += 1
        elif hero_score == best_opp_score:
            ties += 1 # Simple tie counting sufficient for W/T/L percentage

        sims_run += 1 # Count successfully completed simulations

    # --- End Simulation Loop ---

    if sims_run == 0: return 0.0, 0.0, 100.0 # Avoid division by zero

    win_pct = (wins / sims_run) * 100
    tie_pct = (ties / sims_run) * 100
    lose_pct = 100 - win_pct - tie_pct

    return win_pct, tie_pct, lose_pct

# ────────── User Input & Actions ──────────

def get_hero_action(game_state: GameState, hero: Player) -> Tuple[str, int]:
    """Prompt hero for action and return validated action and amount."""
    pot = game_state.pot + sum(p.total_bet_in_pot for p in game_state.players) # Total pot includes current bets
    to_call = game_state.current_bet - hero.current_bet
    can_check = to_call == 0
    can_fold = True
    can_call = to_call > 0 and hero.stack >= to_call
    can_raise = hero.stack > to_call # Must have chips behind after calling to raise

    actions = []
    if can_fold: actions.append("fold (f)")
    if can_check: actions.append("check (x)")
    if can_call:
        pot_odds = to_call / (pot + to_call) * 100 if (pot + to_call) > 0 else 0
        actions.append(f"call {to_call} (c) [Odds:{pot_odds: >3.0f}%]")
    
    # Betting/Raising options
    min_raise_abs = game_state.current_bet + max(game_state.bb_amount, game_state.current_bet - (game_state.last_raiser.current_bet if game_state.last_raiser else 0))
    min_raise_abs = max(min_raise_abs, game_state.bb_amount * 2) # Min open is 2x usually


    if can_raise:
        # Calculate raise sizes relative to the pot *after* hero calls
        pot_after_call = pot + to_call
        
        bets = {}
        half_pot_raise = int(pot_after_call * 0.5)
        three_qrt_pot_raise = int(pot_after_call * 0.75)
        pot_raise = int(pot_after_call * 1.0)
        
        # Amounts here are the amount *added* on top of the call
        bets['½ pot'] = (half_pot_raise, to_call + half_pot_raise)
        bets['¾ pot'] = (three_qrt_pot_raise, to_call + three_qrt_pot_raise)
        bets['pot'] = (pot_raise, to_call + pot_raise)
        
        # Check legality and affordability
        valid_raises = {}
        for name, (raise_amount, total_bet) in bets.items():
            if total_bet >= min_raise_abs and total_bet <= hero.stack + hero.current_bet:
                 valid_raises[name] = total_bet

        # Add min-raise option if different from others
        if min_raise_abs <= hero.stack + hero.current_bet and min_raise_abs not in valid_raises.values():
            valid_raises['min'] = min_raise_abs

        # Add All-in option
        all_in_total_bet = hero.stack + hero.current_bet
        if all_in_total_bet > game_state.current_bet: # Only show if it's actually a raise/bet
             valid_raises['all-in'] = all_in_total_bet

        # Generate raise prompts, ordered by size
        sorted_raises = sorted(valid_raises.items(), key=lambda item: item[1])
        
        for i, (name, total_bet) in enumerate(sorted_raises):
             # Calculate fold equity estimate (simple version)
             # Average fold % based on opponent profiles remaining
             active_opps = game_state.get_active_opponents(hero)
             fold_equity_est = 0
             if active_opps:
                 total_fold_prob = 0
                 for opp in active_opps:
                      tendencies = POSTFLOP_TENDENCIES.get(opp.profile, POSTFLOP_TENDENCIES['tight']) # Default if profile unknown
                      bet_ratio = (total_bet - to_call) / pot_after_call if pot_after_call > 0 else 1
                      # Simple interpolation between half-pot and pot fold%
                      f_half = tendencies['fold_vs_raise_half_pot']
                      f_pot = tendencies['fold_vs_raise_pot']
                      if bet_ratio <= 0.5: fold_prob = f_half * (bet_ratio / 0.5) # Linear below half pot
                      elif bet_ratio <= 1.0: fold_prob = f_half + (f_pot - f_half) * ((bet_ratio - 0.5) / 0.5) # Linear between half and full
                      else: fold_prob = f_pot + (1.0 - f_pot) * (1 - (1/ (1+bet_ratio-1))) # Diminishing returns above pot

                      total_fold_prob += max(0, min(1, fold_prob)) # Clamp probability
                 fold_equity_est = (total_fold_prob / len(active_opps)) * 100 if active_opps else 0

             actions.append(f"raise {total_bet} ({name}, r{i+1}) [FE: {fold_equity_est:.0f}%]")

    # Display action options
    print("Your options:")
    print("  " + " | ".join(actions))

    while True:
        try:
            raw_input = input(f"{ANSI['cyan']}> Your move: {ANSI['reset']}").strip().lower()
            if not raw_input: continue

            parts = raw_input.split()
            command = parts[0]

            if (command.startswith('f') or command == 'fold') and can_fold:
                return "fold", 0
            if (command.startswith('x') or command == 'check') and can_check: return "check", 0
            if (command.startswith('c') or command == 'call') and can_call:
                return "call", to_call

            if (command.startswith('r') or command == 'raise') and can_raise:
                 # Try parsing 'r1', 'r2', etc.
                 if len(command) > 1 and command[1:].isdigit():
                     raise_idx = int(command[1:]) - 1
                     if 0 <= raise_idx < len(sorted_raises):
                         _, amount = sorted_raises[raise_idx]
                         return "raise", amount
                 # Try parsing raise amount directly: 'raise 150'
                 elif len(parts) > 1 and parts[1].isdigit():
                      amount = int(parts[1])
                      # Check if this amount is one of the suggested raises or all-in
                      if amount in [r[1] for r in sorted_raises]:
                           return "raise", amount
                      # Allow custom raise amount if valid
                      elif amount >= min_raise_abs and amount <= hero.stack + hero.current_bet:
                           print(f"{ANSI['yellow']}Custom raise amount accepted.{ANSI['reset']}")
                           return "raise", amount
                      else:
                          print(f"Invalid raise amount. Min raise: {min_raise_abs}, Max: {hero.stack + hero.current_bet}")

                 else: # Try parsing shorthand like 'pot', 'half', 'allin'
                     matched = False
                     for i, (name, total_bet) in enumerate(sorted_raises):
                          name_parts = name.split() # Handle '½ pot' etc.
                          if command == name or command == name_parts[0] or command == f"r{i+1}":
                              return "raise", total_bet
                     if not matched: print("Invalid raise command.")


            print("Invalid input. Try again.")

        except (EOFError, KeyboardInterrupt):
            print("\nAction interrupted. Exiting.")
            sys.exit(0)
        except Exception as e:
            print(f"Error processing input: {e}")


# ────────── Main Game Loop ──────────

def print_game_state(game_state: GameState, hero: Player, show_equity: bool = True, sims: int = 1000):
    print("\n" + "=" * 70)
    # Header: Positions and Stacks
    header = []
    for i, p in enumerate(game_state.players):
        is_hero = p == hero
        is_dealer = i == game_state.dealer_pos_idx
        name_str = f"{ANSI['bold']}{p.name}{ANSI['reset']}" if is_hero else p.name
        pos_str = f"({p.position})"
        dealer_str = f"{ANSI['yellow']}D{ANSI['reset']}" if is_dealer else ""
        stack_str = f"{p.stack}"
        if p.is_all_in: stack_str += f" {ANSI['red']}[ALL-IN]{ANSI['reset']}"
        elif not p.in_hand: stack_str = f"{ANSI['grey']}[FOLDED]{ANSI['reset']}"

        header.append(f"{name_str}{pos_str}{dealer_str}: {stack_str}")
    print(" | ".join(header))
    print("-" * 70)

    # Pot and Board
    pot_total = game_state.pot + sum(p.total_bet_in_pot for p in game_state.players)
    print(f"Pot: {ANSI['yellow']}{pot_total}{ANSI['reset']}  | Board: {join(game_state.board_cards) if game_state.board_cards else '--'}")

    # Hero Hand and Equity
    if hero.hole_cards:
         print(f"Your Hand: {join(hero.hole_cards)}")
         if show_equity and hero.in_hand and game_state.street != "Showdown":
              active_opponents = game_state.get_active_opponents(hero)
              if active_opponents:
                  # Use estimated opponent ranges for equity calc
                  opp_ranges = []
                  for opp in active_opponents:
                      # Simple range estimation: Use preflop range if known, else assume wide range?
                      # TODO: Refine range estimation based on postflop actions
                      if opp.preflop_range:
                           opp_ranges.append(opp.preflop_range)
                      else:
                           # If no preflop range known (e.g., they folded pre but we calc equity anyway?)
                           # Or if we need a fallback postflop range
                           # Use a default wide range based on profile?
                           fallback_range = EXPANDED_RANGES[opp.profile]['BU_open'] # Wide default
                           opp_ranges.append(fallback_range)

                  start_time = time.time()
                  win, tie, lose = calculate_equity(hero.treys_cards, game_state.board_cards, active_opponents, sims, opp_ranges)
                  elapsed = time.time() - start_time
                  print(f"Equity vs Range ({len(active_opponents)} opp): {ANSI['♣']}{win: >4.1f}%{ANSI['reset']} Win / {ANSI['yellow']}{tie: >4.1f}%{ANSI['reset']} Tie / {ANSI['♥']}{lose: >4.1f}%{ANSI['reset']} Lose ({sims} sims, {elapsed:.2f}s)")
              else:
                   print("No active opponents remaining.")


    print("-" * 70)


def play_hand(game_state: GameState, hero: Player, args: argparse.Namespace):
    """Plays a single hand of poker."""
    game_state.setup_new_hand()
    print(f"\n--- Starting Hand #{game_state.stats.hands_played + 1} ---")
    print(f"Dealer is {game_state.players[game_state.dealer_pos_idx].name}")

    # Pre-flop betting round
    current_street = "Pre-flop"
    while not game_state.hand_over:
        if game_state.street != current_street:
             print(f"\n--- {game_state.street} ---")
             current_street = game_state.street
             # Reset last actions for the new street (already done in end_betting_round)

        active_players = game_state.get_active_players()
        players_can_act = [p for p in active_players if not p.is_all_in]

        # If only one player left who can act, or everyone is all-in, end round/hand
        if len(players_can_act) <= 1 and len(active_players) > 1:
            # Check if bets are equalized among those not all-in
            if len(players_can_act) == 1:
                 if players_can_act[0].current_bet >= game_state.current_bet:
                     game_state.end_betting_round()
                     continue
            elif not players_can_act: # All remaining are all-in
                game_state.end_betting_round()
                continue # Will proceed to next street or showdown


        current_player = game_state.players[game_state.action_idx]

        if not current_player.in_hand or current_player.is_all_in:
             game_state.advance_action()
             continue

        # Display state before player acts
        print_game_state(game_state, hero, sims=args.sims)
        print(f"Action on {ANSI['bold']}{current_player.name}{ANSI['reset']} ({current_player.position}) Stack: {current_player.stack} To Call: {max(0, game_state.current_bet - current_player.current_bet)}")

        action, amount = "", 0
        if current_player.is_hero:
            action, amount = get_hero_action(game_state, hero)
        else:
            # AI makes decision
            action, amount = current_player.make_decision(game_state, game_state.stats)
            time.sleep(0.5) # Small delay for realism

        game_state.process_action(current_player, action, amount)

        if not game_state.hand_over:
             # Check if betting round is over AFTER the action
             if game_state.betting_round_over():
                  game_state.end_betting_round()
             else:
                  game_state.advance_action()


    # Hand conclusion (already printed by award_pot or go_to_showdown)
    print("-" * 70)
    # Record hand played for stats
    if game_state.stats: game_state.stats.hands_played += 1
    # Record final winnings/losses for the hand for each player
    if game_state.stats:
         for i, p in enumerate(game_state.players):
              net_change = p.stack - p.initial_stack # How much stack changed this hand
              # This is slightly flawed if rebuys happened mid-hand, but good estimate
              # A better way: Track pot contributions vs winnings per hand.
              # Let's stick to simpler net change for now. Revisit if needed.
              # Correction: record_win already tracks winnings. Need to track *loss* (total bet)
              loss = p.total_bet_in_pot
              # Net is tracked via record_win, but we need to subtract losses too.
              # Let's reset winnings tracking per hand and use stack change.
              pass # Stats.winnings are cumulative totals, updated in award_pot

    # Update initial stacks for next hand's comparison (or handle rebuys better)
    # for p in game_state.players:
    #      p.initial_stack = p.stack # Reset baseline for next hand calculation? No, keep initial session bankroll.

def main():
    parser = argparse.ArgumentParser(description="Realistic 4-Hand Texas Hold'em CLI Trainer")
    parser.add_argument('--hands', type=int, default=None, help="Number of hands to play automatically.")
    parser.add_argument('--bankroll', type=int, default=2000, help="Starting bankroll for each player (in chips).")
    parser.add_argument('--bb', type=int, default=20, help="Big blind amount.")
    parser.add_argument('--seed', type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument('--villain', type=str, default='tight', choices=['tight', 'loose', 'maniac'], help="Opponent AI profile.")
    parser.add_argument('--profile', type=str, default='TAG', choices=['TAG', 'LAG', 'Nit'], help="Hero profile (unused for interactive, for future auto-hero mode).")
    parser.add_argument('--sims', type=int, default=1000, help="Number of Monte Carlo simulations for equity calculation.")
    parser.add_argument('--auto', action='store_true', help="Run in full auto mode (Hero plays like --villain profile). Not implemented yet.")

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if args.auto:
        print("Error: --auto mode not fully implemented yet.")
        # Need to implement hero AI decision based on args.profile
        # For now, just exit or run interactively.
        # sys.exit(1)
        interactive = False # Force non-interactive if flag present
    else:
        # Check if stdin is connected to a terminal
        interactive = sys.stdin.isatty()
        if not interactive:
            print("Running in non-interactive mode (e.g., piped input). Use --auto or run in a terminal for interaction.")
            # Decide behaviour: exit or run auto? Let's default to limited hands non-interactively
            if args.hands is None: args.hands = 10 # Play a few hands automatically if piped


    # Setup Players
    hero = Player("Hero", args.bankroll, profile=args.profile, is_hero=True)
    opponents = [Player(f"Villain {i+1}", args.bankroll, profile=args.villain) for i in range(NUM_PLAYERS - 1)]
    players = [hero] + opponents
    player_names = [p.name for p in players]


    # Setup Game State and Stats
    game = GameState(players, args.bb, args.seed)
    stats = StatsTracker()
    game.stats = stats # Link stats tracker to game state

    hands_played = 0
    try:
        while True:
            if args.hands is not None and hands_played >= args.hands:
                print(f"\nFinished playing {args.hands} hands.")
                break

            play_hand(game, hero, args)
            hands_played += 1

            # Check for broke players (already handled by rebuy in Player.reset_for_new_hand)

            if interactive:
                 try:
                     again = input("\nPlay another hand? (Y/n): ").strip().lower()
                     if again == 'n':
                         break
                 except (EOFError, KeyboardInterrupt):
                    break # Exit loop gracefully
            elif args.hands is None: # Non-interactive without specific hand count = play one hand
                 break


    except (KeyboardInterrupt, EOFError):
        print("\n👋 Exiting trainer...")
    finally:
        # Print session stats upon exit
        stats.report(player_names)


if __name__ == '__main__':
    main()
