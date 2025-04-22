# poker_simulator.py
"""
CLI Texas Hold’em with Stacks & ANSI‑colour Suits
=================================================
* **Colourful output** – suits are printed red (♥,♦) or blue/green (♠,♣).
* **Player stacks** – you and 3 opponents start with 1 000 chips.
  * Small blind = 10   Big blind = 20 (rotating dealer button ignored for simplicity).
  * Bets are simplified: `call` = match current bet, `raise` = +20 chips.
  * Pot and stacks update every action.
* Falls back to auto mode if `stdin` is missing.

Requirements: Python ≥ 3.8, no external libs.
"""

from __future__ import annotations

import argparse
import itertools
import random
import sys
from collections import Counter
from typing import Dict, List, Tuple

# ────────── Card helpers ──────────
RANKS = "23456789TJQKA"
SUITS = "cdhs"  # clubs diamonds hearts spades
SUIT_SYM = {"c": "♣", "d": "♦", "h": "♥", "s": "♠"}
ANSI = {
    "♠": "\033[34m",  # blue
    "♣": "\033[32m",  # green
    "♥": "\033[31m",  # red
    "♦": "\033[31m",  # red
    "reset": "\033[0m",
}
RANK_VAL: Dict[str, int] = {r: i for i, r in enumerate(RANKS, start=2)}


def deck() -> List[str]:
    return [r + s for r in RANKS for s in SUITS]


def pc(card: str) -> str:
    sym = SUIT_SYM[card[1]]
    return f"{ANSI[sym]}{card[0]}{sym}{ANSI['reset']}"


def join(cards: List[str]) -> str:
    return " ".join(pc(c) for c in cards)

# ────────── Hand evaluation (same as before) ──────────
CATS = [
    "High Card","One Pair","Two Pair","Trips","Straight","Flush",
    "Full House","Quads","Straight Flush",
]
CAT_SCORE = {n:i for i,n in enumerate(CATS)}


def _is_straight(vals: List[int]) -> Tuple[bool,int]:
    vals=sorted(set(vals),reverse=True)
    if {14,5,4,3,2}.issubset(vals):
        return True,5
    for i in range(len(vals)-4):
        if vals[i]-vals[i+4]==4:
            return True, vals[i]
    return False,0


def _rank5(cards: Tuple[str,...]) -> Tuple[int,List[int]]:
    vals=sorted((RANK_VAL[c[0]] for c in cards),reverse=True)
    suits=[c[1] for c in cards]
    cnt=Counter(vals).most_common()
    flush=len(set(suits))==1
    straight,top=_is_straight(vals)
    if flush and straight:
        return CAT_SCORE["Straight Flush"],[top]
    if cnt[0][1]==4:
        quad=cnt[0][0]; kicker=max(v for v in vals if v!=quad)
        return CAT_SCORE["Quads"],[quad,kicker]
    if cnt[0][1]==3 and cnt[1][1]==2:
        return CAT_SCORE["Full House"],[cnt[0][0],cnt[1][0]]
    if flush:
        return CAT_SCORE["Flush"],vals
    if straight:
        return CAT_SCORE["Straight"],[top]
    if cnt[0][1]==3:
        trips=cnt[0][0]; kick=[v for v in vals if v!=trips][:2]
        return CAT_SCORE["Trips"],[trips]+kick
    if cnt[0][1]==2 and cnt[1][1]==2:
        high,low=max(cnt[0][0],cnt[1][0]),min(cnt[0][0],cnt[1][0])
        kicker=max(v for v in vals if v not in (high,low))
        return CAT_SCORE["Two Pair"],[high,low,kicker]
    if cnt[0][1]==2:
        pair=cnt[0][0]; kick=[v for v in vals if v!=pair][:3]
        return CAT_SCORE["One Pair"],[pair]+kick
    return CAT_SCORE["High Card"],vals


def best7(cards7: List[str]) -> Tuple[int,List[int]]:
    return max(_rank5(c) for c in itertools.combinations(cards7,5))

# ────────── Equity (unchanged) ──────────

def equity(player: List[str], board: List[str], opp_n:int, sims:int) -> Tuple[float,float,float]:
    wins=ties=0
    mast=deck()
    for c in player+board:
        mast.remove(c)
    need=(5-len(board))+opp_n*2
    if need>len(mast):
        return 0,0,100
    for _ in range(sims):
        d=mast.copy(); random.shuffle(d)
        comm=board+[d.pop() for _ in range(5-len(board))]
        opps=[[d.pop(),d.pop()] for _ in range(opp_n)]
        pr=best7(player+comm)
        opp_rs=[best7(o+comm) for o in opps]
        best=max([pr]+opp_rs)
        if pr==best:
            if opp_rs.count(best)==0:wins+=1
            else:ties+=1
    return wins*100/sims, ties*100/sims, (sims-wins-ties)*100/sims

# ────────── Gameplay helpers ──────────

def suggest(win:float)->str:
    if win>=55: return "RAISE"
    if win>=35: return "CALL/CHECK"
    return "FOLD"

BET_UNIT=20
SMALL_BLIND=10
BIG_BLIND=20
START_STACK=1000


def auto_decision(win:float, street:int, to_call:int)->str:
    if to_call>0:
        return "fold" if win<30 else "call"
    else:
        return "check" if win<50 else "raise"

# ────────── Hand loop with stacks ──────────

def play_hand(interactive:bool, sims:int, stacks:List[int]) -> None:
    pot=0
    d=deck(); random.shuffle(d)
    player=[d.pop(),d.pop()]
    opps=[[d.pop(),d.pop()] for _ in range(3)]

    # post blinds (Opp0=SB, Opp1=BB)
    stacks[1]-=SMALL_BLIND; pot+=SMALL_BLIND
    stacks[2]-=BIG_BLIND;   pot+=BIG_BLIND
    current_bet=BIG_BLIND

    board:List[str]=[]
    stage={0:"Pre‑flop",3:"Flop",4:"Turn",5:"River"}
    active=[True,True,True,True]  # you + 3 opps (we only allow you to act)

    while True:
        win,tie,lose=equity(player,board,3,sims)
        print("\n"+"="*60)
        print(f"*** {stage[len(board)]}  |  Pot: {pot}  |  Your stack: {stacks[0]}")
        print("Your hand :",join(player))
        print("Board     :",join(board) if board else "--")
        print(f"Win/Tie/Lose: {win:5.1f}% | {tie:4.1f}% | {lose:5.1f}%  | Suggested: {suggest(win)}")

        to_call=current_bet
        move=""
        if interactive:
            try:
                prompt=f"Your move (fold/call{'/check' if to_call==0 else ''}/raise): "
                move=input(prompt).strip().lower()
            except (EOFError,OSError):
                interactive=False
        if not interactive:
            move=auto_decision(win,len(board),to_call)
            print(f"[AUTO] {move.upper()}")
        if move=="fold":
            active[0]=False
            print("🚪 You folded – hand over.")
            return
        if move in ("call","check"):
            call_amt=to_call if to_call>0 else 0
            stacks[0]-=call_amt; pot+=call_amt
        elif move=="raise":
            raise_amt=to_call+BET_UNIT
            stacks[0]-=raise_amt; pot+=raise_amt; current_bet+=BET_UNIT
        # (opponents auto‑call if still active & can afford)
        for i in range(1,4):
            if active[i]:
                diff=current_bet-(SMALL_BLIND if i==1 else BIG_BLIND if i==2 and len(board)==0 else 0)
                diff=max(diff,0)
                diff=min(diff,stacks[i])
                stacks[i]-=diff; pot+=diff
        # deal next street
        if len(board)==0:
            board.extend([d.pop() for _ in range(3)])
            current_bet=0
        elif len(board)==3:
            board.append(d.pop()); current_bet=0
        elif len(board)==4:
            board.append(d.pop()); current_bet=0
        else:
            break

    # showdown
    pr=best7(player+board)
    opp_rs=[best7(o+board) for o in opps]
    everyone=[("You",pr,0)]+[(f"Opp {i+1}",r,i+1) for i,r in enumerate(opp_rs)]
    best=max(r for _,r,_ in everyone)
    winners=[idx for name,r,idx in everyone if r==best]
    share=pot//len(winners)
    for idx in winners:
        stacks[idx]+=share
    print("\n*** Showdown ***")
    print("Board    :",join(board))
    print("You      :",join(player))
    for i,o in enumerate(opps,1):
        print(f"Opp {i}  :",join(o))
    names=["You"]+ [f"Opp {i}" for i in range(1,4)]
    print("\nResult   :",", ".join(names[i] for i in winners),"win" if len(winners)==1 else "split pot")
    print("Pot paid :",share,"each  |  Your new stack:",stacks[0])

# ────────── Main loop ──────────

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('-a','--auto',action='store_true'); ap.add_argument('-n','--hands',type=int); ap.add_argument('--sims',type=int,default=1000)
    args=ap.parse_args()
    interactive=not args.auto
    hands_left=args.hands if args.hands else None
    stacks=[START_STACK]*4
    try:
        while True:
            play_hand(interactive,args.sims,stacks)
            if hands_left is not None:
                hands_left-=1
                if hands_left<=0:
                    break
            if interactive:
                again=input("\nPlay another hand? (y/n): ").strip().lower()
                if again!='y': break
            if any(s<=0 for s in stacks):
                print("One player is broke – resetting stacks."); stacks=[START_STACK]*4
    except KeyboardInterrupt:
        print("\n👋 Bye!")
        sys.exit()

if __name__=='__main__':
    main()
