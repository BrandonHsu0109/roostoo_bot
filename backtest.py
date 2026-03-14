# backtest.py
#
# Purpose:
#   Offline backtest for Roostoo ticker snapshots (logs/ticker_history.csv).
#   - Buys at ask, sells at bid, pays fee_rate on notional.
#   - Long-only cross-sectional momentum with:
#       * spread + liquidity filters
#       * momentum threshold
#       * breadth (market regime) gate
#   - Contest-like execution constraints:
#       * rebalance only every N minutes
#       * max_orders_per_rebalance (default 1)
#       * min_trade_usd
#       * min_delta_usd applies to BUY only (so exits are not blocked)
#
# Input CSV columns expected:
#   ts, pair, bid, ask, last, change_24h, coin_value, unit_value
#
# Usage:
#   python backtest.py logs\ticker_history.csv

from __future__ import annotations

import sys
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(dd.min()) if len(dd) else 0.0


def backtest(
    csv_path: str,
    *,
    initial_usd: float = 50000.0,

    # Signal windows (minutes)
    horizon_minutes: int = 30,
    vol_window_minutes: int = 60,

    # Portfolio
    top_n: int = 5,
    cash_buffer: float = 0.40,
    max_weight_per_coin: float = 0.25,

    # Filters
    min_unit_value: float = 0.0,        # UnitTradeValue filter
    max_spread_pct: float = 0.0031,     # (ask-bid)/mid filter

    # Entry / regime
    mom_threshold: float = 0.005,       # require mom > threshold to be eligible
    breadth_threshold: float = 0.55,    # require fraction of trending coins >= threshold

    # Execution constraints
    rebalance_every: int = 5,           # only trade every N minutes
    max_orders_per_rebalance: int = 1,  # contest-like (1 order per decision)
    min_trade_usd: float = 50.0,        # ignore tiny trades
    min_delta_usd: float = 200.0,       # applies to BUY only (exits not blocked)
    fee_rate: float = 0.00012,          # fee on notional (approx)

    eps: float = 1e-12,
    verbose: bool = True,
) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)

    # Minute-bucket timestamps to remove jitter
    df["tmin"] = (df["ts"].astype(np.int64) // 60000) * 60000

    # For each (minute, pair), keep last observation
    df = (
        df.sort_values(["tmin", "pair", "ts"])
          .groupby(["tmin", "pair"], as_index=False)
          .last()
    )

    # Ensure numeric
    df["bid"] = df["bid"].astype(float)
    df["ask"] = df["ask"].astype(float)
    df["unit_value"] = df["unit_value"].astype(float)

    # Derived fields
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df["spread_pct"] = np.where(df["mid"] > 0, (df["ask"] - df["bid"]) / df["mid"], np.nan)

    # Pivot to time x pair matrices
    mid = df.pivot(index="tmin", columns="pair", values="mid").sort_index()
    bid = df.pivot(index="tmin", columns="pair", values="bid").reindex(mid.index)
    ask = df.pivot(index="tmin", columns="pair", values="ask").reindex(mid.index)
    unitv = df.pivot(index="tmin", columns="pair", values="unit_value").reindex(mid.index)
    spreadp = df.pivot(index="tmin", columns="pair", values="spread_pct").reindex(mid.index)

    pairs = mid.columns.to_list()
    times = mid.index.to_numpy()

    # Returns matrix for volatility
    rets = mid.pct_change(fill_method=None)

    # Portfolio state
    cash = float(initial_usd)
    pos = {p: 0.0 for p in pairs}  # quantities in coins
    equity_curve = []
    trades = 0

    h = int(horizon_minutes)
    k = int(vol_window_minutes)
    need = max(h + 1, k + 1)

    for i in range(len(times)):
        # Mark-to-market PV using bid (conservative liquidation)
        pv = cash
        for p in pairs:
            q = pos[p]
            if q != 0.0:
                b = bid.iloc[i][p]
                if pd.notna(b) and b > 0:
                    pv += q * float(b)

        # Always append PV (so equity_curve length == minutes)
        # Warm-up period: no trading until enough history exists
        if i < need:
            equity_curve.append(pv)
            continue

        # Only trade on rebalance minutes
        if (i % int(rebalance_every)) != 0:
            equity_curve.append(pv)
            continue

        # Compute momentum and volatility vectors at time i
        mom = (mid.iloc[i] / mid.iloc[i - h]) - 1.0
        vol_window = rets.iloc[i - k + 1: i + 1]
        vol = vol_window.std(ddof=1)

        # Base tradability filters at time i
        base_ok = (
            (unitv.iloc[i] >= float(min_unit_value)) &
            (spreadp.iloc[i] <= float(max_spread_pct)) &
            (mid.iloc[i].notna()) &
            (bid.iloc[i].notna()) &
            (ask.iloc[i].notna())
        )

        base_ok_count = int(base_ok.sum())
        if base_ok_count == 0:
            # No base-ok markets -> target all cash (sell-down if holding)
            target = {p: 0.0 for p in pairs}
        else:
            trending = ((mom > float(mom_threshold)) & base_ok)
            breadth = float(trending.sum()) / float(base_ok_count)

            # IMPORTANT: target defaults to ALL CASH.
            # If breadth passes, we fill targets for selected coins; otherwise we keep 0s -> exit to cash.
            target = {p: 0.0 for p in pairs}

            if breadth >= float(breadth_threshold):
                # Candidate scoring: require mom above threshold and finite vol
                score = mom / (vol + eps)
                ok = base_ok & (mom > float(mom_threshold)) & score.notna() & np.isfinite(score)

                score_ok = score.where(ok).dropna()
                if len(score_ok) > 0:
                    top = score_ok.sort_values(ascending=False).head(int(top_n))
                    selected_pairs = top.index.to_list()

                    investable = float(pv) * (1.0 - float(cash_buffer))
                    scores = top.to_numpy(dtype=float)
                    ssum = float(np.sum(scores))

                    if ssum > 0 and investable > 0:
                        raw_w = scores / ssum
                        capped_w = np.minimum(raw_w, float(max_weight_per_coin))
                        capped_sum = float(np.sum(capped_w))
                        w = (capped_w / capped_sum) if capped_sum > 0 else raw_w

                        for p, wi in zip(selected_pairs, w):
                            m = mid.iloc[i][p]
                            if pd.notna(m) and float(m) > 0:
                                notional = investable * float(wi)
                                target[p] = float(notional) / float(m)

        # ---- Execute trades (contest-like: at most K orders) ----
        trade_candidates: list[Tuple[float, str, str, float, float]] = []
        # tuple: (abs_notional, pair, side, qty, price)

        for p in pairs:
            cur_q = float(pos[p])
            tgt_q = float(target[p])
            delta = tgt_q - cur_q
            if abs(delta) < 1e-12:
                continue

            if delta > 0:
                side = "BUY"
                px = ask.iloc[i][p]
                qty = float(delta)
            else:
                side = "SELL"
                px = bid.iloc[i][p]
                qty = float(-delta)

            if pd.isna(px) or float(px) <= 0 or qty <= 0:
                continue

            notional = float(px) * qty

            # Always enforce a minimum meaningful trade size
            if notional < float(min_trade_usd):
                continue

            # IMPORTANT: min_delta_usd should not block exits.
            if side == "BUY" and notional < float(min_delta_usd):
                continue

            trade_candidates.append((notional, p, side, qty, float(px)))

        # Execute only the largest K by notional
        trade_candidates.sort(key=lambda x: x[0], reverse=True)
        for notional, p, side, qty, px in trade_candidates[: int(max_orders_per_rebalance)]:
            fee = notional * float(fee_rate)

            if side == "BUY":
                cost = notional + fee
                if cash >= cost:
                    cash -= cost
                    pos[p] += qty
                    trades += 1
            else:  # SELL
                # Sell only what we have
                if pos[p] >= qty:
                    pos[p] -= qty
                    cash += (notional - fee)
                    trades += 1

        # End-of-minute equity after trades (still valued at bid)
        pv2 = cash
        for p in pairs:
            q = pos[p]
            if q != 0.0:
                b = bid.iloc[i][p]
                if pd.notna(b) and b > 0:
                    pv2 += q * float(b)

        equity_curve.append(float(pv2))

    equity = np.asarray(equity_curve, dtype=float)
    total_return = (equity[-1] / equity[0]) - 1.0 if len(equity) else 0.0
    mdd = max_drawdown(equity)

    # Per-minute returns for Sharpe (not annualized)
    if len(equity) >= 2:
        r = np.diff(equity) / equity[:-1]
        sharpe = float(np.mean(r) / (np.std(r, ddof=1) + eps)) if len(r) >= 2 else 0.0
    else:
        sharpe = 0.0

    out = {
        "final_equity": float(equity[-1]) if len(equity) else float(initial_usd),
        "total_return": float(total_return),
        "max_drawdown": float(mdd),
        "sharpe": float(sharpe),
        "trades": int(trades),
        "minutes": int(len(equity)),
    }

    if verbose:
        print("Final equity:", round(out["final_equity"], 2))
        print("Total return:", round(out["total_return"] * 100, 3), "%")
        print("Max drawdown:", round(out["max_drawdown"] * 100, 3), "%")
        print("Sharpe (per-minute):", round(out["sharpe"], 4))
        print("Trades:", out["trades"])
        print("Minutes:", out["minutes"])

    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(r"Usage: python backtest.py logs\ticker_history.csv")
        sys.exit(1)
    backtest(sys.argv[1])