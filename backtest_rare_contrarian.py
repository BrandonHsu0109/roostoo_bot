# backtest_rare_contrarian.py
#
# Rare-event long-only contrarian with:
# - ONE position at a time
# - hard exits checked every minute (TP / SL / time stop)
# - market regime gate (skip entries during broad selloffs)
# - robust trade logging so you can see EXACTLY when/why trades happen
#
# Input CSV columns expected:
#   ts, pair, bid, ask, last, change_24h, coin_value, unit_value
#
# Run:
#   python backtest_rare_contrarian.py logs\ticker_history.csv
#
# Or from command line:
#   python -c "import backtest_rare_contrarian as b; b.backtest_rare_contrarian(r'logs\\ticker_history.csv', lookback=30, entry_rL=-0.03)"

from __future__ import annotations

import sys
import numpy as np
import pandas as pd


def max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(dd.min())


def backtest_rare_contrarian(
    csv_path: str,
    *,
    initial_usd: float = 50000.0,

    # Signal (oversold) lookback in minutes
    lookback: int = 30,
    entry_rL: float = -0.02,               # enter if rL <= this threshold (e.g., -0.03)
    entry_ch24_max: float | None = -0.05,  # if set: require 24h change <= -5% (can set None)

    # Market regime gate: skip entries if whole market is dumping
    market_r_min: float = -0.005,          # if median rL across base_ok < this, skip entries

    # Exits (checked every minute)
    take_profit: float = 0.01,             # +1%
    stop_loss: float = -0.015,             # -1.5%
    max_hold_minutes: int = 60,            # time stop (minutes)

    # Sizing
    invest_frac: float = 0.10,             # fraction of PV to deploy when entering

    # Filters
    min_unit_value: float = 8_000_000.0,
    max_spread_pct: float = 0.0031,

    # Execution cadence and costs
    rebalance_every: int = 5,
    fee_rate: float = 0.00012,
    min_trade_usd: float = 50.0,

    # Misc
    eps: float = 1e-12,
    verbose: bool = True,
    print_trade_log: bool = True,
) -> dict:
    df = pd.read_csv(csv_path)

    # minute bucket + last per (minute, pair)
    df["tmin"] = (df["ts"].astype(np.int64) // 60000) * 60000
    df = df.sort_values(["tmin", "pair", "ts"]).groupby(["tmin", "pair"], as_index=False).last()

    # numeric
    df["bid"] = df["bid"].astype(float)
    df["ask"] = df["ask"].astype(float)
    df["unit_value"] = df["unit_value"].astype(float)
    df["change_24h"] = df["change_24h"].astype(float)

    # derived
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df["spr"] = np.where(df["mid"] > 0, (df["ask"] - df["bid"]) / df["mid"], np.nan)

    # pivot matrices
    mid = df.pivot(index="tmin", columns="pair", values="mid").sort_index()
    bid = df.pivot(index="tmin", columns="pair", values="bid").reindex(mid.index)
    ask = df.pivot(index="tmin", columns="pair", values="ask").reindex(mid.index)
    unitv = df.pivot(index="tmin", columns="pair", values="unit_value").reindex(mid.index)
    spr = df.pivot(index="tmin", columns="pair", values="spr").reindex(mid.index)
    ch24 = df.pivot(index="tmin", columns="pair", values="change_24h").reindex(mid.index)

    pairs = mid.columns.to_list()

    # Portfolio state: at most ONE open position
    cash = float(initial_usd)
    pos_pair: str | None = None
    pos_qty: float = 0.0
    entry_px: float | None = None   # entry price (ask)
    entry_i: int | None = None      # entry minute index

    equity_curve: list[float] = []
    trades = 0

    # Trade log rows:
    # ("BUY"/"SELL", tmin, pair, px, notional_or_pnl, held, reason, extra...)
    trade_log: list[tuple] = []

    need = int(lookback) + 1

    for i in range(len(mid.index)):
        tmin_i = int(mid.index[i])

        # --- Mark-to-market PV at bid (conservative) ---
        pv = cash
        if pos_pair is not None and pos_qty > 0:
            b = bid.iloc[i].get(pos_pair)
            if pd.notna(b) and float(b) > 0:
                pv += pos_qty * float(b)

        # warm-up
        if i < need:
            equity_curve.append(float(pv))
            continue

        # --- Exit check every minute ---
        if pos_pair is not None and pos_qty > 0 and entry_px is not None and entry_i is not None:
            b = bid.iloc[i].get(pos_pair)
            if pd.notna(b) and float(b) > 0:
                pnl = (float(b) / float(entry_px)) - 1.0
                held = i - entry_i

                exit_reason = None
                if pnl >= float(take_profit):
                    exit_reason = "take_profit"
                if pnl <= float(stop_loss):
                    exit_reason = "stop_loss"
                if held >= int(max_hold_minutes):
                    exit_reason = "time_stop" if exit_reason is None else (exit_reason + "+time_stop")

                if exit_reason is not None:
                    notional = float(b) * float(pos_qty)
                    fee = notional * float(fee_rate)
                    cash += (notional - fee)
                    trades += 1

                    trade_log.append(("SELL", tmin_i, pos_pair, float(b), float(pnl), int(held), exit_reason))

                    pos_pair, pos_qty, entry_px, entry_i = None, 0.0, None, None

                    # recompute PV after exit
                    pv = cash

        # --- Entry logic only on rebalance minutes and only if flat ---
        if (i % int(rebalance_every)) == 0 and pos_pair is None:
            base_ok = (
                (unitv.iloc[i] >= float(min_unit_value)) &
                (spr.iloc[i] <= float(max_spread_pct)) &
                mid.iloc[i].notna() & bid.iloc[i].notna() & ask.iloc[i].notna()
            )

            if int(base_ok.sum()) > 0:
                rL = (mid.iloc[i] / mid.iloc[i - int(lookback)]) - 1.0
                rL = rL.where(base_ok)

                market_r = float(rL.median(skipna=True)) if rL.notna().any() else np.nan

                # Market regime gate
                if np.isfinite(market_r) and market_r >= float(market_r_min):
                    # Entry candidates: extreme losers
                    candidates = rL[rL <= float(entry_rL)].dropna()

                    # Optional 24h filter
                    if entry_ch24_max is not None and len(candidates) > 0:
                        c = ch24.iloc[i].reindex(candidates.index)
                        candidates = candidates[c <= float(entry_ch24_max)]

                    if len(candidates) > 0:
                        pick = candidates.sort_values(ascending=True).index[0]
                        a = ask.iloc[i].get(pick)

                        if pd.notna(a) and float(a) > 0:
                            invest = float(pv) * float(invest_frac)

                            if invest >= float(min_trade_usd):
                                qty = invest / float(a)
                                notional = float(a) * float(qty)
                                fee = notional * float(fee_rate)

                                if cash >= (notional + fee):
                                    cash -= (notional + fee)
                                    pos_pair = pick
                                    pos_qty = float(qty)
                                    entry_px = float(a)  # entry at ask
                                    entry_i = i
                                    trades += 1

                                    trade_log.append((
                                        "BUY", tmin_i, pick, float(a),
                                        float(invest),
                                        0,
                                        "entry",
                                        {
                                            "rL_pick": float(candidates.loc[pick]),
                                            "market_r": float(market_r),
                                            "lookback": int(lookback),
                                            "entry_rL": float(entry_rL),
                                            "entry_ch24_max": entry_ch24_max,
                                            "market_r_min": float(market_r_min),
                                            "take_profit": float(take_profit),
                                            "stop_loss": float(stop_loss),
                                            "max_hold_minutes": int(max_hold_minutes),
                                            "invest_frac": float(invest_frac),
                                        }
                                    ))

                                    # recompute PV after entry (mark at bid)
                                    bnow = bid.iloc[i].get(pos_pair)
                                    pv = cash
                                    if pd.notna(bnow) and float(bnow) > 0:
                                        pv += pos_qty * float(bnow)

        equity_curve.append(float(pv))

    # Force close at end so equity reflects realized liquidation
    if pos_pair is not None and pos_qty > 0:
        b = bid.iloc[-1].get(pos_pair)
        if pd.notna(b) and float(b) > 0 and entry_px is not None and entry_i is not None:
            pnl = (float(b) / float(entry_px)) - 1.0
            held = (len(mid.index) - 1) - entry_i
            notional = float(b) * float(pos_qty)
            fee = notional * float(fee_rate)
            cash += (notional - fee)
            trades += 1
            trade_log.append(("SELL", int(mid.index[-1]), pos_pair, float(b), float(pnl), int(held), "final_liquidation"))
            pos_pair, pos_qty, entry_px, entry_i = None, 0.0, None, None
            # update last equity point
            equity_curve[-1] = float(cash)

    equity = np.asarray(equity_curve, dtype=float)
    total_return = (equity[-1] / equity[0]) - 1.0 if len(equity) else 0.0
    mdd = max_drawdown(equity)

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

    if print_trade_log:
        print("\nTRADE LOG (all trades):")
        for row in trade_log:
            print(row)

    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(r"Usage: python backtest_rare_contrarian.py logs\ticker_history.csv")
        raise SystemExit(1)
    backtest_rare_contrarian(sys.argv[1])