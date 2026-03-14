# backtest_rare_contrarian_v2.py
#
# Rare-event long-only contrarian with:
# - ONE position at a time
# - hard exits checked every minute (TP / SL / time stop)
# - market regime gate (skip entries during broad selloffs)
# - entry quality filters:
#     (A) dip band-pass: entry_rL_floor <= rL <= entry_rL
#     (B) stabilization: require short-term bounce/slowdown before buying
# - detailed trade logging
#
# Run:
#   python -c "import backtest_rare_contrarian_v2 as b; b.backtest_rare_contrarian(r'logs\\ticker_history.csv', ...)"

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

    # Signal lookback in minutes
    lookback: int = 30,

    # Entry dip threshold: buy when rL <= entry_rL (e.g. -0.03)
    entry_rL: float = -0.03,

    # OPTIONAL dip floor: also require rL >= entry_rL_floor (band-pass).
    # This avoids buying the *most* extreme dumps (often falling knives).
    entry_rL_floor: float | None = -0.04,   # set None to disable

    # OPTIONAL 24h filter (can set None)
    entry_ch24_max: float | None = -0.05,

    # Market regime gate (skip entries if whole market is dumping)
    market_r_min: float = -0.0025,          # stricter gate (median rL must be >= -0.25%)

    # Stabilization confirmation (to avoid buying while still dumping)
    # Require last 1-min return >= confirm_r1_min (0.0 means not negative)
    confirm_r1_min: float | None = 0.0,
    # Require last 3-min return >= confirm_r3_min (e.g. -0.5% over 3 minutes)
    confirm_r3_min: float | None = -0.005,

    # Exits (checked every minute)
    take_profit: float = 0.006,
    stop_loss: float = -0.01,
    max_hold_minutes: int = 30,

    # Sizing
    invest_frac: float = 0.05,

    # Filters
    min_unit_value: float = 8_000_000.0,
    max_spread_pct: float = 0.0031,

    # Execution cadence and costs
    rebalance_every: int = 5,
    fee_rate: float = 0.00012,
    min_trade_usd: float = 50.0,

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
    entry_px: float | None = None
    entry_i: int | None = None

    equity_curve: list[float] = []
    trades = 0

    trade_log: list[tuple] = []

    need = int(lookback) + 3  # need extra for confirm_r3

    for i in range(len(mid.index)):
        tmin_i = int(mid.index[i])

        # PV at bid
        pv = cash
        if pos_pair is not None and pos_qty > 0:
            b = bid.iloc[i].get(pos_pair)
            if pd.notna(b) and float(b) > 0:
                pv += pos_qty * float(b)

        if i < need:
            equity_curve.append(float(pv))
            continue

        # Exit check every minute
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
                    pv = cash

        # Entry only on rebalance minute and only if flat
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
                    # candidate dips
                    candidates = rL[rL <= float(entry_rL)].dropna()

                    # Dip floor (band-pass)
                    if entry_rL_floor is not None and len(candidates) > 0:
                        candidates = candidates[candidates >= float(entry_rL_floor)]

                    # 24h filter
                    if entry_ch24_max is not None and len(candidates) > 0:
                        c = ch24.iloc[i].reindex(candidates.index)
                        candidates = candidates[c <= float(entry_ch24_max)]

                    if len(candidates) > 0:
                        pick = candidates.sort_values(ascending=True).index[0]

                        m_now = mid.iloc[i].get(pick)
                        m_1 = mid.iloc[i - 1].get(pick)
                        m_3 = mid.iloc[i - 3].get(pick)
                        if pd.isna(m_now) or pd.isna(m_1) or pd.isna(m_3) or float(m_now) <= 0 or float(m_1) <= 0 or float(m_3) <= 0:
                            equity_curve.append(float(pv))
                            continue

                        r1 = float(m_now / m_1 - 1.0)
                        r3 = float(m_now / m_3 - 1.0)

                        # Stabilization confirmation
                        if confirm_r1_min is not None and r1 < float(confirm_r1_min):
                            equity_curve.append(float(pv))
                            continue
                        if confirm_r3_min is not None and r3 < float(confirm_r3_min):
                            equity_curve.append(float(pv))
                            continue

                        a = ask.iloc[i].get(pick)
                        if pd.isna(a) or float(a) <= 0:
                            equity_curve.append(float(pv))
                            continue

                        invest = float(pv) * float(invest_frac)
                        if invest < float(min_trade_usd):
                            equity_curve.append(float(pv))
                            continue

                        qty = invest / float(a)
                        notional = float(a) * float(qty)
                        fee = notional * float(fee_rate)

                        if cash >= notional + fee:
                            cash -= (notional + fee)
                            pos_pair = pick
                            pos_qty = float(qty)
                            entry_px = float(a)
                            entry_i = i
                            trades += 1

                            trade_log.append((
                                "BUY", tmin_i, pick, float(a), float(invest), 0, "entry",
                                {
                                    "rL_pick": float(candidates.loc[pick]),
                                    "market_r": float(market_r),
                                    "ch24_pick": float(ch24.iloc[i].get(pick)) if pd.notna(ch24.iloc[i].get(pick)) else None,
                                    "unitv_pick": float(unitv.iloc[i].get(pick)) if pd.notna(unitv.iloc[i].get(pick)) else None,
                                    "spr_pick": float(spr.iloc[i].get(pick)) if pd.notna(spr.iloc[i].get(pick)) else None,
                                    "r1": r1,
                                    "r3": r3,
                                }
                            ))

                            # recompute pv after entry
                            bnow = bid.iloc[i].get(pos_pair)
                            pv = cash
                            if pd.notna(bnow) and float(bnow) > 0:
                                pv += pos_qty * float(bnow)

        equity_curve.append(float(pv))

    # force close at end
    if pos_pair is not None and pos_qty > 0 and entry_px is not None and entry_i is not None:
        b = bid.iloc[-1].get(pos_pair)
        if pd.notna(b) and float(b) > 0:
            pnl = (float(b) / float(entry_px)) - 1.0
            held = (len(mid.index) - 1) - entry_i
            notional = float(b) * float(pos_qty)
            fee = notional * float(fee_rate)
            cash += (notional - fee)
            trades += 1
            trade_log.append(("SELL", int(mid.index[-1]), pos_pair, float(b), float(pnl), int(held), "final_liquidation"))
            equity_curve[-1] = float(cash)

    equity = np.asarray(equity_curve, dtype=float)
    total_return = (equity[-1] / equity[0]) - 1.0 if len(equity) else 0.0
    mdd = max_drawdown(equity)
    r = np.diff(equity) / equity[:-1] if len(equity) >= 2 else np.array([])
    sharpe = float(np.mean(r) / (np.std(r, ddof=1) + eps)) if len(r) >= 2 else 0.0

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
        print(r"Usage: python backtest_rare_contrarian_v2.py logs\ticker_history.csv")
        raise SystemExit(1)
    backtest_rare_contrarian(sys.argv[1])