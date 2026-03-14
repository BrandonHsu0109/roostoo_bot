import sys
import numpy as np
import pandas as pd

def max_drawdown(equity):
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(dd.min()) if len(dd) else 0.0

def backtest_csmr(
    csv_path: str,
    initial_usd: float = 50000.0,
    lookback: int = 30,                 # use 15 or 30; your IC says these work
    vol_window: int = 60,
    bottom_n: int = 5,                  # long the worst N performers
    cash_buffer: float = 0.40,
    max_weight_per_coin: float = 0.25,
    min_unit_value: float = 8_000_000.0,
    max_spread_pct: float = 0.0031,
    rebalance_every: int = 10,          # IMPORTANT: trade less
    max_orders_per_rebalance: int = 1,
    min_trade_usd: float = 50.0,
    min_delta_usd: float = 200.0,       # BUY only (exits not blocked)
    fee_rate: float = 0.00012,
    eps: float = 1e-12,
):
    df = pd.read_csv(csv_path)
    df["tmin"] = (df["ts"].astype(np.int64)//60000)*60000
    df = df.sort_values(["tmin","pair","ts"]).groupby(["tmin","pair"], as_index=False).last()

    df["bid"] = df["bid"].astype(float)
    df["ask"] = df["ask"].astype(float)
    df["unit_value"] = df["unit_value"].astype(float)
    df["mid"] = (df["bid"]+df["ask"])/2.0
    df["spr"] = (df["ask"]-df["bid"])/df["mid"]

    mid = df.pivot(index="tmin", columns="pair", values="mid").sort_index()
    bid = df.pivot(index="tmin", columns="pair", values="bid").reindex(mid.index)
    ask = df.pivot(index="tmin", columns="pair", values="ask").reindex(mid.index)
    unitv = df.pivot(index="tmin", columns="pair", values="unit_value").reindex(mid.index)
    spr = df.pivot(index="tmin", columns="pair", values="spr").reindex(mid.index)
    rets = mid.pct_change(fill_method=None)

    pairs = mid.columns.to_list()
    cash = float(initial_usd)
    pos = {p: 0.0 for p in pairs}
    equity_curve = []
    trades = 0

    need = max(lookback+1, vol_window+1)

    for i in range(len(mid.index)):
        pv = cash
        for p in pairs:
            q = pos[p]
            if q != 0.0:
                b = bid.iloc[i][p]
                if pd.notna(b) and b > 0:
                    pv += q * float(b)

        if i < need:
            equity_curve.append(pv); continue
        if (i % int(rebalance_every)) != 0:
            equity_curve.append(pv); continue

        base_ok = (
            (unitv.iloc[i] >= float(min_unit_value)) &
            (spr.iloc[i] <= float(max_spread_pct)) &
            mid.iloc[i].notna() & bid.iloc[i].notna() & ask.iloc[i].notna()
        )
        if int(base_ok.sum()) == 0:
            target = {p: 0.0 for p in pairs}
        else:
            mom = (mid.iloc[i] / mid.iloc[i - lookback]) - 1.0

            # losers = most negative momentum (mean reversion bet)
            mom_ok = mom.where(base_ok).dropna()
            target = {p: 0.0 for p in pairs}

            if len(mom_ok) > 0:
                losers = mom_ok.sort_values(ascending=True).head(int(bottom_n))
                # only long "real losers" (avoid tiny negatives near 0)
                losers = losers[losers < 0]
                if len(losers) > 0:
                    vol = rets.iloc[i - vol_window + 1: i + 1].std(ddof=1)
                    score = (1.0 / (vol + eps)).reindex(losers.index)
                    score = score.replace([np.inf, -np.inf], np.nan).dropna()

                    if len(score) > 0:
                        investable = float(pv) * (1.0 - float(cash_buffer))
                        w = score.to_numpy(dtype=float)
                        w = w / float(np.sum(w))
                        w = np.minimum(w, float(max_weight_per_coin))
                        w = w / float(np.sum(w))
                        for p, wi in zip(score.index.to_list(), w):
                            m = mid.iloc[i][p]
                            if pd.notna(m) and float(m) > 0:
                                target[p] = (investable * float(wi)) / float(m)

        # execute at most K orders; min_delta only for BUY
        cands = []
        for p in pairs:
            cur = float(pos[p]); tgt = float(target[p])
            delta = tgt - cur
            if abs(delta) < 1e-12: continue
            if delta > 0:
                side="BUY"; px=mid.iloc[i][p]; qty=float(delta)
            else:
                side="SELL"; px=mid.iloc[i][p]; qty=float(-delta)

            if pd.isna(px) or float(px) <= 0 or qty <= 0: continue
            notional = float(px) * qty
            if notional < float(min_trade_usd): continue
            if side=="BUY" and notional < float(min_delta_usd): continue
            cands.append((notional,p,side,qty,float(px)))

        cands.sort(key=lambda x:x[0], reverse=True)
        for notional,p,side,qty,px in cands[:int(max_orders_per_rebalance)]:
            fee = notional * float(fee_rate)
            if side=="BUY":
                if cash >= notional+fee:
                    cash -= (notional+fee); pos[p]+=qty; trades+=1
            else:
                if pos[p] >= qty:
                    pos[p]-=qty; cash += (notional-fee); trades+=1

        pv2 = cash
        for p in pairs:
            q = pos[p]
            if q != 0.0:
                b = bid.iloc[i][p]
                if pd.notna(b) and b > 0:
                    pv2 += q * float(b)
        equity_curve.append(float(pv2))

    equity = np.array(equity_curve, dtype=float)
    total_return = (equity[-1]/equity[0]) - 1.0
    mdd = max_drawdown(equity)
    r = np.diff(equity)/equity[:-1]
    sharpe = float(np.mean(r)/(np.std(r,ddof=1)+1e-12)) if len(r)>=2 else 0.0

    print("Final equity:", round(float(equity[-1]),2))
    print("Total return:", round(total_return*100,3), "%")
    print("Max drawdown:", round(mdd*100,3), "%")
    print("Sharpe (per-minute):", round(sharpe,4))
    print("Trades:", trades)
    print("Minutes:", len(equity))

if __name__=="__main__":
    if len(sys.argv)<2:
        print(r"Usage: python backtest_csmr.py logs\ticker_history.csv")
        raise SystemExit(1)
    backtest_csmr(sys.argv[1])