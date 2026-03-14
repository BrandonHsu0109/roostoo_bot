import pandas as pd
import numpy as np

CSV = r"logs\ticker_history.csv"

def information_coefficient(x: pd.Series, y: pd.Series) -> float:
    # Spearman IC (rank correlation). Robust to outliers.
    if x.isna().all() or y.isna().all():
        return np.nan
    return x.corr(y, method="spearman")

def main():
    df = pd.read_csv(CSV)

    # minute bucket + last per (minute, pair)
    df["tmin"] = (df["ts"].astype(np.int64) // 60000) * 60000
    df = df.sort_values(["tmin", "pair", "ts"]).groupby(["tmin", "pair"], as_index=False).last()

    # mid/spread
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df["spr"] = (df["ask"] - df["bid"]) / df["mid"]

    # pivot matrices
    mid = df.pivot(index="tmin", columns="pair", values="mid").sort_index()
    spr = df.pivot(index="tmin", columns="pair", values="spr").reindex(mid.index)
    unitv = df.pivot(index="tmin", columns="pair", values="unit_value").reindex(mid.index)
    ch24 = df.pivot(index="tmin", columns="pair", values="change_24h").reindex(mid.index)  # already percent (e.g., 0.01)

    # forward returns (what we're trying to predict)
    fwd5 = mid.shift(-5) / mid - 1.0
    fwd15 = mid.shift(-15) / mid - 1.0
    fwd30 = mid.shift(-30) / mid - 1.0

    # feature helpers
    def mom(L):  # momentum over L minutes
        return mid / mid.shift(L) - 1.0

    def zscore(L):  # mean-reversion signal: (price - MA)/std
        ma = mid.rolling(L).mean()
        sd = mid.rolling(L).std(ddof=1)
        return (mid - ma) / (sd + 1e-12)

    feats = {
        "mom_5": mom(5),
        "mom_15": mom(15),
        "mom_30": mom(30),
        "mom_60": mom(60),
        "z_30": zscore(30),
        "z_60": zscore(60),
        "ch24": ch24,                      # 24h change from API
        "spr": spr,                        # spread
        "log_unitv": np.log(unitv.clip(lower=1.0)),
    }

    targets = {"fwd5": fwd5, "fwd15": fwd15, "fwd30": fwd30}

    # compute average IC across minutes (cross-sectional per minute)
    results = []
    common_index = mid.index

    for feat_name, X in feats.items():
        for tgt_name, Y in targets.items():
            ics = []
            for t in common_index:
                x = X.loc[t]
                y = Y.loc[t]
                # require enough finite values
                m = x.notna() & y.notna()
                if m.sum() < 20:
                    continue
                ics.append(information_coefficient(x[m], y[m]))
            avg_ic = float(np.nanmean(ics)) if len(ics) else np.nan
            results.append((feat_name, tgt_name, avg_ic, len(ics)))

    out = pd.DataFrame(results, columns=["feature", "target", "avg_spearman_ic", "n_minutes"])
    out = out.sort_values("avg_spearman_ic", ascending=False)

    print(out.head(20).to_string(index=False))
    print("\nBottom 10 (most negative IC):")
    print(out.tail(10).to_string(index=False))

if __name__ == "__main__":
    main()