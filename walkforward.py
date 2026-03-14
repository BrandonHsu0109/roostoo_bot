import backtest

CSV = r"logs\ticker_history.csv"

# Pick the config you want to evaluate (start with Config A)
PARAMS = dict(
    horizon_minutes=30,
    vol_window_minutes=60,
    top_n=5,
    cash_buffer=0.40,
    max_weight_per_coin=0.25,
    min_unit_value=8_000_000.0,
    max_spread_pct=0.002283,
    mom_threshold=0.005,
    breadth_threshold=0.55,
    rebalance_every=5,
    max_orders_per_rebalance=1,
    min_trade_usd=50.0,
    min_delta_usd=200.0,
    fee_rate=0.0,
)

def run():
    # We will test on last 25%, last 33%, last 50% (three holdouts)
    splits = [0.75, 0.67, 0.50]
    for frac in splits:
        import pandas as pd
        df = pd.read_csv(CSV)
        df["tmin"] = (df["ts"] // 60000) * 60000
        mins = sorted(df["tmin"].unique())
        cut = int(len(mins) * frac)

        train_mins = set(mins[:cut])
        test_mins = set(mins[cut:])

        train = df[df["tmin"].isin(train_mins)].drop(columns=["tmin"])
        test = df[df["tmin"].isin(test_mins)].drop(columns=["tmin"])

        train_path = rf"logs\wf_train_{int(frac*100)}.csv"
        test_path  = rf"logs\wf_test_{int(frac*100)}.csv"
        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)

        print(f"\n=== Split train={int(frac*100)}% / test={int((1-frac)*100)}% ===")
        print("TRAIN")
        backtest.backtest(train_path, **PARAMS)
        print("TEST")
        backtest.backtest(test_path, **PARAMS)

if __name__ == "__main__":
    run()