# tune.py
#
# Purpose:
# - Grid search over a small set of strategy + execution parameters
# - Uses your collected CSV (logs\ticker_history.csv)
# - Requires backtest.py to define backtest.backtest(...) that RETURNS a dict:
#     {"final_equity","total_return","max_drawdown","sharpe","trades","minutes"}
#
# Run:
#   python tune.py
#
# Tip:
# - Keep the grid SMALL. You want signal, not a huge search that overfits.

import backtest

CSV = r"logs\ticker_history.csv"


def run():
    # ----------------------------
    # Fixed “execution realism” / portfolio constraints
    # ----------------------------
    base = dict(
        cash_buffer=0.40,
        top_n=5,
        max_weight_per_coin=0.25,
        fee_rate=0.00012,           # approx taker fee in examples; adjust if you learn real fee
        min_trade_usd=50.0,
        min_delta_usd=200.0,        # ignore tiny adjustments (reduces churn)
        max_orders_per_rebalance=1, # contest-like (roughly 1 trade per decision)
        vol_window_minutes=60,      # keep stable unless you have lots of data
    )

    # ----------------------------
    # Grid (small, purposeful)
    # ----------------------------
    # Spread caps based on your empirical percentiles:
    # 90% ~ 0.002283, 95% ~ 0.0031
    max_spreads = [0.0031, 0.002283]

    # Horizons: 30 vs 60 (you found 60 often better)
    horizons = [60, 30]

    # Breadth regime gate (market must be “trending enough”)
    breadths = [0.55, 0.45]

    # Momentum thresholds (cost-aware levels)
    mom_thresholds = [0.01, 0.0075, 0.005]

    # Rebalance cadence (in minutes)
    rebalances = [5, 10]

    # Liquidity filter (UnitTradeValue) candidates based on your stats:
    # 50% ~ 11.6M, 75% ~ 26.8M
    # Include 0.0 to test "no filter"
    min_unit_values = [0.0, 8_000_000.0, 12_000_000.0, 27_000_000.0]

    # ----------------------------
    # Run grid
    # ----------------------------
    results = []
    total = 0

    for max_spread in max_spreads:
        for horizon in horizons:
            for breadth in breadths:
                for mom_th in mom_thresholds:
                    for rebalance_every in rebalances:
                        for min_unit_value in min_unit_values:
                            total += 1
                            m = backtest.backtest(
                                CSV,
                                horizon_minutes=horizon,
                                max_spread_pct=max_spread,
                                mom_threshold=mom_th,
                                breadth_threshold=breadth,
                                rebalance_every=rebalance_every,
                                min_unit_value=min_unit_value,
                                **base
                            )
                            results.append((
                                m,
                                max_spread,
                                horizon,
                                breadth,
                                mom_th,
                                rebalance_every,
                                min_unit_value
                            ))

    # ----------------------------
    # Rank and print
    # ----------------------------
    # Primary sort: final equity (leaderboard-like)
    # Secondary sort: sharpe (stability)
    results.sort(key=lambda x: (x[0]["final_equity"], x[0]["sharpe"]), reverse=True)

    print(f"Evaluated configs: {total}")
    print("Top 15 configs by final equity (then sharpe):")
    for (m, max_spread, horizon, breadth, mom_th, rebalance_every, min_unit_value) in results[:15]:
        print(
            f"equity={m['final_equity']:.2f} "
            f"ret={m['total_return']*100:.3f}% "
            f"mdd={m['max_drawdown']*100:.3f}% "
            f"sharpe={m['sharpe']:.4f} "
            f"trades={m['trades']} "
            f"max_spread={max_spread} "
            f"horizon={horizon} "
            f"mom_th={mom_th} "
            f"breadth={breadth} "
            f"rebalance={rebalance_every} "
            f"min_unit={min_unit_value:.0f}"
        )

    # Also print a "sanity" list: configs that trade at least a little
    # (to avoid the trivial “0 trades == 0 return” solutions)
    print("\nTop 10 configs with >= 5 trades:")
    traded = [r for r in results if r[0]["trades"] >= 5]
    for (m, max_spread, horizon, breadth, mom_th, rebalance_every, min_unit_value) in traded[:10]:
        print(
            f"equity={m['final_equity']:.2f} "
            f"ret={m['total_return']*100:.3f}% "
            f"mdd={m['max_drawdown']*100:.3f}% "
            f"sharpe={m['sharpe']:.4f} "
            f"trades={m['trades']} "
            f"max_spread={max_spread} "
            f"horizon={horizon} "
            f"mom_th={mom_th} "
            f"breadth={breadth} "
            f"rebalance={rebalance_every} "
            f"min_unit={min_unit_value:.0f}"
        )


if __name__ == "__main__":
    run()