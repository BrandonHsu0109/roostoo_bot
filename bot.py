# bot.py
from __future__ import annotations

import os
import time
import csv
import json
import math
import argparse
from typing import Dict, Any, Optional, List

from roostoo_client import RoostooClient
from rules import parse_exchange_info, PairRule
from store import MarketStore
from strategy import RareEventContrarian, RareContrarianConfig
from execution import ExecutionEngine, ExecutionConfig


def portfolio_value_usd(rules: Dict[str, PairRule], store: MarketStore, balance_json: Dict[str, Any]) -> float:
    wallet = balance_json.get("Wallet", {})
    usd = float(wallet.get("USD", {}).get("Free", 0.0)) + float(wallet.get("USD", {}).get("Lock", 0.0))
    total = usd

    for pair, rule in rules.items():
        coin = getattr(rule, "coin", None)
        if not coin:
            continue
        qty = float(wallet.get(coin, {}).get("Free", 0.0)) + float(wallet.get(coin, {}).get("Lock", 0.0))
        if qty <= 0:
            continue
        latest = store.latest(pair)
        if not latest:
            continue
        bid = float(latest.get("bid", 0.0))
        if bid > 0:
            total += qty * bid

    return total


def ensure_csv_header(path: str, header: list) -> None:
    if os.path.exists(path):
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(header)


def append_tickers_to_csv(path: str, server_time_ms: int, ticker_data: Dict[str, Any]) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for pair, d in ticker_data.items():
            w.writerow([
                server_time_ms,
                pair,
                d.get("MaxBid"),
                d.get("MinAsk"),
                d.get("LastPrice"),
                d.get("Change"),
                d.get("CoinTradeValue"),
                d.get("UnitTradeValue"),
            ])


def append_decision_jsonl(path: str, record: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def round_down(x: float, decimals: int) -> float:
    p = 10 ** max(decimals, 0)
    return math.floor(x * p) / p


def round_up(x: float, decimals: int) -> float:
    p = 10 ** max(decimals, 0)
    return math.ceil(x * p) / p


def pick_bootstrap_pair(ticker_data: Dict[str, Any], rules: Dict[str, PairRule]) -> Optional[str]:
    """
    Pick among tradable pairs: top liquidity then smallest spread.
    """
    rows = []
    for pair, d in ticker_data.items():
        r = rules.get(pair)
        if r is None or not getattr(r, "can_trade", False):
            continue
        try:
            bid = float(d.get("MaxBid", 0.0))
            ask = float(d.get("MinAsk", 0.0))
            unitv = float(d.get("UnitTradeValue", 0.0))
            if bid <= 0 or ask <= 0 or ask <= bid:
                continue
            mid = (bid + ask) / 2.0
            spr = (ask - bid) / mid
            rows.append((pair, unitv, spr))
        except Exception:
            continue

    if not rows:
        return None

    rows.sort(key=lambda x: x[1], reverse=True)   # liquidity desc
    top = rows[:10]
    top.sort(key=lambda x: x[2])                  # spread asc
    return top[0][0]


def compute_bootstrap_qty(pair: str, ask: float, notional_usd: float, rule: PairRule) -> float:
    """
    Convert notional USD to quantity; respect amount precision and MiniOrder (notional) if present.
    """
    amt_prec = int(getattr(rule, "amount_precision", 6))
    min_notional = float(getattr(rule, "min_order", 0.0) or 0.0)

    target_notional = max(float(notional_usd), min_notional)

    # qty based on ask
    qty = target_notional / ask

    # Ensure notional constraint by rounding UP for the minimum qty
    min_qty = (min_notional / ask) if min_notional > 0 else 0.0
    qty = max(qty, min_qty)

    # round to exchange precision
    qty = round_down(qty, amt_prec)

    # if rounding down broke min_notional, round UP just enough
    if min_notional > 0 and (qty * ask) < min_notional:
        qty = round_up(min_qty, amt_prec)

    return max(qty, 0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop-seconds", type=int, default=60)
    parser.add_argument("--history-minutes", type=int, default=720)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--public-only", action="store_true")
    parser.add_argument("--logdir", type=str, default="logs")

    # Bootstrap (deployment compliance)
    parser.add_argument("--no-bootstrap", action="store_true", help="Disable bootstrap trade fallback.")
    parser.add_argument("--bootstrap-deadline-min", type=int, default=23 * 60)
    parser.add_argument("--bootstrap-notional-usd", type=float, default=60.0)  # >= 50 to clear any min_trade filters

    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)

    api_key = os.getenv("ROOSTOO_API_KEY", "")
    secret_key = os.getenv("ROOSTOO_SECRET_KEY", "")
    base_url = os.getenv("ROOSTOO_BASE_URL", "https://mock-api.roostoo.com")

    if (not args.public_only) and (not api_key or not secret_key):
        raise RuntimeError("Missing ROOSTOO_API_KEY or ROOSTOO_SECRET_KEY environment variables.")

    client = RoostooClient(
        base_url=base_url,
        api_key=api_key if api_key else "DUMMY",
        secret_key=secret_key if secret_key else "DUMMY",
    )

    # Time sync
    try:
        offset = client.sync_time()
    except Exception as e:
        offset = 0
        print(f"[startup] time sync failed: {e}")
    print(f"[startup] time offset ms = {offset}")

    # Exchange rules
    info = client.get_exchange_info()
    rules = parse_exchange_info(info)
    tradable_pairs = [p for p, r in rules.items() if getattr(r, "can_trade", False)]
    print(f"[startup] tradable pairs: {len(tradable_pairs)}")

    # Store
    store = MarketStore(maxlen=args.history_minutes)

    # Logs
    ticker_csv = os.path.join(args.logdir, "ticker_history.csv")
    ensure_csv_header(ticker_csv, ["ts", "pair", "bid", "ask", "last", "change_24h", "coin_value", "unit_value"])
    decision_jsonl = os.path.join(args.logdir, "decisions.jsonl")

    # Strategy config (baseline you validated)
    cfg = RareContrarianConfig(
        lookback_minutes=30,
        entry_rL=-0.025,
        market_r_min=-0.0025,
        confirm_r1_min=0.002,
        confirm_r3_min=-0.005,
        entry_ch24_max=None,
        invest_frac=0.05,
        take_profit=0.006,
        stop_loss=-0.01,
        max_hold_minutes=30,
        min_unit_value=8_000_000.0,
        max_spread_pct=0.0031,
    )
    strat = RareEventContrarian(cfg)

    # Execution
    exec_engine: Optional[ExecutionEngine] = None
    if not args.public_only:
        exec_engine = ExecutionEngine(
            client=client,
            rules=rules,
            cfg=ExecutionConfig(
                min_trade_usd=10.0,          # allow bootstrap small trades
                max_orders_per_cycle=1,
                dry_run=args.dry_run,
                use_market_only=True,
            )
        )

    # Cadence
    rebalance_every = 1

    # Time resync policy
    resync_every_sec = 10 * 60
    last_resync = time.time()

    initial_usd = float(info.get("InitialWallet", {}).get("USD", 50000.0))
    last_pv = initial_usd

    # Bootstrap state
    bootstrap_enabled = (not args.no_bootstrap) and (not args.public_only)
    bootstrap_done_path = os.path.join(args.logdir, "bootstrap_done.json")
    bootstrap_stage = 0  # 0=idle, 1=bought-wait-sell, 2=done
    bootstrap_pair: Optional[str] = None
    bootstrap_buy_minute: Optional[int] = None
    bot_start_server_ms: Optional[int] = None
    have_any_trade_action = False

    if os.path.exists(bootstrap_done_path):
        bootstrap_stage = 2

    print("[startup] running loop... (Ctrl+C to stop)")
    while True:
        loop_start = time.time()

        # periodic time resync
        if loop_start - last_resync >= resync_every_sec:
            try:
                offset = client.sync_time()
                print(f"[time] resynced offset ms = {offset}")
            except Exception as e:
                print(f"[time] resync failed: {e}")
            last_resync = loop_start

        # --- ticker fetch (robust) ---
        try:
            ticker = client.get_ticker(pair=None)
        except Exception as e:
            print(f"[ticker] exception: {e}")
            time.sleep(max(args.loop_seconds, 30))
            continue

        if not ticker.get("Success", True):
            print(f"[ticker] error: {ticker.get('ErrMsg')}")
            time.sleep(args.loop_seconds)
            continue

        server_time_ms = int(ticker.get("ServerTime", 0))
        ticker_data = ticker.get("Data", {})

        if bot_start_server_ms is None:
            bot_start_server_ms = server_time_ms

        # store + log raw ticks
        store.update_from_ticker(server_time_ms, ticker_data)
        append_tickers_to_csv(ticker_csv, server_time_ms, ticker_data)

        minute_index = server_time_ms // 60000
        do_rebalance = (minute_index % rebalance_every == 0)

        pv = last_pv
        wallet: Dict[str, Any] = {}
        bal: Dict[str, Any] = {}
        actions: List[Any] = []

        # Determine if we need wallet (signed)
        need_wallet = (not args.public_only) and (do_rebalance or strat.in_position or strat.pending_entry_pair is not None or bootstrap_stage == 1 or (bootstrap_enabled and bootstrap_stage == 0))

        if args.public_only:
            pv = initial_usd
            targets, debug = strat.step(
                store=store,
                rules=rules,
                portfolio_value_usd=pv,
                wallet={},
                now_ts_ms=server_time_ms,
                do_rebalance=do_rebalance,
                public_only=True,
            )
        else:
            if need_wallet:
                try:
                    bal = client.get_balance()
                except Exception as e:
                    print(f"[balance] exception: {e}")
                    time.sleep(args.loop_seconds)
                    continue

                if not bal.get("Success", True):
                    print(f"[balance] error: {bal.get('ErrMsg')}")
                    time.sleep(args.loop_seconds)
                    continue

                wallet = bal.get("Wallet", {})
                pv = portfolio_value_usd(rules, store, bal)
            else:
                pv = last_pv

            # --- Strategy step ---
            targets, debug = strat.step(
                store=store,
                rules=rules,
                portfolio_value_usd=pv,
                wallet=wallet,
                now_ts_ms=server_time_ms,
                do_rebalance=do_rebalance,
                public_only=False,
            )

            # --- Bootstrap logic (only if no trades yet) ---
            uptime_min = int((server_time_ms - int(bot_start_server_ms)) // 60000) if bot_start_server_ms else 0

            # Trigger BUY on a rebalance minute after deadline if no strategy trades happened
            if bootstrap_enabled and bootstrap_stage == 0 and (not have_any_trade_action) and (uptime_min >= int(args.bootstrap_deadline_min)) and do_rebalance and (not strat.in_position):
                pick = pick_bootstrap_pair(ticker_data, rules)
                if pick:
                    d = ticker_data.get(pick, {})
                    ask = float(d.get("MinAsk", 0.0))
                    if ask > 0:
                        qty = compute_bootstrap_qty(pick, ask, float(args.bootstrap_notional_usd), rules[pick])
                        if qty > 0:
                            # Override targets to force a single BUY
                            forced = {p: 0.0 for p in rules.keys()}
                            # target qty = existing qty + bootstrap qty; simplest is set target to bootstrap qty
                            forced[pick] = float(qty)

                            if exec_engine is not None:
                                actions = exec_engine.execute_cycle(forced, store, bal)

                            bootstrap_stage = 1
                            bootstrap_pair = pick
                            bootstrap_buy_minute = int(minute_index)
                            print(f"[bootstrap] BUY triggered pair={pick} qty={qty}")

            # Trigger SELL at least 1 minute later
            if bootstrap_enabled and bootstrap_stage == 1 and bootstrap_pair is not None and bootstrap_buy_minute is not None:
                if int(minute_index) >= int(bootstrap_buy_minute) + 1:
                    forced = {p: 0.0 for p in rules.keys()}  # target all cash -> sell holding
                    if exec_engine is not None:
                        actions = exec_engine.execute_cycle(forced, store, bal)

                    # mark done
                    try:
                        with open(bootstrap_done_path, "w", encoding="utf-8") as f:
                            json.dump({"pair": bootstrap_pair, "ts": server_time_ms}, f, ensure_ascii=False)
                    except Exception:
                        pass

                    bootstrap_stage = 2
                    print(f"[bootstrap] SELL triggered pair={bootstrap_pair} (done)")

            # Execute strategy actions when appropriate
            should_execute = do_rebalance or bool(debug.get("exit_signal", False))
            if exec_engine is not None and should_execute and need_wallet:
                # Only execute if bootstrap didn't already execute this loop
                if not actions:
                    actions = exec_engine.execute_cycle(targets, store, bal)

        if actions:
            have_any_trade_action = True

        last_pv = pv

        picked = debug.get("picked_pair")
        record = {
            "ts": server_time_ms,
            "portfolio_value": float(pv),
            "rebalance": bool(do_rebalance),
            "mode": "public-only" if args.public_only else ("dry-run" if args.dry_run else "live"),

            # strategy diagnostics
            "strategy": debug.get("strategy"),
            "state": debug.get("state"),
            "picked_pair": picked,
            "entry_signal": bool(debug.get("entry_signal", False)),
            "exit_signal": bool(debug.get("exit_signal", False)),
            "exit_reason": debug.get("exit_reason", ""),
            "market_r": debug.get("market_r"),
            "r1": debug.get("r1"),
            "r3": debug.get("r3"),
            "reason": str(debug.get("reason", "")),

            # bootstrap diagnostics
            "bootstrap_enabled": bool(bootstrap_enabled),
            "bootstrap_stage": int(bootstrap_stage),
            "bootstrap_pair": bootstrap_pair,
            "uptime_min": int((server_time_ms - int(bot_start_server_ms)) // 60000) if bot_start_server_ms else 0,

            "actions": int(len(actions)),
        }
        append_decision_jsonl(decision_jsonl, record)

        print(
            f"[loop] ts={server_time_ms} PV={pv:.2f} rebalance={do_rebalance} "
            f"state={record['state']} picked={picked} exit={record['exit_signal']} "
            f"actions={len(actions)} reason={record['reason']}"
        )

        elapsed = time.time() - loop_start
        time.sleep(max(0.0, args.loop_seconds - elapsed))


if __name__ == "__main__":
    main()