# bot.py

from __future__ import annotations

import os
import time
import csv
import json
import argparse
import datetime as dt
from typing import Dict, Any, Optional

from roostoo_client import RoostooClient
from rules import parse_exchange_info, PairRule, round_qty, valid_min_order
from store import MarketStore
from strategy import RareEventContrarian, RareContrarianConfig
from execution import ExecutionEngine, ExecutionConfig


def portfolio_value_usd(rules: Dict[str, PairRule], store: MarketStore, balance_json: Dict[str, Any]) -> float:
    wallet = balance_json.get("Wallet", {})
    usd = float(wallet.get("USD", {}).get("Free", 0.0)) + float(wallet.get("USD", {}).get("Lock", 0.0))
    total = usd

    # Value coin holdings at bid (conservative liquidation)
    for pair, rule in rules.items():
        coin = rule.coin
        qty = float(wallet.get(coin, {}).get("Free", 0.0)) + float(wallet.get(coin, {}).get("Lock", 0.0))
        if qty <= 0:
            continue
        latest = store.latest(pair)
        if latest is None:
            continue
        bid = float(latest.get("bid", 0.0))
        if bid <= 0:
            continue
        total += qty * bid

    return total


def ensure_csv_header(path: str, header: list) -> None:
    if os.path.exists(path):
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)


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


def utc_day_key(ts_ms: int) -> str:
    """UTC calendar day key like '2026-03-21'."""
    return dt.datetime.utcfromtimestamp(ts_ms / 1000.0).strftime("%Y-%m-%d")


def utc_hour_minute(ts_ms: int) -> tuple[int, int]:
    d = dt.datetime.utcfromtimestamp(ts_ms / 1000.0)
    return d.hour, d.minute


def load_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_json_atomic(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def count_live_orders(actions: list[Any]) -> int:
    """Count successfully submitted live orders in actions list."""
    n = 0
    for a in actions:
        if not isinstance(a, dict):
            continue
        if a.get("dry_run"):
            continue
        if a.get("Success") is True and ("OrderDetail" in a):
            n += 1
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop-seconds", type=int, default=60)
    parser.add_argument("--history-minutes", type=int, default=720)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--public-only", action="store_true", help="Run without signed endpoints (no balance, no trading).")
    parser.add_argument("--logdir", type=str, default="logs")
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

    # Sync time
    offset = client.sync_time()
    print(f"[startup] time offset ms = {offset}")

    # Exchange rules
    info = client.get_exchange_info()
    rules = parse_exchange_info(info)
    tradable_pairs = [p for p, r in rules.items() if r.can_trade]
    print(f"[startup] tradable pairs: {len(tradable_pairs)}")

    # Store (in-memory history)
    store = MarketStore(maxlen=args.history_minutes)

    # Logs
    ticker_csv = os.path.join(args.logdir, "ticker_history.csv")
    ensure_csv_header(ticker_csv, ["ts", "pair", "bid", "ask", "last", "change_24h", "coin_value", "unit_value"])
    decision_jsonl = os.path.join(args.logdir, "decisions.jsonl")

    # Daily activity microtrade (for 'active day' requirement) 
    # If there were no trades on a UTC calendar day, the bot will perform one tiny BUY then SELL
    # near the end of the day to ensure at least 1 trade that day.
    daily_state_path = os.path.join(args.logdir, "daily_state.json")

    # Configure via environment variables (defaults are safe)
    daily_micro_usd = float(os.getenv("DAILY_MICRO_USD", "100"))            # notional per microtrade leg
    daily_micro_pair_pref = os.getenv("DAILY_MICRO_PAIR", "BTC/USD")        # preferred pair (falls back to most liquid)
    daily_micro_hour = int(os.getenv("DAILY_MICRO_HOUR", "23"))             # UTC hour
    daily_micro_minute = int(os.getenv("DAILY_MICRO_MINUTE", "50"))         # UTC minute
    daily_micro_window_min = int(os.getenv("DAILY_MICRO_WINDOW_MIN", "5"))  # trigger window length in minutes

    daily = {
        "day": None,              
        "trades_today": 0,        
        "activity_stage": "NONE", 
        "activity_pair": None,
        "activity_qty": 0.0,
        "last_update_ts": 0,
    }
    daily.update(load_json(daily_state_path))

    # Strategy
    cfg = RareContrarianConfig(
        lookback_minutes=30,
        entry_rL=-0.025,
        market_r_min=-0.004,
        confirm_r1_min=0.001,
        confirm_r3_min=-0.005,
        entry_ch24_max=None,
        invest_frac=0.12,
        take_profit=0.006,
        stop_loss=-0.008,
        max_hold_minutes=30,
        min_unit_value=8_000_000.0,
        max_spread_pct=0.0031,
        cooldown_after_stop_mins=30,
    )
    strat = RareEventContrarian(cfg)

    exec_engine: Optional[ExecutionEngine] = None
    if not args.public_only:
        exec_engine = ExecutionEngine(
            client=client,
            rules=rules,
            cfg=ExecutionConfig(
                min_trade_usd=50.0,
                max_orders_per_cycle=1,
                dry_run=args.dry_run,
                use_market_only=True,
            )
        )

    # Rebalance cadence for entry decisions (strategy exits run every minute)
    rebalance_every = 1

    # Time resync policy
    resync_every = 10 * 60
    last_resync = time.time()

    initial_usd = float(info.get("InitialWallet", {"USD": 50000}).get("USD", 50000))
    last_pv = initial_usd

    print("[startup] running loop... (Ctrl+C to stop)")

    while True:
        loop_start = time.time()

        # Resync time occasionally
        if loop_start - last_resync >= resync_every:
            try:
                offset = client.sync_time()
                print(f"[time] resynced offset ms = {offset}")
            except Exception as e:
                print(f"[time] resync failed: {e}")
            last_resync = loop_start

        # Pull tickers (all pairs)
        ticker = client.get_ticker(pair=None)
        if not ticker.get("Success", True):
            err = ticker.get("ErrMsg")
            print(f"[ticker] error: {err}")
            time.sleep(args.loop_seconds)
            continue

        server_time_ms = int(ticker.get("ServerTime", 0))
        ticker_data = ticker.get("Data", {})

        # Daily state rollover (UTC day)
        now_day = utc_day_key(server_time_ms)
        if daily.get("day") != now_day:
            daily["day"] = now_day
            daily["trades_today"] = 0
            daily["activity_stage"] = "NONE"
            daily["activity_pair"] = None
            daily["activity_qty"] = 0.0
            daily["last_update_ts"] = server_time_ms

        # Update store + log raw tickers
        store.update_from_ticker(server_time_ms, ticker_data)
        append_tickers_to_csv(ticker_csv, server_time_ms, ticker_data)

        minute_index = server_time_ms // 60000
        do_rebalance = (minute_index % rebalance_every == 0)

        # Daily activity microtrade trigger window (UTC minutes)
        h, m = utc_hour_minute(server_time_ms)
        now_min = h * 60 + m
        deadline_min = daily_micro_hour * 60 + daily_micro_minute
        in_activity_window = (deadline_min <= now_min < (deadline_min + daily_micro_window_min))

        actions: list[Any] = []
        pv = last_pv

        # Decide whether we need a fresh wallet snapshot (signed endpoint)
        need_wallet = (not args.public_only) and (
            do_rebalance
            or strat.in_position
            or strat.pending_entry_pair is not None
            or ((daily.get("activity_stage") == "NEED_SELL") and (not args.dry_run))
            or (in_activity_window and (daily.get("trades_today", 0) == 0) and (not args.dry_run))
        )

        wallet: Dict[str, Any] = {}
        bal: Dict[str, Any] = {}

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
                    err = bal.get("ErrMsg")
                    print(f"[balance] error: {err}")
                    time.sleep(args.loop_seconds)
                    continue

                wallet = bal.get("Wallet", {})
                pv = portfolio_value_usd(rules, store, bal)
            else:
                pv = last_pv
                wallet = {}

            targets, debug = strat.step(
                store=store,
                rules=rules,
                portfolio_value_usd=pv,
                wallet=wallet,
                now_ts_ms=server_time_ms,
                do_rebalance=do_rebalance,
                public_only=False,
            )

            # Only try to execute if entry_signal or exit_signal (avoid trading every rebalance minute)
            should_execute = bool(debug.get("entry_signal", False)) or bool(debug.get("exit_signal", False))
            if exec_engine is not None and should_execute and need_wallet:
                actions = exec_engine.execute_cycle(targets, store, bal)

        # Update daily trade count from strategy execution
        if (not args.public_only) and (not args.dry_run):
            daily["trades_today"] = int(daily.get("trades_today", 0)) + count_live_orders(actions)
            daily["last_update_ts"] = server_time_ms

        # Daily activity microtrade
        # If there have been NO trades today and we are in the trigger window, do a tiny BUY then next loop SELL.
        activity_action = ""

        if (not args.public_only) and (not args.dry_run) and need_wallet:
            stage = str(daily.get("activity_stage", "NONE"))

            # (1) If we previously bought, try to sell ASAP to avoid carrying risk.
            if stage == "NEED_SELL" and daily.get("activity_pair"):
                apair = str(daily["activity_pair"])
                rule = rules.get(apair)
                latest = store.latest(apair) if rule else None
                coin = rule.coin if rule else None

                have_qty = 0.0
                if coin:
                    have_qty = float(wallet.get(coin, {}).get("Free", 0.0)) + float(wallet.get(coin, {}).get("Lock", 0.0))

                sell_qty_raw = min(float(daily.get("activity_qty", 0.0) or 0.0), have_qty)

                if latest and sell_qty_raw > 0:
                    bid = float(latest.get("bid", 0.0))
                    sell_qty = round_qty(rule, sell_qty_raw)
                    if sell_qty > 0 and bid > 0 and valid_min_order(rule, bid, sell_qty):
                        try:
                            resp = client.place_order(pair=apair, side="SELL", quantity=str(sell_qty), order_type="MARKET")
                            actions.append(resp)
                            if isinstance(resp, dict) and resp.get("Success") is True:
                                daily["trades_today"] = int(daily.get("trades_today", 0)) + 1
                                daily["activity_stage"] = "NONE"
                                daily["activity_pair"] = None
                                daily["activity_qty"] = 0.0
                            activity_action = f"daily_micro SELL {apair} qty={sell_qty}"
                        except Exception as e:
                            activity_action = f"daily_micro SELL error: {e}"

            # (2) If no trades today and within window, do the BUY leg (counts as today's trade).
            elif stage == "NONE":
                if in_activity_window and int(daily.get("trades_today", 0)) == 0 and (not strat.in_position):
                    # Choose preferred pair if tradable; else fall back to most liquid by UnitTradeValue.
                    apair = None
                    if daily_micro_pair_pref in rules and rules[daily_micro_pair_pref].can_trade:
                        apair = daily_micro_pair_pref
                    else:
                        best_pair = None
                        best_uv = -1.0
                        for p, d in ticker_data.items():
                            rr = rules.get(p)
                            if rr is None or (not rr.can_trade):
                                continue
                            try:
                                uv = float(d.get("UnitTradeValue") or 0.0)
                            except Exception:
                                uv = 0.0
                            if uv > best_uv:
                                best_uv = uv
                                best_pair = p
                        apair = best_pair

                    if apair and apair in rules:
                        rule = rules[apair]
                        latest = store.latest(apair)
                        usd_free = float(wallet.get("USD", {}).get("Free", 0.0))

                        if latest and usd_free > 0:
                            ask = float(latest.get("ask", 0.0))
                            # Ensure notional meets exchange minimum; keep it small.
                            min_notional = max(50.0, float(rule.min_order_notional))
                            usd_notional = float(max(daily_micro_usd, min_notional))
                            usd_notional = float(min(usd_notional, usd_free * 0.95))

                            if ask > 0 and usd_notional > 0:
                                buy_qty_raw = usd_notional / ask
                                buy_qty = round_qty(rule, buy_qty_raw)

                                if buy_qty > 0 and valid_min_order(rule, ask, buy_qty):
                                    try:
                                        resp = client.place_order(pair=apair, side="BUY", quantity=str(buy_qty), order_type="MARKET")
                                        actions.append(resp)
                                        if isinstance(resp, dict) and resp.get("Success") is True:
                                            daily["trades_today"] = int(daily.get("trades_today", 0)) + 1
                                            daily["activity_stage"] = "NEED_SELL"
                                            daily["activity_pair"] = apair
                                            daily["activity_qty"] = float(buy_qty)
                                        activity_action = f"daily_micro BUY {apair} qty={buy_qty} notional~{usd_notional:.2f}"
                                    except Exception as e:
                                        activity_action = f"daily_micro BUY error: {e}"

        # Persist daily state every loop (safe across restarts)
        try:
            save_json_atomic(daily_state_path, daily)
        except Exception:
            pass

        last_pv = pv

        # Pick selected pairs as a list for compatibility with former JSON viewer
        picked = debug.get("picked_pair")
        selected_pairs = [picked] if picked else []

        action_summary = str(actions[0])[:300] if actions else ""

        record = {
            "ts": server_time_ms,
            "portfolio_value": float(pv),
            "rebalance": bool(do_rebalance),
            "selected_pairs": selected_pairs,
            "reason": str(debug.get("reason", "")),
            "mode": "public-only" if args.public_only else ("dry-run" if args.dry_run else "live"),
            "day": daily.get("day"),
            "trades_today": int(daily.get("trades_today", 0)),
            "activity_stage": daily.get("activity_stage"),
            "activity_action": activity_action,

            # Extra diagnostics (new)
            "strategy": debug.get("strategy"),
            "state": debug.get("state"),
            "position_pair": debug.get("position_pair"),
            "entry_price": debug.get("entry_price"),
            "entry_ts_ms": debug.get("entry_ts_ms"),
            "entry_signal": bool(debug.get("entry_signal", False)),
            "exit_signal": bool(debug.get("exit_signal", False)),
            "exit_reason": debug.get("exit_reason", ""),
            "picked_pair": picked,
            "picked_rL": debug.get("picked_rL"),
            "market_r": debug.get("market_r"),
            "r1": debug.get("r1"),
            "r3": debug.get("r3"),
            "action_summary": action_summary,
        }
        append_decision_jsonl(decision_jsonl, record)

        print(
            f"[loop] ts={server_time_ms} PV={pv:.2f} rebalance={do_rebalance} "
            f"state={record['state']} picked={picked} exit={record['exit_signal']} actions={len(actions)} "
            f"reason={record['reason']}"
        )

        elapsed = time.time() - loop_start
        time.sleep(max(0.0, args.loop_seconds - elapsed))


if __name__ == "__main__":
    main()