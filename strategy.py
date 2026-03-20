# strategy.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List
import numpy as np

from rules import PairRule
from store import MarketStore


@dataclass
class RareContrarianConfig:
    # Entry
    lookback_minutes: int = 30
    entry_rL: float = -0.03              # rL <= -3%
    market_r_min: float = -0.0025        # median rL across base_ok >= -0.25%
    confirm_r1_min: float = 0.001        # last 1-min return >= +0.2%
    confirm_r3_min: float = -0.005       # last 3-min return >= -0.5% (loose)
    entry_ch24_max: Optional[float] = None  # None = no 24h filter

    # Risk / exits
    invest_frac: float = 0.05            # invest 5% of PV
    take_profit: float = 0.006           # +0.6%
    stop_loss: float = -0.008             # -0.8%
    max_hold_minutes: int = 30           # time stop

    # Filters
    min_unit_value: float = 8_000_000.0
    max_spread_pct: float = 0.0031

    eps: float = 1e-12


class RareEventContrarian:
    """
    Single-position, long-only rare-event contrarian.
    - Entry considered only on rebalance minutes.
    - Exit checked every minute.
    """

    def __init__(self, cfg: RareContrarianConfig):
        self.cfg = cfg

        # Position state (for public-only simulation and for exit logic)
        self.pos_pair: Optional[str] = None
        self.entry_price: Optional[float] = None
        self.entry_ts_ms: Optional[int] = None

        # Pending entry metadata (helps live mode approximate entry price/time)
        self.pending_entry_pair: Optional[str] = None
        self.pending_entry_price: Optional[float] = None
        self.pending_entry_ts_ms: Optional[int] = None

    @property
    def in_position(self) -> bool:
        return self.pos_pair is not None

    def _infer_wallet_holding(
        self,
        rules: Dict[str, PairRule],
        store: MarketStore,
        wallet: Dict[str, Any],
    ) -> Tuple[Optional[str], float]:
        """Pick the largest USD-value holding at bid among tradable coins."""
        if not wallet:
            return None, 0.0

        best_pair = None
        best_qty = 0.0
        best_val = 0.0

        for pair, rule in rules.items():
            if not getattr(rule, "can_trade", False):
                continue
            coin = getattr(rule, "coin", None)
            if not coin:
                continue
            w = wallet.get(coin, {})
            qty = float(w.get("Free", 0.0)) + float(w.get("Lock", 0.0))
            if qty <= 0:
                continue
            latest = store.latest(pair)
            if not latest:
                continue
            bid = float(latest.get("bid", 0.0))
            if bid <= 0:
                continue
            val = qty * bid
            if val > best_val:
                best_val = val
                best_pair = pair
                best_qty = qty

        return best_pair, float(best_qty)

    def _sync_from_wallet(
        self,
        held_pair: Optional[str],
        now_ts_ms: int,
        store: MarketStore,
    ) -> None:
        """Align internal state to wallet holding."""
        if held_pair is None:
            self.pos_pair = None
            self.entry_price = None
            self.entry_ts_ms = None
            self.pending_entry_pair = None
            self.pending_entry_price = None
            self.pending_entry_ts_ms = None
            return

        if self.pos_pair != held_pair:
            # new holding detected
            self.pos_pair = held_pair

            if self.pending_entry_pair == held_pair and self.pending_entry_ts_ms is not None:
                self.entry_ts_ms = self.pending_entry_ts_ms
                self.entry_price = self.pending_entry_price
            else:
                self.entry_ts_ms = now_ts_ms
                latest = store.latest(held_pair) or {}
                self.entry_price = float(latest.get("ask", latest.get("mid", 0.0))) or None

            self.pending_entry_pair = None
            self.pending_entry_price = None
            self.pending_entry_ts_ms = None

        if not self.entry_price or self.entry_price <= 0:
            latest = store.latest(held_pair) or {}
            px = float(latest.get("ask", latest.get("mid", 0.0)))
            self.entry_price = px if px > 0 else None

    def step(
        self,
        *,
        store: MarketStore,
        rules: Dict[str, PairRule],
        portfolio_value_usd: float,
        wallet: Dict[str, Any],
        now_ts_ms: int,
        do_rebalance: bool,
        public_only: bool,
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        cfg = self.cfg
        targets = {pair: 0.0 for pair in rules.keys()}

        debug: Dict[str, Any] = {
            "strategy": "rare_contrarian",
            "state": "FLAT",
            "position_pair": None,
            "entry_price": None,
            "entry_ts_ms": None,
            "entry_signal": False,
            "exit_signal": False,
            "exit_reason": "",
            "reason": "",
            "picked_pair": None,
            "picked_rL": None,
            "market_r": None,
            "r1": None,
            "r3": None,
        }

        # Determine holding
        if public_only:
            held_pair = self.pos_pair
        else:
            held_pair, _ = self._infer_wallet_holding(rules, store, wallet)
            self._sync_from_wallet(held_pair, now_ts_ms, store)

        # Exit check every minute if holding
        if self.pos_pair is not None:
            debug["state"] = "HOLDING"
            debug["position_pair"] = self.pos_pair
            debug["entry_price"] = self.entry_price
            debug["entry_ts_ms"] = self.entry_ts_ms

            latest = store.latest(self.pos_pair) or {}
            bid = float(latest.get("bid", 0.0))
            if bid <= 0 or not self.entry_price or not self.entry_ts_ms:
                debug["reason"] = "holding (missing bid/entry)"
                return targets, debug

            pnl = (bid / float(self.entry_price)) - 1.0
            held_min = int((now_ts_ms - int(self.entry_ts_ms)) // 60000)

            exit_reason = None
            if pnl >= float(cfg.take_profit):
                exit_reason = "take_profit"
            if pnl <= float(cfg.stop_loss):
                exit_reason = "stop_loss"
            if held_min >= int(cfg.max_hold_minutes):
                exit_reason = "time_stop" if exit_reason is None else (exit_reason + "+time_stop")

            if exit_reason:
                debug["exit_signal"] = True
                debug["exit_reason"] = exit_reason
                debug["reason"] = f"exit {exit_reason} pnl={pnl:.4f} held_min={held_min}"
                if public_only:
                    # simulate exit
                    self.pos_pair = None
                    self.entry_price = None
                    self.entry_ts_ms = None
                return targets, debug

            debug["reason"] = f"holding pnl={pnl:.4f} held_min={held_min}"
            return targets, debug

        # Flat: entry only on rebalance
        if not do_rebalance:
            debug["reason"] = "flat (not rebalance minute)"
            return targets, debug

        L = int(cfg.lookback_minutes)
        need = max(L + 1, 4)

        base_ok: List[str] = []
        rL_map: Dict[str, float] = {}

        for pair, rule in rules.items():
            if not getattr(rule, "can_trade", False):
                continue
            latest = store.latest(pair)
            if not latest:
                continue

            unitv = float(latest.get("unit_value", 0.0))
            spr = float(latest.get("spread_pct", 1e9))
            if unitv < float(cfg.min_unit_value):
                continue
            if spr > float(cfg.max_spread_pct):
                continue
            if not store.has_history(pair, need):
                continue

            mids = np.asarray(store.get_field(pair, "mid", need), dtype=float)
            if len(mids) < need or not np.isfinite(mids).all():
                continue
            if mids[-1] <= 0 or mids[-1 - L] <= 0:
                continue

            rL = (mids[-1] / mids[-1 - L]) - 1.0
            base_ok.append(pair)
            rL_map[pair] = float(rL)

        if not base_ok:
            debug["reason"] = "no base_ok pairs (filters/history)"
            return targets, debug

        market_r = float(np.median([rL_map[p] for p in base_ok]))
        debug["market_r"] = market_r
        if market_r < float(cfg.market_r_min):
            debug["reason"] = f"market gate: median rL {market_r:.6f} < {cfg.market_r_min}"
            return targets, debug

        candidates: List[Tuple[str, float, float, float]] = []
        for pair in base_ok:
            rL = rL_map[pair]
            if rL > float(cfg.entry_rL):
                continue

            mids4 = np.asarray(store.get_field(pair, "mid", 4), dtype=float)
            if len(mids4) < 4 or not np.isfinite(mids4).all():
                continue
            if mids4[-1] <= 0 or mids4[-2] <= 0 or mids4[-4] <= 0:
                continue

            r1 = (mids4[-1] / mids4[-2]) - 1.0
            r3 = (mids4[-1] / mids4[-4]) - 1.0
            if r1 < float(cfg.confirm_r1_min):
                continue
            if r3 < float(cfg.confirm_r3_min):
                continue

            if cfg.entry_ch24_max is not None:
                ch24 = float((store.latest(pair) or {}).get("change_24h", 0.0))
                if ch24 > float(cfg.entry_ch24_max):
                    continue

            candidates.append((pair, float(rL), float(r1), float(r3)))

        if not candidates:
            debug["reason"] = "no candidates after entry+confirm filters"
            return targets, debug

        candidates.sort(key=lambda x: x[1])  # most oversold
        pick, pick_rL, pick_r1, pick_r3 = candidates[0]

        latest = store.latest(pick) or {}
        mid_now = float(latest.get("mid", 0.0))
        ask_now = float(latest.get("ask", mid_now))
        if mid_now <= 0:
            debug["reason"] = "picked but mid<=0"
            return targets, debug

        invest_usd = float(portfolio_value_usd) * float(cfg.invest_frac)
        qty = invest_usd / mid_now

        targets[pick] = float(qty)

        debug["entry_signal"] = True
        debug["picked_pair"] = pick
        debug["picked_rL"] = pick_rL
        debug["r1"] = pick_r1
        debug["r3"] = pick_r3
        debug["reason"] = "entry_signal"

        # remember pending entry meta
        self.pending_entry_pair = pick
        self.pending_entry_ts_ms = int(now_ts_ms)
        self.pending_entry_price = float(ask_now) if ask_now > 0 else float(mid_now)

        if public_only:
            # simulate immediate fill
            self.pos_pair = pick
            self.entry_ts_ms = int(now_ts_ms)
            self.entry_price = self.pending_entry_price
            self.pending_entry_pair = None
            self.pending_entry_ts_ms = None
            self.pending_entry_price = None

        return targets, debug