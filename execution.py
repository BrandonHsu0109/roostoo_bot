from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

from roostoo_client import RoostooClient
from rules import PairRule, round_qty, valid_min_order
from store import MarketStore


@dataclass
class ExecutionConfig:
    min_trade_usd: float = 50.0          # extra safety threshold in USD
    max_orders_per_cycle: int = 1       # keep simple (competition suggests ~1/min)
    min_delta_usd: float = 200.0
    dry_run: bool = True                # start in dry-run
    use_market_only: bool = True        # avoid pending orders initially


class ExecutionEngine:
    def __init__(self, client: RoostooClient, rules: Dict[str, PairRule], cfg: ExecutionConfig):
        self.client = client
        self.rules = rules
        self.cfg = cfg

    @staticmethod
    def _wallet_qty(wallet: Dict[str, Any], coin: str) -> float:
        # wallet structure: Wallet -> coin -> {Free, Lock}
        if coin not in wallet:
            return 0.0
        d = wallet[coin]
        return float(d.get("Free", 0.0)) + float(d.get("Lock", 0.0))

    def build_orders(
        self,
        targets_qty_by_pair: Dict[str, float],
        store: MarketStore,
        balance_json: Dict[str, Any],
    ) -> List[Tuple[str, str, float, float]]:
        """
        Returns list of (pair, side, qty, est_price_for_checks)
        side: BUY/SELL
        qty: coin amount (float, not yet rounded)
        est_price_for_checks: use ask for BUY and bid for SELL
        """
        wallet = balance_json.get("Wallet", {})
        orders: List[Tuple[str, str, float, float]] = []

        for pair, target_qty in targets_qty_by_pair.items():
            rule = self.rules.get(pair)
            if rule is None or not rule.can_trade:
                continue

            latest = store.latest(pair)
            if latest is None:
                continue

            coin = rule.coin
            current_qty = self._wallet_qty(wallet, coin)

            delta = float(target_qty) - float(current_qty)
            if abs(delta) <= 0.0:
                continue

            if delta > 0:
                side = "BUY"
                est_price = float(latest["ask"])  # conservative cost
                qty = delta
            else:
                side = "SELL"
                est_price = float(latest["bid"])  # conservative revenue
                qty = -delta

            # notional checks
            notional = est_price * qty
            if notional < self.cfg.min_delta_usd:
                continue
            min_notional = max(self.cfg.min_trade_usd, rule.min_order_notional)
            if notional <= min_notional:
                continue

            orders.append((pair, side, qty, est_price))

        # Prioritize biggest notional differences
        orders.sort(key=lambda x: x[2] * x[3], reverse=True)
        return orders[: self.cfg.max_orders_per_cycle]

    def execute_cycle(
        self,
        targets_qty_by_pair: Dict[str, float],
        store: MarketStore,
        balance_json: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Places orders (or prints them if dry_run).
        Returns list of order responses or simulated actions.
        """
        actions: List[Dict[str, Any]] = []
        orders = self.build_orders(targets_qty_by_pair, store, balance_json)

        for pair, side, qty_raw, est_price in orders:
            rule = self.rules[pair]

            qty_rounded = round_qty(rule, qty_raw)
            if qty_rounded <= 0:
                continue

            # Check MiniOrder using est_price (ask for buy, bid for sell)
            if not valid_min_order(rule, est_price, qty_rounded):
                continue

            if self.cfg.dry_run:
                actions.append({
                    "dry_run": True,
                    "pair": pair,
                    "side": side,
                    "qty": qty_rounded,
                    "type": "MARKET",
                    "est_price": est_price
                })
            else:
                resp = self.client.place_order(
                    pair=pair,
                    side=side,
                    quantity=str(qty_rounded),
                    order_type="MARKET"
                )
                actions.append(resp)

        return actions