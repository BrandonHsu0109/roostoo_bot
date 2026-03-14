from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from typing import Dict, Any


@dataclass(frozen=True)
class PairRule:
    coin: str
    unit: str
    can_trade: bool
    price_precision: int
    amount_precision: int
    min_order_notional: float  # "MiniOrder" in USD notional


def parse_exchange_info(info_json: Dict[str, Any]) -> Dict[str, PairRule]:
    pairs = info_json.get("TradePairs", {})
    out: Dict[str, PairRule] = {}
    for pair, d in pairs.items():
        out[pair] = PairRule(
            coin=d["Coin"],
            unit=d["Unit"],
            can_trade=bool(d["CanTrade"]),
            price_precision=int(d["PricePrecision"]),
            amount_precision=int(d["AmountPrecision"]),
            min_order_notional=float(d["MiniOrder"]),
        )
    return out


def _round_down(value: float, decimals: int) -> float:
    """
    Round DOWN to a fixed number of decimal places (important for exchange rules).
    """
    q = Decimal("1").scaleb(-decimals)  # 10^-decimals
    d = Decimal(str(value)).quantize(q, rounding=ROUND_DOWN)
    return float(d)


def round_price(rule: PairRule, price: float) -> float:
    return _round_down(price, rule.price_precision)


def round_qty(rule: PairRule, qty: float) -> float:
    return _round_down(qty, rule.amount_precision)


def valid_min_order(rule: PairRule, price: float, qty: float) -> bool:
    return (price * qty) > rule.min_order_notional