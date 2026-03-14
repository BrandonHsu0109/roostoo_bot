from collections import deque
from typing import Any, Deque, Dict, Optional, List


class MarketStore:
    """
    Stores per-pair history you build yourself by polling /v3/ticker.
    Each record contains: ts, bid, ask, mid, spread_pct, last, unit_value, coin_value, change_24h
    """
    def __init__(self, maxlen: int = 720):
        self.maxlen = maxlen
        self.data: Dict[str, Deque[Dict[str, Any]]] = {}

    def update_from_ticker(self, server_time_ms: int, ticker_data: Dict[str, Any]) -> None:
        for pair, d in ticker_data.items():
            bid = float(d["MaxBid"])
            ask = float(d["MinAsk"])
            last = float(d["LastPrice"])
            mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else last
            spread = max(0.0, ask - bid)
            spread_pct = (spread / mid) if mid > 0 else 0.0

            rec = {
                "ts": int(server_time_ms),
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "spread_pct": spread_pct,
                "last": last,
                "change_24h": float(d.get("Change", 0.0)),
                "coin_value": float(d.get("CoinTradeValue", 0.0)),
                "unit_value": float(d.get("UnitTradeValue", 0.0)),
            }

            if pair not in self.data:
                self.data[pair] = deque(maxlen=self.maxlen)
            self.data[pair].append(rec)

    def latest(self, pair: str) -> Optional[Dict[str, Any]]:
        dq = self.data.get(pair)
        if not dq:
            return None
        return dq[-1]

    def get_field(self, pair: str, field: str, n: int) -> List[float]:
        dq = self.data.get(pair)
        if not dq:
            return []
        take = list(dq)[-n:]
        return [float(x[field]) for x in take if field in x]

    def has_history(self, pair: str, n: int) -> bool:
        dq = self.data.get(pair)
        return dq is not None and len(dq) >= n