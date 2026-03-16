import time
import hmac
import hashlib
import requests
from typing import Any, Dict, Optional, Tuple


class RoostooClient:
    """
    Minimal, reliable Roostoo API client.

    Key points from your README:
    - SIGNED endpoints require:
      - headers: RST-API-KEY, MSG-SIGNATURE
      - timestamp param (13-digit ms)
      - signature = HMAC_SHA256(secret, totalParams)
      - totalParams is sorted key=value joined by &
    - POST uses Content-Type = application/x-www-form-urlencoded
    """

    def __init__(self, base_url: str, api_key: str, secret_key: str, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.secret_key = secret_key.encode("utf-8")
        self.timeout = timeout

        # server_time - local_time (ms). Use to generate timestamps that match server.
        self._time_offset_ms = 0

        self.session = requests.Session()

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    def sync_time(self) -> int:
        """Sync local clock to server clock (offset in ms)."""
        data = self.get_server_time()
        server_ms = int(data["ServerTime"])
        local_ms = self._now_ms()
        self._time_offset_ms = server_ms - local_ms
        return self._time_offset_ms

    def _timestamp(self) -> str:
        return str(self._now_ms() + self._time_offset_ms)

    def _build_total_params(self, payload: Dict[str, Any]) -> str:
        # Make sure everything is stringified consistently
        sorted_keys = sorted(payload.keys())
        parts = [f"{k}={payload[k]}" for k in sorted_keys]
        return "&".join(parts)

    def _sign(self, payload: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, Any], str]:
        payload = dict(payload)  # copy
        payload["timestamp"] = self._timestamp()
        total_params = self._build_total_params(payload)

        signature = hmac.new(
            self.secret_key,
            total_params.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

        headers = {
            "RST-API-KEY": self.api_key,
            "MSG-SIGNATURE": signature,
        }
        return headers, payload, total_params

    def _request_with_retry(self, method: str, url: str, *,
                            headers: Optional[Dict[str, str]] = None,
                            params: Optional[Dict[str, Any]] = None,
                            data: Optional[str] = None,
                            max_retries: int = 3) -> Dict[str, Any]:
        last_err = None
        for attempt in range(max_retries):
            try:
                resp = self.session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    params=params,
                    data=data,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                last_err = e
                # simple backoff
                time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"HTTP request failed after retries: {method} {url} err={last_err}")

    # -------------------------
    # Public endpoints
    # -------------------------
    def get_server_time(self) -> Dict[str, Any]:
        url = f"{self.base_url}/v3/serverTime"
        return self._request_with_retry("GET", url)

    def get_exchange_info(self) -> Dict[str, Any]:
        url = f"{self.base_url}/v3/exchangeInfo"
        return self._request_with_retry("GET", url)

    def get_ticker(self, pair: Optional[str] = None) -> Dict[str, Any]:
        url = f"{self.base_url}/v3/ticker"
        params = {"timestamp": self._timestamp()}
        if pair:
            params["pair"] = pair
        return self._request_with_retry("GET", url, params=params)

    # -------------------------
    # Signed endpoints
    # (We build the query string ourselves to match signing)
    # -------------------------
    def get_balance(self) -> Dict[str, Any]:
        headers, payload, total_params = self._sign({})
        url = f"{self.base_url}/v3/balance?{total_params}"
        data = self._request_with_retry("GET", url, headers=headers)

        if isinstance(data, dict):
            # Prefer SpotWallet for spot trading
            if "Wallet" not in data:
                spot = data.get("SpotWallet")
                if isinstance(spot, dict):
                    data["Wallet"] = spot
                else:
                    # Fallback if API ever returns Wallet under a different key
                    margin = data.get("MarginWallet")
                    if isinstance(margin, dict):
                        data["Wallet"] = margin

        return data

    def get_pending_count(self) -> Dict[str, Any]:
        headers, payload, total_params = self._sign({})
        url = f"{self.base_url}/v3/pending_count?{total_params}"
        return self._request_with_retry("GET", url, headers=headers)

    def place_order(self, pair: str, side: str, quantity: str,
                    order_type: str = "MARKET",
                    price: Optional[str] = None) -> Dict[str, Any]:
        """
        side: BUY or SELL
        order_type: MARKET or LIMIT
        quantity: coin amount as string
        price: required for LIMIT
        """
        url = f"{self.base_url}/v3/place_order"
        payload: Dict[str, Any] = {
            "pair": pair,
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": str(quantity),
        }
        if payload["type"] == "LIMIT":
            if price is None:
                raise ValueError("LIMIT orders require price")
            payload["price"] = str(price)

        headers, signed_payload, total_params = self._sign(payload)
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        return self._request_with_retry("POST", url, headers=headers, data=total_params)

    def query_order(self, order_id: Optional[int] = None,
                    pair: Optional[str] = None,
                    pending_only: Optional[bool] = None,
                    limit: Optional[int] = None,
                    offset: Optional[int] = None) -> Dict[str, Any]:
        url = f"{self.base_url}/v3/query_order"
        payload: Dict[str, Any] = {}

        if order_id is not None:
            payload["order_id"] = str(order_id)
        else:
            if pair is not None:
                payload["pair"] = pair
            if pending_only is not None:
                payload["pending_only"] = "TRUE" if pending_only else "FALSE"
            if limit is not None:
                payload["limit"] = str(limit)
            if offset is not None:
                payload["offset"] = str(offset)

        headers, signed_payload, total_params = self._sign(payload)
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        return self._request_with_retry("POST", url, headers=headers, data=total_params)

    def cancel_order(self, order_id: Optional[int] = None, pair: Optional[str] = None) -> Dict[str, Any]:
        url = f"{self.base_url}/v3/cancel_order"
        payload: Dict[str, Any] = {}

        if order_id is not None:
            payload["order_id"] = str(order_id)
        elif pair is not None:
            payload["pair"] = pair
        # if neither provided => cancel all pending

        headers, signed_payload, total_params = self._sign(payload)
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        return self._request_with_retry("POST", url, headers=headers, data=total_params)