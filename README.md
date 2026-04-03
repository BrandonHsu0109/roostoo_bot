# roostoo_bot
### Quant trading bot for SG vs HK Quant Trading Hackathon

This repository contains a fully autonomous **spot-only** crypto trading bot for the Roostoo exchange API. It runs continuously (e.g., on AWS EC2 for this hackathon), fetches market data, generates signals, and places trades via signed endpoints.

---

## 1. Project Overview

### Strategy description
**Rare-Event Contrarian (“Buy the Dip with Confirmation”)**  
The bot looks for coins that have experienced an unusually large short-term drop, then enters a **long** position only if there are signs of stabilization and the broader market is not in a synchronized selloff.

### Core idea of the strategy
- **Mean reversion / contrarian** strategy (rare dip → rebound)
- A **confirmation gate** for reducing “catching a falling knife”
- Strict exits (TP/SL/time-stop) for risk control

### Key features
- **Spot-only, long-only** trading (no shorting, no leverage due to competition rules)
- **Single position at a time** (simpler risk control and evaluation, reduce trade frequency for better quality trades)
- **Per-minute decision loop** (not HFT; respects rate limits)
- **Market regime filter** (avoid buying dips when the whole market is falling)
- **Take-profit / stop-loss / time-stop** exits
- Robust API signing + time synchronization for signed endpoints
- Structured logging for auditability:
  - `logs/decisions.jsonl` (one JSON record per loop)
  - `logs/ticker_history.csv` (minute snapshots of market tickers)

---

## 2. Architecture

### Components
- **Data Module (`store.py`)**
    - Maintains rolling market history per pair (mid/bid/ask/spread/liquidity)
    - Updates every minute from `GET /v3/ticker`
- **Strategy Module (`strategy.py`)**
    - Implements entry/exit logic and pair selection rules
- **Execution Module (`execution.py`)**
    - Converts signals into orders (market orders)
    - Applies exchange precision rules and basic safety checks
- **API Module (`roostoo_client.py`)**
    - Time sync with `v3/serverTime`
    - Signed requests with HMAC SHA256 for protected endpoints
    - Normalizes balance payload so spot wallet is consistently readable
- **Logging Module (`bot.py`)**
    - Console status lines for monitoring
    - Structured JSONL and CSV outputs in `logs/`

### Tech stack used
- **Python 3**
- Libraries:
    - `requests`(HTTP client)
    - `numpy`(lightweight numeric computations)
    - `pandas`(optional, used only for offline analysis scripts if needed)
- Deployment:
    - AWS EC2 (Amazon Linux)
    - run via `tmux`

---

## 3. Strategy Explanation

### **Entry conditions (evaluated every minute while FLAT)**

The bot only considers entry when it is **not holding** any position.

**Step1 - Base filters**
For each tradable pair:
- **Spread filter:** `(ask - bid) / mid <= max_spread`
- **Liquidity filter:** `UnitTradeValue >= min_unit`
- **History requirement:** enough minute-history exists to compute returns

**Step2 - Rare dip (oversold)**
Compute lookback return (default lookaback ~30 minutes):
- `rL = mid_now / mid_{lookback} - 1`
Requirement:
- `rL <= entry_rL` (e.g., -2.5% to -3.0%)

**Step3 - Market regime filter (avoid systemic selloff)**
Compute `market_r` as the median of `rL` across all base-filter-passing pairs:
- `market_r = median(rL across base_ok pairs)`
Requirement:
- `market_r >= market_r_min` (e.g., -0.25%)

**Step4 - Stabilization confirmation**
Compute short-horizon returns:
- `r1 = mid_now / mid_{1 min ago} - 1`
- `r3 = mid_now / mid_{3 min ago} - 1`
Requirement:
- `r1 >= confirm_r1_min` (e.g., +0.1%)
- `r3 >= confirm_r3_min` (small guard against continued damping)

**Step5 - Selection**
Among cnadidates that pass all filters, select the most oversold:
- pisk the pair with the smallest `rL`

### **Exit conditions (checked every minute while HOLDING)**
Define:
- `pnl = bid_now / entry_price - 1`
Exit if any trigger fires:
- **Take Profit (TP):** `pnl >= take_profit` (e.g., +0.6%)
- **Stop Loss (SL):** `pnl <= stop_loss` (e.g., -0.8%)
- **Time Stop:** held minutes `>= max_hold_minutes` (e.g., 30)

### **Risk management rules**
- **Single postion at a time**
- **Hard TP/SL/time-stop to prevent runaway losses or indefinite holding**
- **Market regime filter reduces exposure during broad selloffs**

### **Position sizing logic**
- Target notional:
    - `invest_usd = PV * invest_frac`
- Quantity:
    - `qty = invest_usd / mid_price`

Note: Trade sizes scale automatically according to portfolio value `PV`. Adjust `invest_frac` to control risk.

### **Assumptions**
- Spot-only trading (no shorting)
- Market orders fill reliably in the provided environment
- Signed endpoints require timestamps within server tolerance; client calls `sync_time()` periodically
- Balance payload may use `SpotWallet`; client normalizes it to `Wallet` for consistent PV calculation

---

## 4. Setup instructions & How to run bot

### 4.1 Install dependencies (recommended: virtual environment)
```bash
    python -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
```

### 4.2 Set API keys
Set environment variables:

**Windows (Command Prompt)**
```bat
    set "ROOSTOO_API_KEY=YOUR_KEY"
    set "ROOSTOO_SECRET_KEY=YOUR_SECRET"
    set "ROOSTOO_BASE_URL=https://mock-api.roostoo.com"
```

**Linux/EC2 (recommended)**
Create `~/.roostoo-bot.env`:
```bash
    ROOSTOO_API_KEY=YOUR_KEY
    ROOSTOO_SECRET_KEY=YOUR_SECRET
    ROOSTOO_BASE_URL=https://mock-api.roostoo.com
```
Load it:
```bash
    set -a; . ~/.roostoo-bot.env; set +a
```

### 4.3 Signed endpoint test (optional but recommended before running)
```bash
    python -c "import os; from roostoo_client import RoostooClient; c=RoostooClient(os.getenv('ROOSTOO_BASE_URL','https://mock-api.roostoo.com'), os.getenv('ROOSTOO_API_KEY',''), os.getenv('ROOSTOO_SECRET_KEY','')); print('offset_ms', c.sync_time()); print(c.get_balance())"
```

### 4.4 Run modes
- **Public only (signal generation only, no real trades)**
```bash
    python bot.py --public-only --dry-run --loop-seconds 60
```
- **Dry run (keys required, no real orders placed)**
```bash
    python bot.py --dry-run --loop-seconds 60
```
- **Live run (keys required, real orders placed)**
```bash
    python bot.py --loop-seconds 60
```

### 4.5 Logs
The bot writes logs to `logs/`:
- `logs/decisions.jsonl` - structured decision records
- `logs/ticker_history.csv` - market snapshots

### 4.6 EC2 (Amazon Linux) quickstart with tmux
1. Install tmux:
```bash
    sudo yum install -y tmux   # Amazon Linux 2
    # or
    sudo dnf install -y tmux   # Amazon Linux 2023
```
2. Start a session:
```bash
    tmux new -s session_name
```
3. Load keys and run:
```bash
    set -a; . ~/.roostoo-bot.env; set +a
    cd ~/roostoo_bot
    source .venv/bin/activate
    python bot.py --loop-seconds 60
```
detach session:
- `Ctrl + B` and then `D`

reattach:
```bash
    tmux attach -t session_name
```
