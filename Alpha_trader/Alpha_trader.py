#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alpha Trader — Multi-Source (Deribit Perps, Coinbase Spot, Yahoo CME, Yahoo Crypto)
EDU ONLY. Paper trade first.

Sources:
- deribit_perp   : BTC/ETH perpetuals (public, close to futures you’ll trade)
- coinbase_spot  : Coinbase Exchange spot (public) — exact Coinbase spot price
- yahoo_cme      : Regulated futures via Yahoo Finance (e.g., MET=F, MBT=F, GC=F)
- yahoo_crypto   : Yahoo crypto spot (BTC-USD, ETH-USD, AVAX-USD, SOL-USD)

Strategy:
- Multi-timeframe alignment: Daily + 4h + 1h
- Regime filter (Daily EMA20/50)
- Session VWAP, VRVP (POC/HVA), RSI window, ATR stop, 2R take-profit
- Volatility & optional time-of-day filters
- Backtest (bar-by-bar TP/SL), ledger, P&L
"""

import os, csv, uuid, time, traceback, json, requests
from datetime import datetime, timezone, time as dtime
from dateutil import tz
import numpy as np
import pandas as pd

# ---------------- CONFIG ----------------
LEDGER_FILE = "trades.csv"
TIMEZONE_LOCAL = tz.gettz("America/New_York")

# Default watchlist & data source (change in menu)
WATCHLIST = ["BTC", "ETH", "GC=F"]   # add more if you want
DATA_SOURCE = "deribit_perp"         # deribit_perp | coinbase_spot | yahoo_cme | yahoo_crypto

# Lookbacks / intervals
TIMEFRAME = "1h"
MAX_BARS = 3000               # target ~125 days of 1h
YF_INTERVAL = "60m"
YF_PERIOD = "180d"

# Indicators / rules
EMA_FAST, EMA_SLOW = 20, 50
RSI_LEN, ATR_LEN = 14, 14
ATR_STOP_K = 1.6
TP_R_MULT = 2.0
VRVP_BINS = 40
VRVP_LOOKBACK = 300

# Risk display
INITIAL_EQUITY = 10000.0
RISK_PCT = 0.0075   # 0.75%

# Filters
VOL_PCTL_MIN = 30
USE_TOD_FILTER = False
ALLOW_HOURS_UTC = (dtime(13,0), dtime(20,0))

# Backtest approval
APPROVAL_PF_MIN = 1.3
MIN_TRADES_FOR_APPROVAL = 25

# Symbol maps
YAHOO_CRYPTOS = { "BTC":"BTC-USD", "ETH":"ETH-USD", "AVAX":"AVAX-USD", "SOL":"SOL-USD" }
COINBASE_SPOT = { "BTC":"BTC-USD", "ETH":"ETH-USD", "AVAX":"AVAX-USD", "SOL":"SOL-USD" }
DERIBIT_PERP  = { "BTC":"BTC-PERPETUAL", "ETH":"ETH-PERPETUAL" }
CME_MICRO     = { "BTC":"MBT=F", "ETH":"MET=F" }  # used if you pass BTC/ETH to yahoo_cme

# ---------------- UTILS ----------------
def utcnow(): return datetime.now(timezone.utc)
def fmt_ts(ts):
    if isinstance(ts, pd.Timestamp):
        ts = ts.to_pydatetime().replace(tzinfo=timezone.utc)
    return ts.astimezone(TIMEZONE_LOCAL).strftime("%Y-%m-%d %H:%M %Z")

def ensure_ledger():
    if not os.path.exists(LEDGER_FILE):
        with open(LEDGER_FILE, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id","opened_at","symbol","source","side","entry","stop","take_profit",
                        "size","notes","status","closed_at","exit_price","pnl_abs","pnl_pct","r_multiple","tags"])

def read_ledger(): ensure_ledger(); return pd.read_csv(LEDGER_FILE)
def write_ledger(df): df.to_csv(LEDGER_FILE, index=False)
def append_trade(row):
    ensure_ledger()
    with open(LEDGER_FILE, "a", newline="") as f:
        w = csv.writer(f); w.writerow([
            row.get("id"), row.get("opened_at"), row.get("symbol"), row.get("source"),
            row.get("side"), row.get("entry"), row.get("stop"), row.get("take_profit"),
            row.get("size"), row.get("notes",""), row.get("status"), row.get("closed_at",""),
            row.get("exit_price",""), row.get("pnl_abs",""), row.get("pnl_pct",""), row.get("r_multiple",""),
            row.get("tags","{}")
        ])

# ---------------- INDICATORS ----------------
def ema(s: pd.Series, n: int): return s.ewm(span=n, adjust=False).mean()

def rsi(close: pd.Series, n: int = RSI_LEN):
    d = close.diff(); up = d.clip(lower=0); down = -d.clip(upper=0)
    rs = up.ewm(alpha=1/n, adjust=False).mean() / down.ewm(alpha=1/n, adjust=False).mean()
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, n: int = ATR_LEN):
    prev = df["close"].shift()
    tr = pd.concat([(df["high"]-df["low"]).abs(), (df["high"]-prev).abs(), (df["low"]-prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def session_vwap(df: pd.DataFrame):
    y = df.copy()
    y["date"] = y.index.tz_convert("UTC").date if y.index.tz is not None else y.index.date
    tp = (y["high"]+y["low"]+y["close"])/3.0; pv = tp * y["volume"]
    y["cum_pv"] = pv.groupby(y["date"]).cumsum(); y["cum_vol"] = y["volume"].groupby(y["date"]).cumsum().replace(0,np.nan)
    y["vwap"] = y["cum_pv"] / y["cum_vol"]; return y["vwap"]

def vrvp(df: pd.DataFrame, bins: int = VRVP_BINS, lookback: int = VRVP_LOOKBACK):
    w = df.tail(lookback)
    if len(w)<20: return {"poc":np.nan,"hva_low":np.nan,"hva_high":np.nan}
    lo = min(w["low"].min(), w["close"].min()); hi = max(w["high"].max(), w["close"].max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi<=lo: return {"poc":np.nan,"hva_low":np.nan,"hva_high":np.nan}
    hist, edges = np.histogram(w["close"].values, bins=bins, range=(lo,hi), weights=w["volume"].values)
    centers = (edges[:-1]+edges[1:])/2.0
    if hist.sum()<=0: return {"poc":np.nan,"hva_low":np.nan,"hva_high":np.nan}
    poc = centers[hist.argmax()]; pos = hist[hist>0]; cut = np.percentile(pos,70) if len(pos) else 0
    mask = hist>=cut
    if mask.any(): hva_low = centers[mask].min(); hva_high = centers[mask].max()
    else: hva_low=hva_high=np.nan
    return {"poc":float(poc),"hva_low":float(hva_low),"hva_high":float(hva_high)}

# ---------------- DATA ADAPTERS ----------------
def _df_from_ohlc_list(rows, ts_ms=True):
    df = pd.DataFrame(rows)

    # Normalize timestamp column name
    if "ts" not in df.columns:
        if "time" in df.columns:
            df = df.rename(columns={"time": "ts"})
        elif 0 in df.columns:
            df = df.rename(columns={0:"ts",1:"open",2:"high",3:"low",4:"close",5:"volume"})

    # Parse timestamps first (do NOT cast 'ts' to float)
    if ts_ms:
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    else:
        df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)

    # Cast only numeric columns
    for col in ("open","high","low","close","volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean and index by timestamp
    df = df.dropna(subset=["open","high","low","close","volume"])
    df = df.drop_duplicates("ts").set_index("ts").sort_index()
    df.index = df.index.tz_convert("UTC")
    return df


def fetch_deribit_perp(symbol_code: str, max_bars: int = MAX_BARS) -> pd.DataFrame:
    sym = symbol_code.upper()
    if sym not in DERIBIT_PERP:
        raise ValueError("Deribit perps supported: BTC, ETH")
    instrument = DERIBIT_PERP[sym]

    now_ms = int(time.time() * 1000)
    start_ms = now_ms - max_bars * 3600 * 1000  # 1h bars back

    url = "https://www.deribit.com/api/v2/public/get_tradingview_chart_data"
    params = {
        "instrument_name": instrument,
        "resolution": "60",
        "start_timestamp": start_ms,   # <-- correct param name
        "end_timestamp": now_ms        # <-- correct param name
    }

    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    js = r.json()
    if not js.get("result"):
        raise ValueError("No Deribit data")

    res = js["result"]
    rows = []
    # Deribit returns arrays: ticks (ms), open, high, low, close, volume
    for i, ts in enumerate(res["ticks"]):
        rows.append({
            "ts": ts,
            "open":  res["open"][i],
            "high":  res["high"][i],
            "low":   res["low"][i],
            "close": res["close"][i],
            "volume":res["volume"][i],
        })
    return _df_from_ohlc_list(rows, ts_ms=True)

def fetch_coinbase_spot(symbol_code: str, max_bars: int = MAX_BARS) -> pd.DataFrame:
    prod = COINBASE_SPOT.get(symbol_code.upper())
    if not prod: raise ValueError(f"Coinbase spot map missing for {symbol_code}")
    url = f"https://api.exchange.coinbase.com/products/{prod}/candles"
    params = {"granularity": 3600}
    r = requests.get(url, params=params, timeout=20); r.raise_for_status()
    data = r.json()
    if not data: raise ValueError("No Coinbase data")
    rows = []
    for row in data:
        ts_s, low, high, open_, close, vol = row
        rows.append({"ts": ts_s, "open": open_, "high": high, "low": low, "close": close, "volume": vol})
    df = _df_from_ohlc_list(rows, ts_ms=False)
    return df.tail(max_bars)

def fetch_yf(symbol: str, period=YF_PERIOD, interval=YF_INTERVAL) -> pd.DataFrame:
    import yfinance as yf
    data = yf.download(tickers=symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    if data.empty: raise ValueError(f"No yfinance data for {symbol}")
    data = data.rename(columns=str.lower)[["open","high","low","close","volume"]].dropna()
    data.index = (data.index.tz_localize("UTC") if data.index.tz is None else data.index.tz_convert("UTC"))
    return data.astype(float)

def fetch_yahoo_cme(symbol_or_short: str) -> pd.DataFrame:
    s = symbol_or_short.strip().upper()
    yf_symbol = s if ("=F" in s or s.endswith("=F")) else CME_MICRO.get(s)
    if not yf_symbol:
        raise ValueError("For CME futures, use MET=F, MBT=F, GC=F or pass BTC/ETH (mapped to MICROs).")
    return fetch_yf(yf_symbol)

def fetch_yahoo_crypto(symbol_code: str) -> pd.DataFrame:
    yf_sym = YAHOO_CRYPTOS.get(symbol_code.upper())
    if not yf_sym: raise ValueError(f"No Yahoo mapping for {symbol_code}. Add to YAHOO_CRYPTOS.")
    return fetch_yf(yf_sym)

def get_data_1h(symbol: str, source: str) -> tuple[pd.DataFrame, str]:
    if source == "deribit_perp":   return fetch_deribit_perp(symbol), "deribit_perp"
    if source == "coinbase_spot":  return fetch_coinbase_spot(symbol), "coinbase_spot"
    if source == "yahoo_cme":      return fetch_yahoo_cme(symbol), "yahoo_cme"
    if source == "yahoo_crypto":   return fetch_yahoo_crypto(symbol), "yahoo_crypto"
    raise ValueError("Unknown data source")

# ---------------- FEATURES / CONTEXT ----------------
def resample_ohlc(df_1h: pd.DataFrame, rule: str) -> pd.DataFrame:
    # use lowercase '4h' to avoid FutureWarning
    o = df_1h["open"].resample(rule).first()
    h = df_1h["high"].resample(rule).max()
    l = df_1h["low"].resample(rule).min()
    c = df_1h["close"].resample(rule).last()
    v = df_1h["volume"].resample(rule).sum()
    out = pd.concat([o,h,l,c,v], axis=1)
    out.columns = ["open","high","low","close","volume"]
    return out.dropna().astype(float)

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ema_fast"] = ema(d["close"], EMA_FAST)
    d["ema_slow"] = ema(d["close"], EMA_SLOW)
    d["rsi"] = rsi(d["close"], RSI_LEN)
    d["atr"] = atr(d, ATR_LEN)
    d["vwap"] = session_vwap(d)
    return d

def tf_alignment_pack(df1h: pd.DataFrame):
    d1h = compute_features(df1h)
    d4h = compute_features(resample_ohlc(df1h, "4h"))
    dd  = compute_features(resample_ohlc(df1h, "1D"))

    return d1h, d4h, dd

def within_hours_utc(ts: pd.Timestamp) -> bool:
    if not USE_TOD_FILTER: return True
    start, end = ALLOW_HOURS_UTC; t = ts.time()
    return (start <= t <= end)

def volatility_ok(d1h: pd.DataFrame, pctl_min: int = VOL_PCTL_MIN) -> bool:
    series = d1h["atr"].dropna()
    if len(series) < 50: return True
    thr = np.percentile(series.values, pctl_min)
    return float(series.iloc[-1]) >= float(thr)

def build_signal_context(d1h: pd.DataFrame, d4h: pd.DataFrame, dd: pd.DataFrame):
    last1h = d1h.iloc[-1]
    last4h = d4h.iloc[-1]
    lastd  = dd.iloc[-1]

    # scalarize 1h values
    c1h   = float(last1h["close"])
    vwap1 = float(last1h["vwap"])
    ema1f = float(last1h["ema_fast"])
    ema1s = float(last1h["ema_slow"])
    rsi1  = float(last1h["rsi"])
    atr1  = float(last1h["atr"])

    # 4h scalars
    c4h   = float(last4h["close"])
    vwap4 = float(last4h["vwap"])

    # daily scalars
    emadf = float(lastd["ema_fast"])
    emads = float(lastd["ema_slow"])

    profile = vrvp(d1h, bins=VRVP_BINS, lookback=VRVP_LOOKBACK)
    poc = profile.get("poc", np.nan)
    poc = float(poc) if poc == poc else np.nan

    regime_long  = emadf > emads
    regime_short = emadf < emads
    tf_ok_long   = regime_long  and (c4h > vwap4) and (c1h > vwap1)
    tf_ok_short  = regime_short and (c4h < vwap4) and (c1h < vwap1)
    long_window  = (50.0 <= rsi1 <= 65.0)
    short_window = (35.0 <= rsi1 <= 50.0)
    poc_ok_long  = (np.isnan(poc) or c1h >= poc)
    poc_ok_short = (np.isnan(poc) or c1h <= poc)

    one_h = pd.Series(
        {"close": c1h, "vwap": vwap1, "ema_fast": ema1f, "ema_slow": ema1s, "rsi": rsi1, "atr": atr1},
        name=d1h.index[-1]
    )

    return {
        "last_ts": d1h.index[-1],
        "last1h": one_h,
        "last4h": last4h,
        "lastd":  lastd,
        "profile": {"poc": poc, "hva_low": profile.get("hva_low"), "hva_high": profile.get("hva_high")},
        "tf_ok_long": bool(tf_ok_long),
        "tf_ok_short": bool(tf_ok_short),
        "regime": "bull" if regime_long else ("bear" if regime_short else "neutral"),
        "long_window": bool(long_window),
        "short_window": bool(short_window),
        "poc_ok_long": bool(poc_ok_long),
        "poc_ok_short": bool(poc_ok_short),
    }

def propose_trade(ctx: dict):
    if not within_hours_utc(ctx["last_ts"]): return None
    last = ctx["last1h"]
    reasons = []; signal=None

    if ctx["tf_ok_long"] and ctx["long_window"] and ctx["poc_ok_long"]:
        signal="LONG"; reasons += [
            "Daily trend up (EMA20>EMA50)",
            "4h and 1h above session VWAP",
            f"RSI {last['rsi']:.1f} within 50–65",
        ]
        if np.isfinite(ctx["profile"].get("poc", np.nan)):
            reasons.append(f"Away from POC ~ {ctx['profile']['poc']:.2f}")
    elif ctx["tf_ok_short"] and ctx["short_window"] and ctx["poc_ok_short"]:
        signal="SHORT"; reasons += [
            "Daily trend down (EMA20<EMA50)",
            "4h and 1h below session VWAP",
            f"RSI {last['rsi']:.1f} within 35–50",
        ]
        if np.isfinite(ctx["profile"].get("poc", np.nan)):
            reasons.append(f"Away from POC ~ {ctx['profile']['poc']:.2f}")

    if not signal: return None

    entry = float(last["close"])
    stop_dist = ATR_STOP_K * float(last["atr"])
    if not np.isfinite(stop_dist) or stop_dist <= 0: return None

    if signal=="LONG":
        sl = entry - stop_dist; tp = entry + TP_R_MULT * (entry - sl)
    else:
        sl = entry + stop_dist; tp = entry - TP_R_MULT * (sl - entry)

    tags = {"regime":ctx["regime"],"tf_ok":True,
            "rsi_bucket":"50-65" if signal=="LONG" else "35-50",
            "poc_dir":"above" if signal=="LONG" else "below"}
    return {"signal":signal,"entry":entry,"stop":sl,"take_profit":tp,"why":reasons,"tags":tags}

# ---------------- BACKTEST ----------------
def compute_max_drawdown(equity_curve: pd.Series) -> float:
    peak = equity_curve.cummax(); dd = equity_curve - peak; return float(dd.min())

def backtest_symbol(symbol: str, source: str):
    df1h, src = get_data_1h(symbol, source)
    if len(df1h) < 500:
        print(f"[{symbol} @ {source}] Not enough data ({len(df1h)} bars)."); return {"symbol":symbol,"trades":0}
    d1h, d4h, dd = tf_alignment_pack(df1h)
    trades = []; in_trade=False; side=None; entry=sl=tp=None; open_time=None; tags=None

    for i in range(max(EMA_SLOW,RSI_LEN,ATR_LEN)+2, len(d1h)-1):
        ts = d1h.index[i]
        d4_idx = d4h.index.asof(ts); d_idx = dd.index.asof(ts)
        if pd.isna(d4_idx) or pd.isna(d_idx): continue

        ctx = build_signal_context(d1h.loc[:ts], d4h.loc[:d4_idx], dd.loc[:d_idx])
        if not volatility_ok(d1h.loc[:ts]): continue
        if not within_hours_utc(ts): continue

        bar = d1h.iloc[i]

        if in_trade:
            hit_tp = (bar.high >= tp) if side=="LONG" else (bar.low <= tp)
            hit_sl = (bar.low <= sl) if side=="LONG" else (bar.high >= sl)
            exit_price=None
            if hit_tp and hit_sl: exit_price=sl
            elif hit_tp: exit_price=tp
            elif hit_sl: exit_price=sl
            if exit_price is not None:
                r_per_unit = (exit_price - entry) if side=="LONG" else (entry - exit_price)
                risk = abs(entry - sl); r_mult = (r_per_unit / risk) if risk > 0 else np.nan
                trades.append({"open":open_time,"close":ts,"side":side,"entry":entry,"exit":exit_price,"sl":sl,"tp":tp,"r":r_mult,"pnl":r_per_unit,"tags":tags})
                in_trade=False; side=None; entry=sl=tp=None; open_time=None; tags=None
            continue

        sig = propose_trade(ctx)
        if sig:
            in_trade=True; side=sig["signal"]; entry=float(bar.close)
            sl=float(sig["stop"]); tp=float(sig["take_profit"]); open_time=ts; tags=sig["tags"]

    if in_trade:
        last = d1h.iloc[-1]
        r_per_unit = (last.close - entry) if side=="LONG" else (entry - last.close)
        risk = abs(entry - sl); r_mult = (r_per_unit / risk) if risk > 0 else np.nan
        trades.append({"open":open_time,"close":d1h.index[-1],"side":side,"entry":entry,"exit":float(last.close),"sl":sl,"tp":tp,"r":r_mult,"pnl":r_per_unit,"tags":tags})

    if not trades:
        print(f"[{symbol} @ {source}] Backtest produced 0 trades.")
        return {"symbol":symbol,"trades":0}

    tdf = pd.DataFrame(trades)
    wins = (tdf["pnl"]>0).sum(); losses=(tdf["pnl"]<=0).sum()
    win_rate = (wins/len(tdf))*100
    pf = (tdf.loc[tdf["pnl"]>0,"pnl"].sum()/abs(tdf.loc[tdf["pnl"]<=0,"pnl"].sum())) if losses>0 else float("inf")
    avg_r = tdf["r"].mean(); max_dd = compute_max_drawdown(tdf["pnl"].cumsum())
    out_csv = f"backtest_{symbol.replace('=','')}_{source}.csv"; tdf.to_csv(out_csv, index=False)
    print(f"[{symbol} @ {source}] Trades:{len(tdf)} | Win:{win_rate:.1f}% | PF:{pf:.2f} | AvgR:{avg_r:.2f} | MaxDD:{max_dd:.2f} | Saved:{out_csv}")
    return {"symbol":symbol,"source":source,"trades":int(len(tdf)),"win_rate":float(win_rate),"pf":float(pf),"avg_r":float(avg_r),"max_dd":float(max_dd)}

# ---------------- ANALYZE ----------------
def analyze_symbol(symbol: str, source: str):
    df1h, src = get_data_1h(symbol, source)
    d1h, d4h, dd = tf_alignment_pack(df1h)
    if not volatility_ok(d1h):
        print(f"\n[{symbol}@{source}] Volatility filter: ATR too low vs history → skip.")
        return None, src, float(d1h.iloc[-1].close)

    ctx = build_signal_context(d1h, d4h, dd)
    sig = propose_trade(ctx)

    print("\n=== Analysis ===")
    print(f"Symbol: {symbol}  | Source: {src}")
    print(f"Last bar: {fmt_ts(d1h.index[-1])}")
    l1 = ctx["last1h"]; prof = ctx["profile"]
    print(f"Close: {l1['close']:.4f} | VWAP: {l1['vwap']:.4f} | EMA{EMA_FAST}:{l1['ema_fast']:.2f} EMA{EMA_SLOW}:{l1['ema_slow']:.2f} | RSI:{l1['rsi']:.1f}")
    if np.isfinite(prof.get('poc', np.nan)):
        print(f"VRVP — POC≈{prof['poc']:.2f}  HVA≈[{prof.get('hva_low'):.2f}–{prof.get('hva_high'):.2f}] (lookback {VRVP_LOOKBACK})")
    print(f"Regime: {ctx['regime']} | TF OK (long/short): {ctx['tf_ok_long']}/{ctx['tf_ok_short']}")

    if sig:
        print(f"\nSuggestion: go {sig['signal']} at {sig['entry']:.2f}")
        print(f"Stop loss at {sig['stop']:.2f} | Close position (TP) at {sig['take_profit']:.2f}")
        print("Why: " + "; ".join(sig["why"]))
        per_unit_risk = abs(sig["entry"] - sig["stop"]); risk_dollars = INITIAL_EQUITY * RISK_PCT
        qty = (risk_dollars / per_unit_risk) if per_unit_risk > 0 else 0
        print(f"(Ref size @ {RISK_PCT*100:.2f}% of ${INITIAL_EQUITY:,.0f} → ~{qty:.4f} units)")
    else:
        print("\nNo suggested position now (filters not aligned).")
    return sig, src, float(l1["close"])

# ---------------- LEDGER ----------------
def log_new_trade(symbol, source, side, entry, stop, tp, size, notes="", tags=None):
    tid = str(uuid.uuid4())[:8].upper()
    row = {
        "id":tid, "opened_at":utcnow().isoformat(), "symbol":symbol, "source":source,
        "side":side.upper(), "entry":float(entry), "stop":float(stop), "take_profit":float(tp),
        "size":float(size), "notes":notes, "status":"OPEN", "closed_at":"",
        "exit_price":"", "pnl_abs":"", "pnl_pct":"", "r_multiple":"", "tags": json.dumps(tags or {})
    }
    append_trade(row); print(f"\nLogged {tid}: {side.upper()} {symbol} @ {entry:.4f} | SL {stop:.4f} | TP {tp:.4f} | size {size}")
    return tid

def close_trade(trade_id: str, exit_price: float):
    df = read_ledger()
    if df.empty or trade_id not in set(df["id"].astype(str)): print("Trade ID not found."); return
    idx = df.index[df["id"].astype(str)==trade_id][0]; row = df.loc[idx]
    if row["status"]=="CLOSED": print("Trade already closed."); return
    side=row["side"]; entry=float(row["entry"]); size=float(row["size"])
    pnl_per_unit = (exit_price - entry) if side=="LONG" else (entry - exit_price)
    risk_per_unit = abs(entry - float(row["stop"])); r_mult = (pnl_per_unit / risk_per_unit) if risk_per_unit>0 else np.nan
    pnl_abs = pnl_per_unit * size; pnl_pct = (pnl_per_unit / entry) * (1 if side=="LONG" else -1) * 100.0
    df.loc[idx, ["status","closed_at","exit_price","pnl_abs","pnl_pct","r_multiple"]] = [
        "CLOSED", utcnow().isoformat(), float(exit_price), float(pnl_abs), float(pnl_pct), float(r_mult) if np.isfinite(r_mult) else ""
    ]
    write_ledger(df); print(f"\nClosed {trade_id} @ {exit_price:.4f} | PnL ${pnl_abs:.2f} | R={r_mult:.2f}")

def pnl_summary():
    df = read_ledger(); print("\n=== P&L Summary ===")
    if df.empty: print("No trades logged."); return
    total=len(df); openpos=(df["status"]=="OPEN").sum(); closed=df[df["status"]=="CLOSED"].copy()
    print(f"Total:{total} | Open:{openpos} | Closed:{len(closed)}")
    if not closed.empty:
        closed["pnl_abs"]=pd.to_numeric(closed["pnl_abs"], errors="coerce")
        wins=(closed["pnl_abs"]>0).sum(); total_pnl=closed["pnl_abs"].sum(); avg=closed["pnl_abs"].mean(); wr=(wins/len(closed))*100
        print(f"Closed PnL: ${total_pnl:.2f} | Avg/trade: ${avg:.2f} | Win rate: {wr:.1f}%")
        if "r_multiple" in closed:
            closed["r_multiple"]=pd.to_numeric(closed["r_multiple"], errors="coerce")
            if closed["r_multiple"].notna().any():
                print(f"Avg R: {closed['r_multiple'].mean():.2f} | Best R: {closed['r_multiple'].max():.2f}")
    else:
        print("No closed trades yet.")

# ---------------- APPROVAL PIPELINE ----------------
def backtest_all_and_approve(symbols, source):
    print(f"\nRunning backtests on source: {source} …")
    approvals = {}; report=[]
    for s in symbols:
        try:
            res = backtest_symbol(s, source)
        except Exception as e:
            print(f"[{s} @ {source}] backtest error: {e}")
            approvals[s]=False; continue
        if not res or res.get("trades",0)==0: approvals[s]=False; continue
        ok = (res["pf"]>=APPROVAL_PF_MIN) and (res["trades"]>=MIN_TRADES_FOR_APPROVAL)
        approvals[s]=bool(ok)
        report.append((s,res["trades"],res["win_rate"],res["pf"],res["avg_r"],res["max_dd"],ok))
    if report:
        print("\n=== Scorecard (PF≥{:.2f}, trades≥{}) ===".format(APPROVAL_PF_MIN, MIN_TRADES_FOR_APPROVAL))
        print("Symbol | Trades | Win% | PF | AvgR | MaxDD | Approved")
        for (s,tr,wr,pf,ar,dd,ok) in sorted(report, key=lambda x:(-x[3],-x[1])):
            print(f"{s:>6} | {tr:>6} | {wr:>5.1f} | {pf:>4.2f} | {ar:>4.2f} | {dd:>6.2f} | {'YES' if ok else 'no'}")
    else:
        print("No results.")
    return approvals

# ---------------- CLI ----------------
def print_source_help():
    print("""
Data sources:
 - deribit_perp   : True perpetuals for BTC/ETH (public; closest free futures feed)
 - coinbase_spot  : Coinbase spot (public) — exact Coinbase price
 - yahoo_cme      : Regulated futures via Yahoo (MET=F, MBT=F, GC=F, etc.)
 - yahoo_crypto   : Yahoo crypto spot (ETH-USD, BTC-USD, etc.)

Notes:
 - For ETH/BTC perps without region constraints, choose 'deribit_perp'.
 - For regulated US trading, choose 'yahoo_cme' and analyze 'MET=F' / 'MBT=F'.
 - 'yahoo_cme' won’t support AVAX/SOL; use 'deribit_perp' or 'yahoo_crypto' for those.
""")

MENU = """
==============================
Alpha Trader — Main Menu
0) Set data source (current: {source})
1) Analyze market (approved symbols only)
2) Backtest ALL & approve symbols
3) Log a new trade (use last analysis or manual)
4) Close a trade
5) View P&L summary
6) Analyze ANY symbol (override approval)
7) Exit
==============================
"""

def main():
    global DATA_SOURCE
    print("Alpha Trader — Multi-Source (EDU ONLY).")
    print_source_help()
    print(f"Watchlist: {', '.join(WATCHLIST)}\n")
    approvals = {s: False for s in WATCHLIST}
    last_sig_cache = None  # (symbol, sig, source, last_price)

    while True:
        try:
            print(MENU.format(source=DATA_SOURCE))
            choice = input("Choose (0-7): ").strip()

            if choice == "0":
                print_source_help()
                ds = input("Enter data source (deribit_perp | coinbase_spot | yahoo_cme | yahoo_crypto): ").strip()
                if ds in {"deribit_perp","coinbase_spot","yahoo_cme","yahoo_crypto"}:
                    DATA_SOURCE = ds
                    print(f"Data source set to: {DATA_SOURCE}")
                else:
                    print("Invalid source.")
                input("\nPress Enter…")

            elif choice == "1":
                if not any(approvals.values()):
                    print("No approved symbols yet. Run option 2 (Backtest ALL) first.")
                    input("\nPress Enter…"); continue
                for sym in [s for s in WATCHLIST if approvals.get(s)]:
                    try:
                        sig, src, last_px = analyze_symbol(sym, DATA_SOURCE)
                        last_sig_cache = (sym, sig, src, last_px)
                    except Exception as e:
                        print(f"[{sym}] analyze error: {e}")
                input("\nPress Enter…")

            elif choice == "2":
                approvals = backtest_all_and_approve(WATCHLIST, DATA_SOURCE)
                input("\nPress Enter…")

            elif choice == "3":
                ensure_ledger()
                use_last = input("Use last analysis suggestion? (y/n): ").strip().lower()
                if use_last == "y" and last_sig_cache and last_sig_cache[1]:
                    sym, sig, src, last_px = last_sig_cache
                    size = float(input("Position size (units/contracts): ").strip() or "1")
                    notes = input("Notes (optional): ").strip()
                    log_new_trade(sym, src, sig["signal"], sig["entry"], sig["stop"], sig["take_profit"], size, notes, tags=sig.get("tags"))
                else:
                    sym = input("Symbol: ").strip()
                    src = DATA_SOURCE
                    side = input("Side (LONG/SHORT): ").strip().upper()
                    entry = float(input("Entry price: ").strip())
                    stop  = float(input("Stop loss: ").strip())
                    tp    = float(input("Take profit: ").strip())
                    size  = float(input("Position size (units/contracts): ").strip() or "1")
                    notes = input("Notes (optional): ").strip()
                    log_new_trade(sym, src, side, entry, stop, tp, size, notes, tags={})
                input("\nPress Enter…")

            elif choice == "4":
                df = read_ledger(); opens = df[df["status"]=="OPEN"]
                if opens.empty:
                    print("No open trades.")
                else:
                    print("\nOpen trades:")
                    for _, r in opens.iterrows():
                        print(f"- {r['id']}: {r['side']} {r['symbol']} entry {r['entry']} SL {r['stop']} TP {r['take_profit']} (opened {r['opened_at']})")
                    tid = input("Enter Trade ID to close: ").strip()
                    exit_px = float(input("Exit price: ").strip())
                    close_trade(tid, exit_px)
                input("\nPress Enter…")

            elif choice == "5":
                pnl_summary(); input("\nPress Enter…")

            elif choice == "6":
                sym = input("Enter ANY symbol (e.g., BTC, ETH, AVAX, SOL, GC=F, MET=F): ").strip()
                try:
                    sig, src, last_px = analyze_symbol(sym, DATA_SOURCE)
                    last_sig_cache = (sym, sig, src, last_px)
                except Exception as e:
                    print(f"[{sym}] analyze error: {e}")
                input("\nPress Enter…")

            elif choice == "7":
                print("Done. Trade small. Protect downside."); break
            else:
                print("Invalid option.")
        except KeyboardInterrupt:
            print("\nExiting."); break
        except Exception as e:
            print("\nError:", e); traceback.print_exc(); input("\nPress Enter…")

if __name__ == "__main__":
    main()
