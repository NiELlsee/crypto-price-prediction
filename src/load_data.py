"""Загрузка исторических свечей (OHLCV) для SOL и BTC.

Основной источник — публичный API Binance (ключ не нужен), пары SOL/USDT и BTC/USDT.
Запасной источник — Yahoo Finance через библиотеку yfinance (тикеры SOL-USD и BTC-USD).

Файлы сохраняются в data/raw/: sol_<interval>.csv и btc_<interval>.csv.

Примеры запуска из корня проекта:
    python src/load_data.py                       # дневные свечи с 2020-08-11
    python src/load_data.py --interval 1h         # часовые свечи
    python src/load_data.py --source yfinance     # сразу через Yahoo Finance
    python src/load_data.py --start 2021-01-01 --end 2026-10-01
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

# data-api.binance.vision — официальное зеркало Binance только для рыночных данных.
BINANCE_HOSTS = [
    "https://data-api.binance.vision",
    "https://api.binance.com",
    "https://api1.binance.com",
]
BINANCE_SYMBOLS = {"sol": "SOLUSDT", "btc": "BTCUSDT"}
YF_TICKERS = {"sol": "SOL-USD", "btc": "BTC-USD"}

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "n_trades", "taker_buy_base", "taker_buy_quote", "ignore",
]
OUTPUT_COLUMNS = [
    "timestamp", "open", "high", "low", "close", "volume",
    "quote_volume", "n_trades", "taker_buy_base", "taker_buy_quote",
]
# SOL начал торговаться на Binance 11 августа 2020.
DEFAULT_START = "2020-08-11"


def _to_ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _binance_get(path: str, params: dict) -> list:
    """GET-запрос к Binance с перебором зеркал."""
    last_error = None
    for host in BINANCE_HOSTS:
        try:
            resp = requests.get(host + path, params=params, timeout=20)
        except requests.RequestException as exc:
            last_error = f"{host}: {exc}"
            continue
        if resp.status_code == 200:
            return resp.json()
        # 451 / 403 — ограничение по региону, пробуем следующее зеркало
        last_error = f"{host}: HTTP {resp.status_code} {resp.text[:200]}"
    raise RuntimeError(f"Binance недоступен. Последняя ошибка: {last_error}")


def parse_klines(rows: list) -> pd.DataFrame:
    """Превращает ответ /api/v3/klines в аккуратную таблицу."""
    df = pd.DataFrame(rows, columns=KLINE_COLUMNS).drop(columns="ignore")
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    # Последняя свеча может быть ещё не закрыта — убираем её.
    df = df[df["close_time"].astype("int64") < now_ms]
    df["timestamp"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    float_cols = ["open", "high", "low", "close", "volume",
                  "quote_volume", "taker_buy_base", "taker_buy_quote"]
    df[float_cols] = df[float_cols].astype(float)
    df["n_trades"] = df["n_trades"].astype("int64")
    df = df[OUTPUT_COLUMNS].drop_duplicates("timestamp").sort_values("timestamp")
    return df.reset_index(drop=True)


def load_binance(symbol: str, interval: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Скачивает все свечи за период, постранично по 1000 штук."""
    start_ms = _to_ms(start)
    end_ms = _to_ms(end) if end else None
    rows: list = []
    while True:
        params = {"symbol": symbol, "interval": interval, "startTime": start_ms, "limit": 1000}
        if end_ms is not None:
            params["endTime"] = end_ms
        batch = _binance_get("/api/v3/klines", params)
        if not batch:
            break
        rows.extend(batch)
        start_ms = batch[-1][0] + 1
        if len(batch) < 1000:
            break
        time.sleep(0.2)  # бережём лимиты API
    if not rows:
        raise RuntimeError(f"Binance вернул пустой ответ для {symbol}")
    return parse_klines(rows)


def load_yfinance(ticker: str, interval: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Запасной вариант. Внимание: часовые данные в Yahoo доступны только за последние ~730 дней,
    а объём там указан в долларах."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Не установлен yfinance: pip install yfinance") from exc

    df = yf.download(ticker, start=start, end=end, interval=interval,
                     auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(f"Yahoo Finance вернул пустой ответ для {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df = df.rename(columns={c: str(c).lower() for c in df.columns})
    df = df.rename(columns={"date": "timestamp", "datetime": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Загрузка свечей SOL и BTC")
    parser.add_argument("--interval", default="1d", help="1d, 4h, 1h и т.п. (по умолчанию 1d)")
    parser.add_argument("--start", default=DEFAULT_START, help="дата начала, YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="дата конца, YYYY-MM-DD (по умолчанию — сейчас)")
    parser.add_argument("--source", choices=["auto", "binance", "yfinance"], default="auto",
                        help="auto: сначала Binance, при ошибке — Yahoo Finance")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for asset in ("sol", "btc"):
        df, source = None, None
        if args.source in ("auto", "binance"):
            try:
                df = load_binance(BINANCE_SYMBOLS[asset], args.interval, args.start, args.end)
                source = "binance"
            except RuntimeError as exc:
                print(f"[{asset}] {exc}", file=sys.stderr)
                if args.source == "binance":
                    return 1
        if df is None:
            df = load_yfinance(YF_TICKERS[asset], args.interval, args.start, args.end)
            source = "yfinance"

        path = RAW_DIR / f"{asset}_{args.interval}.csv"
        df.to_csv(path, index=False)
        print(f"[{asset}] {source}: {len(df)} свечей, "
              f"{df['timestamp'].min():%Y-%m-%d} — {df['timestamp'].max():%Y-%m-%d} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
