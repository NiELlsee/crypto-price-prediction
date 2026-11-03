"""Загрузка исторических данных SOL и BTC: фьючерсы, funding rate и спот.

Основной объект проекта — бессрочные (perpetual) фьючерсы SOLUSDT и BTCUSDT
на Binance USDⓈ-M. Спотовые свечи качаются для сравнения (базис фьючерс − спот).

Источники:
    фьючерсы и funding  — Binance USDⓈ-M (fapi.binance.com), запасной — Bybit (linear)
    спот                — Binance Spot, запасной — Yahoo Finance

Файлы сохраняются в data/raw/:
    sol_futures_1d.csv, btc_futures_1d.csv   свечи фьючерсов
    sol_funding.csv,    btc_funding.csv      история ставки финансирования (обычно раз в 8 часов)
    sol_spot_1d.csv,    btc_spot_1d.csv      свечи спота

Примеры запуска из корня проекта:
    python src/load_data.py                     # всё: фьючерсы, funding, спот (дневные свечи)
    python src/load_data.py --market futures    # только фьючерсы и funding
    python src/load_data.py --interval 1h       # часовые свечи
    python src/load_data.py --source bybit      # фьючерсы и funding сразу с Bybit
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

ASSETS = {"sol": "SOLUSDT", "btc": "BTCUSDT"}
YF_TICKERS = {"sol": "SOL-USD", "btc": "BTC-USD"}

# data-api.binance.vision — официальное зеркало Binance только для спотовых рыночных данных.
BINANCE_SPOT_HOSTS = ["https://data-api.binance.vision", "https://api.binance.com", "https://api1.binance.com"]
BINANCE_FUTURES_HOSTS = ["https://fapi.binance.com"]
BYBIT_HOST = "https://api.bybit.com"
BYBIT_INTERVALS = {"1d": "D", "4h": "240", "1h": "60"}

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "n_trades", "taker_buy_base", "taker_buy_quote", "ignore",
]
OUTPUT_COLUMNS = [
    "timestamp", "open", "high", "low", "close", "volume",
    "quote_volume", "n_trades", "taker_buy_base", "taker_buy_quote",
]
# SOL появилась на Binance 11 августа 2020; фьючерс запущен позже — API сам вернёт первую доступную свечу.
DEFAULT_START = "2020-08-11"


def _to_ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _get(hosts: list[str], path: str, params: dict):
    """GET-запрос с перебором зеркал. Возвращает разобранный JSON."""
    last_error = None
    for host in hosts:
        try:
            resp = requests.get(host + path, params=params, timeout=20)
        except requests.RequestException as exc:
            last_error = f"{host}: {exc}"
            continue
        if resp.status_code == 200:
            return resp.json()
        # 451 / 403 — ограничение по региону, пробуем следующее зеркало
        last_error = f"{host}: HTTP {resp.status_code} {resp.text[:200]}"
    raise RuntimeError(f"Источник недоступен. Последняя ошибка: {last_error}")


# ---------------------------------------------------------------- Binance: свечи (спот и фьючерсы)

def parse_klines(rows: list) -> pd.DataFrame:
    """Превращает ответ /klines Binance (одинаковый у спота и фьючерсов) в таблицу."""
    df = pd.DataFrame(rows, columns=KLINE_COLUMNS).drop(columns="ignore")
    # Последняя свеча может быть ещё не закрыта — убираем её.
    df = df[df["close_time"].astype("int64") < _now_ms()]
    df["timestamp"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    float_cols = ["open", "high", "low", "close", "volume",
                  "quote_volume", "taker_buy_base", "taker_buy_quote"]
    df[float_cols] = df[float_cols].astype(float)
    df["n_trades"] = df["n_trades"].astype("int64")
    df = df[OUTPUT_COLUMNS].drop_duplicates("timestamp").sort_values("timestamp")
    return df.reset_index(drop=True)


def load_binance_klines(symbol: str, interval: str, start: str, end: str | None,
                        market: str) -> pd.DataFrame:
    """Скачивает все свечи за период кусками по 1000 (лимит API)."""
    hosts, path = ((BINANCE_FUTURES_HOSTS, "/fapi/v1/klines") if market == "futures"
                   else (BINANCE_SPOT_HOSTS, "/api/v3/klines"))
    start_ms, end_ms = _to_ms(start), (_to_ms(end) if end else None)
    rows: list = []
    while True:
        params = {"symbol": symbol, "interval": interval, "startTime": start_ms, "limit": 1000}
        if end_ms is not None:
            params["endTime"] = end_ms
        batch = _get(hosts, path, params)
        if not batch:
            break
        rows.extend(batch)
        start_ms = batch[-1][0] + 1          # следующий кусок — сразу после последней свечи
        if len(batch) < 1000:
            break
        time.sleep(0.2)                      # бережём лимиты API
    if not rows:
        raise RuntimeError(f"Binance вернул пустой ответ для {symbol} ({market})")
    return parse_klines(rows)


# ---------------------------------------------------------------- Binance: funding rate

def load_binance_funding(symbol: str, start: str, end: str | None) -> pd.DataFrame:
    """История ставки финансирования бессрочного фьючерса, кусками по 1000 записей."""
    start_ms, end_ms = _to_ms(start), (_to_ms(end) if end else None)
    rows: list = []
    while True:
        params = {"symbol": symbol, "startTime": start_ms, "limit": 1000}
        if end_ms is not None:
            params["endTime"] = end_ms
        batch = _get(BINANCE_FUTURES_HOSTS, "/fapi/v1/fundingRate", params)
        if not batch:
            break
        rows.extend(batch)
        start_ms = batch[-1]["fundingTime"] + 1
        if len(batch) < 1000:
            break
        time.sleep(0.2)
    if not rows:
        raise RuntimeError(f"Binance вернул пустую историю funding для {symbol}")
    df = pd.DataFrame(rows)
    out = pd.DataFrame({
        "timestamp": pd.to_datetime(df["fundingTime"].astype("int64"), unit="ms", utc=True),
        "funding_rate": df["fundingRate"].astype(float),
    })
    return out.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------- Bybit (запасной для фьючерсов)

def _bybit(path: str, params: dict) -> list:
    data = _get([BYBIT_HOST], path, params)
    if data.get("retCode") != 0:
        raise RuntimeError(f"Bybit: {data.get('retCode')} {data.get('retMsg')}")
    return data["result"]["list"]


def load_bybit_klines(symbol: str, interval: str, start: str, end: str | None) -> pd.DataFrame:
    """Свечи линейного бессрочного фьючерса Bybit. API отдаёт новые свечи первыми,
    поэтому идём от конца периода к началу."""
    if interval not in BYBIT_INTERVALS:
        raise RuntimeError(f"Bybit: интервал {interval} не поддержан в этом скрипте")
    start_ms, end_ms = _to_ms(start), (_to_ms(end) if end else _now_ms())
    rows: list = []
    while end_ms > start_ms:
        batch = _bybit("/v5/market/kline", {"category": "linear", "symbol": symbol,
                                            "interval": BYBIT_INTERVALS[interval],
                                            "start": start_ms, "end": end_ms, "limit": 1000})
        if not batch:
            break
        rows.extend(batch)
        end_ms = int(batch[-1][0]) - 1       # самая старая свеча в куске — последняя в списке
        if len(batch) < 1000:
            break
        time.sleep(0.2)
    if not rows:
        raise RuntimeError(f"Bybit вернул пустой ответ для {symbol}")
    df = pd.DataFrame(rows, columns=["start", "open", "high", "low", "close", "volume", "turnover"])
    df["timestamp"] = pd.to_datetime(df["start"].astype("int64"), unit="ms", utc=True)
    df[["open", "high", "low", "close", "volume", "turnover"]] = \
        df[["open", "high", "low", "close", "volume", "turnover"]].astype(float)
    df = df.rename(columns={"turnover": "quote_volume"})
    # последняя свеча ещё не закрыта
    step = pd.Timedelta({"1d": "1D", "4h": "4h", "1h": "1h"}[interval])
    df = df[df["timestamp"] + step <= pd.Timestamp.now(tz="UTC")]
    df = df[["timestamp", "open", "high", "low", "close", "volume", "quote_volume"]]
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def load_bybit_funding(symbol: str, start: str, end: str | None) -> pd.DataFrame:
    """История funding Bybit: по 200 записей, от конца периода к началу."""
    start_ms, end_ms = _to_ms(start), (_to_ms(end) if end else _now_ms())
    rows: list = []
    while end_ms > start_ms:
        batch = _bybit("/v5/market/funding/history",
                       {"category": "linear", "symbol": symbol, "endTime": end_ms, "limit": 200})
        if not batch:
            break
        rows.extend(batch)
        end_ms = int(batch[-1]["fundingRateTimestamp"]) - 1
        if len(batch) < 200:
            break
        time.sleep(0.2)
    if not rows:
        raise RuntimeError(f"Bybit вернул пустую историю funding для {symbol}")
    df = pd.DataFrame(rows)
    out = pd.DataFrame({
        "timestamp": pd.to_datetime(df["fundingRateTimestamp"].astype("int64"), unit="ms", utc=True),
        "funding_rate": df["fundingRate"].astype(float),
    })
    out = out[out["timestamp"] >= pd.Timestamp(start, tz="UTC")]
    return out.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------- Yahoo (запасной для спота)

def load_yfinance(ticker: str, interval: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Внимание: часовые данные в Yahoo доступны только за последние ~730 дней,
    а объём там указан в долларах."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Не установлен yfinance: pip install yfinance") from exc
    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
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


# ---------------------------------------------------------------- запуск

def _first_ok(attempts: list, label: str):
    """Пробует источники по очереди, возвращает (таблица, имя источника)."""
    for name, fn in attempts:
        try:
            return fn(), name
        except RuntimeError as exc:
            print(f"[{label}] {name}: {exc}", file=sys.stderr)
    raise RuntimeError(f"[{label}] ни один источник не ответил")


def _save(df: pd.DataFrame, name: str, source: str) -> None:
    path = RAW_DIR / name
    df.to_csv(path, index=False)
    print(f"{name}: {source}, {len(df)} строк, "
          f"{df['timestamp'].min():%Y-%m-%d} — {df['timestamp'].max():%Y-%m-%d}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Загрузка данных SOL и BTC: фьючерсы, funding, спот")
    parser.add_argument("--interval", default="1d", help="1d, 4h, 1h (по умолчанию 1d)")
    parser.add_argument("--start", default=DEFAULT_START, help="дата начала, YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="дата конца, YYYY-MM-DD (по умолчанию — сейчас)")
    parser.add_argument("--market", choices=["all", "futures", "spot"], default="all")
    parser.add_argument("--source", choices=["auto", "binance", "bybit"], default="auto",
                        help="для фьючерсов и funding: auto — сначала Binance, при ошибке Bybit")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    a = args
    failed = False
    for asset, symbol in ASSETS.items():
        try:
            if a.market in ("all", "futures"):
                kl, fu = [], []
                if a.source in ("auto", "binance"):
                    kl.append(("binance", lambda: load_binance_klines(symbol, a.interval, a.start, a.end, "futures")))
                    fu.append(("binance", lambda: load_binance_funding(symbol, a.start, a.end)))
                if a.source in ("auto", "bybit"):
                    kl.append(("bybit", lambda: load_bybit_klines(symbol, a.interval, a.start, a.end)))
                    fu.append(("bybit", lambda: load_bybit_funding(symbol, a.start, a.end)))
                df, src = _first_ok(kl, f"{asset} futures")
                _save(df, f"{asset}_futures_{a.interval}.csv", src)
                df, src = _first_ok(fu, f"{asset} funding")
                _save(df, f"{asset}_funding.csv", src)
            if a.market in ("all", "spot"):
                df, src = _first_ok([
                    ("binance", lambda: load_binance_klines(symbol, a.interval, a.start, a.end, "spot")),
                    ("yfinance", lambda: load_yfinance(YF_TICKERS[asset], a.interval, a.start, a.end)),
                ], f"{asset} spot")
                _save(df, f"{asset}_spot_{a.interval}.csv", src)
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
