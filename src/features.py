"""Таблица признаков и целевых переменных для моделей (чекпойнт 3).

Одна строка = один торговый день t. Все признаки посчитаны только из данных,
известных на закрытии дня t (00:00 UTC следующего дня). Целевые переменные —
будущая лог-доходность фьючерса SOL за h дней: y_h(t) = ln(P_{t+h} / P_t).

Использование:
    from features import build_dataset, FEATURES
    data = build_dataset("data/raw")
    X, y = data[FEATURES], data["y_1"]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = (1, 3, 7)


def _load_candles(raw: Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(raw / name)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
    return df.set_index("timestamp").sort_index()


def _load_funding_daily(raw: Path, name: str) -> pd.Series:
    """Сумма выплат funding за сутки UTC. Последняя выплата суток (16:00 UTC)
    известна до закрытия дневной свечи, поэтому утечки нет."""
    f = pd.read_csv(raw / name)
    f["timestamp"] = pd.to_datetime(f["timestamp"], format="ISO8601", utc=True)
    f = f.set_index("timestamp").sort_index()
    daily = f["funding_rate"].resample("D").sum(min_count=1)
    daily.index = daily.index.normalize()
    return daily


def build_dataset(raw_dir: str | Path, interval: str = "1d") -> pd.DataFrame:
    """Возвращает таблицу: признаки (FEATURES), целевые y_1, y_3, y_7 и up_1, up_3, up_7."""
    raw = Path(raw_dir)
    sol = _load_candles(raw, f"sol_futures_{interval}.csv")
    btc = _load_candles(raw, f"btc_futures_{interval}.csv")
    sol_spot = _load_candles(raw, f"sol_spot_{interval}.csv")
    fund_sol = _load_funding_daily(raw, "sol_funding.csv")
    fund_btc = _load_funding_daily(raw, "btc_funding.csv")

    idx = sol.index
    btc = btc.reindex(idx)
    sol_spot = sol_spot.reindex(idx)

    logp = np.log(sol["close"])
    r = logp.diff()                                   # доходность SOL за день t
    r_btc = np.log(btc["close"]).diff()

    f = pd.DataFrame(index=idx)

    # 1. Прошлые доходности (лаги): r_lag0 — доходность самого дня t
    for k in range(5):
        f[f"r_lag{k}"] = r.shift(k)
    for k in range(3):
        f[f"r_btc_lag{k}"] = r_btc.shift(k)

    # 2. Импульс: доходность за 7 и 30 дней, отклонение от средней за 50 дней
    f["mom_7"] = logp - logp.shift(7)
    f["mom_30"] = logp - logp.shift(30)
    f["dist_ma50"] = logp - np.log(sol["close"].rolling(50).mean())
    f["btc_mom_7"] = np.log(btc["close"]).diff(7)

    # 3. Волатильность и размах свечи
    f["vol_7"] = r.rolling(7).std()
    f["vol_30"] = r.rolling(30).std()
    f["vol_ratio"] = f["vol_7"] / f["vol_30"]
    f["range"] = (sol["high"] - sol["low"]) / sol["close"]
    f["range_7"] = f["range"].rolling(7).mean()
    f["btc_vol_30"] = r_btc.rolling(30).std()

    # 4. Объём и доля агрессивных покупок
    qv = sol["quote_volume"]
    f["volume_z"] = np.log(qv) - np.log(qv).rolling(30).mean()
    if "taker_buy_quote" in sol.columns:
        f["taker_buy_share"] = sol["taker_buy_quote"] / qv
        f["taker_buy_share_7"] = f["taker_buy_share"].rolling(7).mean()

    # 5. Связь с BTC
    f["corr_btc_30"] = r.rolling(30).corr(r_btc)
    f["corr_btc_90"] = r.rolling(90).corr(r_btc)
    f["beta_btc_90"] = r.rolling(90).cov(r_btc) / r_btc.rolling(90).var()

    # 6. Funding и базис (данные, которые есть только у фьючерсов)
    fs = fund_sol.reindex(idx)
    f["funding"] = fs
    f["funding_7"] = fs.rolling(7).mean()
    f["funding_change"] = fs - f["funding_7"]
    f["funding_btc_7"] = fund_btc.reindex(idx).rolling(7).mean()
    basis = sol["close"] / sol_spot["close"] - 1
    f["basis"] = basis
    f["basis_7"] = basis.rolling(7).mean()

    # 7. Календарь: будет ли следующий день выходным (известно заранее)
    f["next_is_weekend"] = ((idx + pd.Timedelta(days=1)).day_of_week >= 5).astype(int)

    # Целевые переменные
    for h in HORIZONS:
        f[f"y_{h}"] = logp.shift(-h) - logp
        f[f"up_{h}"] = (f[f"y_{h}"] > 0).astype(float).where(f[f"y_{h}"].notna())

    # Дополнительная цель: реализованная волатильность SOL за следующие 7 дней
    f["rv_7"] = np.sqrt((r ** 2).rolling(7).mean().shift(-7))

    # Убираем разогрев (первые ~90 дней без скользящих окон); строки без таргета
    # в конце оставляем — они нужны для прогноза «на завтра».
    return f.dropna(subset=FEATURES_BASE + (["taker_buy_share_7"] if "taker_buy_share_7" in f else []))


FEATURES_BASE = [
    "r_lag0", "r_lag1", "r_lag2", "r_lag3", "r_lag4",
    "r_btc_lag0", "r_btc_lag1", "r_btc_lag2",
    "mom_7", "mom_30", "dist_ma50", "btc_mom_7",
    "vol_7", "vol_30", "vol_ratio", "range", "range_7", "btc_vol_30",
    "volume_z",
    "corr_btc_30", "corr_btc_90", "beta_btc_90",
    "funding", "funding_7", "funding_change", "funding_btc_7",
    "basis", "basis_7",
    "next_is_weekend",
]
FEATURES = FEATURES_BASE + ["taker_buy_share", "taker_buy_share_7"]


def feature_columns(data: pd.DataFrame) -> list[str]:
    """Признаки, которые есть в таблице (у Bybit нет taker_buy_* — они пропускаются)."""
    return [c for c in FEATURES if c in data.columns]
