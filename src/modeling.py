"""Честная проверка моделей на временных рядах: walk-forward, бейзлайны, метрики (чекпойнт 3).

Главное правило: модель, которая делает прогноз на закрытии дня t, обучена только на
строках s, чей таргет уже известен к этому моменту, то есть s + h <= t («эмбарго» h дней).
Модель переобучается каждые refit_every дней на всей доступной к тому моменту истории
(расширяющееся окно).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit


# ---------------------------------------------------------------- бейзлайны

class ZeroModel(BaseEstimator, RegressorMixin):
    """«Доходность будет нулевой» — ничего не предсказывает."""
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


class MeanModel(BaseEstimator, RegressorMixin):
    """Средняя доходность обучающей выборки (учитывает общий дрейф цены)."""
    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


class PersistenceModel(BaseEstimator, RegressorMixin):
    """«Завтра как сегодня»: прогноз доходности за следующие h дней = доходность за прошлые h дней."""
    def __init__(self, horizon: int = 1):
        self.horizon = horizon

    def fit(self, X, y):
        return self

    def predict(self, X):
        if self.horizon == 7:
            return X["mom_7"].to_numpy()
        return X[[f"r_lag{k}" for k in range(self.horizon)]].sum(axis=1).to_numpy()


# ---------------------------------------------------------------- walk-forward

def walk_forward(model, X: pd.DataFrame, y: pd.Series, test_start: str, horizon: int,
                 refit_every: int = 30, proba: bool = False) -> pd.Series:
    """Прогнозы на тестовом периоде с переобучением каждые refit_every дней.

    Для прогноза на дату t модель обучена на строках s <= t - horizon дней (таргет уже известен).
    proba=True — для классификаторов возвращается вероятность роста.
    """
    test_dates = X.index[(X.index >= pd.Timestamp(test_start, tz="UTC")) & y.notna()]
    preds = pd.Series(np.nan, index=test_dates, dtype=float)
    for i in range(0, len(test_dates), refit_every):
        block = test_dates[i:i + refit_every]
        t0 = block[0]
        train = (X.index <= t0 - pd.Timedelta(days=horizon)) & y.notna()
        m = clone(model).fit(X.loc[train], y.loc[train])
        preds.loc[block] = m.predict_proba(X.loc[block])[:, 1] if proba else m.predict(X.loc[block])
    return preds


def time_series_cv(n_splits: int, horizon: int) -> TimeSeriesSplit:
    """Кросс-валидация для подбора гиперпараметров: фолды идут по времени, между
    обучением и проверкой зазор horizon дней, чтобы таргеты не перекрывались."""
    return TimeSeriesSplit(n_splits=n_splits, gap=horizon)


# ---------------------------------------------------------------- метрики

def _binom_p(hits: int, n: int) -> float:
    return stats.binomtest(hits, n, 0.5, alternative="greater").pvalue if n else np.nan


def regression_metrics(y_true: pd.Series, y_pred: pd.Series, horizon: int) -> dict:
    """MAE, RMSE, R² вне выборки относительно нулевого прогноза и точность направления.

    R²oos = 1 − SSE(модель) / SSE(прогноз 0): > 0 — модель лучше, чем «ничего не предсказывать».
    Точность направления считается только там, где прогноз ≠ 0. Для h > 1 соседние таргеты
    перекрываются, поэтому p-value биномиального теста считаем по каждому h-му дню.
    """
    y_true, y_pred = y_true.loc[y_pred.index], y_pred
    nz = y_pred != 0
    hit = (np.sign(y_pred[nz]) == np.sign(y_true[nz]))
    sub = hit.iloc[::horizon]
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2_oos": 1 - float(np.sum((y_true - y_pred) ** 2) / np.sum(y_true ** 2)),
        "точность_направления": float(hit.mean()) if nz.any() else np.nan,
        "p_value_направление": _binom_p(int(sub.sum()), len(sub)) if nz.any() else np.nan,
    }


def classification_metrics(up_true: pd.Series, proba: pd.Series, horizon: int) -> dict:
    """Точность (порог 0.5), ROC AUC и p-value: лучше ли точность, чем подбрасывание монетки."""
    up_true = up_true.loc[proba.index]
    hit = ((proba > 0.5).astype(int) == up_true)
    sub = hit.iloc[::horizon]
    return {
        "точность": float(hit.mean()),
        "ROC_AUC": roc_auc_score(up_true, proba),
        "доля_прогнозов_роста": float((proba > 0.5).mean()),
        "p_value_точность": _binom_p(int(sub.sum()), len(sub)),
    }


@dataclass
class Experiment:
    """Результаты одной модели на одном горизонте."""
    name: str
    horizon: int
    preds: pd.Series
    metrics: dict
    params: dict | None = None
