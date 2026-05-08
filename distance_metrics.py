"""
Метрики близости для рекомендательных систем.

Все функции возвращают матрицу `scores` формы (n_users, n_items),
где **большее значение = ближе** — единое соглашение, чтобы поверх можно
было одинаково делать `argpartition(-scores, K)` для top-K.

Для дистанций (L1/L2/L∞/Mahalanobis/Minkowski) возвращается -distance,
для сходств (cosine/dot) — само значение.

Использование:
    from utils.distance_metrics import score_all, evaluate_all

    P = user_emb.values  # (n_users, n_features)
    M = all_songs.values # (n_items, n_features)

    scores = score_all(P, M)              # dict: {'l2': ..., 'l1': ..., 'cos': ...}
    results = evaluate_all(P, M, song_ids, train_seen, test_truth, K=10)
"""

from __future__ import annotations

import math
import time
from typing import Callable

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity


def _log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------- метрики -----------------------------

def score_dot(P: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Скалярное произведение: больше = ближе."""
    return P @ M.T


def score_cosine(P: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Косинус: 1 = идентичны, -1 = противоположны."""
    return cosine_similarity(P, M)


def score_l2(P: np.ndarray, M: np.ndarray) -> np.ndarray:
    """-евклидово расстояние."""
    return -cdist(P, M, metric="euclidean")


def score_l1(P: np.ndarray, M: np.ndarray) -> np.ndarray:
    """-манхэттен."""
    return -cdist(P, M, metric="cityblock")


def score_chebyshev(P: np.ndarray, M: np.ndarray) -> np.ndarray:
    """-Чебышёв (L∞)."""
    return -cdist(P, M, metric="chebyshev")

METRICS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "dot": score_dot,
    "cos": score_cosine,
    "l2": score_l2,
    "l1": score_l1,
    "linf": score_chebyshev
}


def score_all(P: np.ndarray, M: np.ndarray) -> dict[str, np.ndarray]:
    """Считает scores сразу для всех метрик из реестра."""
    return {name: fn(P, M) for name, fn in METRICS.items()}


# ----------------------------- метрики качества -----------------------------

def _ndcg_at_k(ranked: list, truth: set, k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, s in enumerate(ranked[:k]) if s in truth)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(truth), k)))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_scores(
    scores: np.ndarray,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    train_seen: dict[str, set],
    test_truth: dict[str, set],
    K: int = 10,
    verbose: bool = True,
    log_every: int = 5000,
) -> dict[str, float]:
    """Считает Precision@K, Recall@K, NDCG@K на матрице scores.

    user_ids[i] соответствует строке scores[i],
    item_ids[j] соответствует столбцу scores[:, j].
    """
    item_to_col = {s: i for i, s in enumerate(item_ids)}

    if verbose:
        _log(f"  copy scores ({scores.shape[0]}x{scores.shape[1]}) -> float64")
    scores = scores.copy().astype(np.float64)

    if verbose:
        _log("  masking train interactions...")
    t0 = time.time()
    for i, u in enumerate(user_ids):
        seen = train_seen.get(u, set())
        if seen:
            cols = [item_to_col[s] for s in seen if s in item_to_col]
            if cols:
                scores[i, cols] = -np.inf

    # NaN могут возникать в Mahalanobis при сингулярной ковариации
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=-np.inf)
    if verbose:
        _log(f"  masking done in {time.time()-t0:.1f}s, computing top-K + metrics...")

    precisions, recalls, ndcgs = [], [], []
    n = scores.shape[0]
    t0 = time.time()
    for i in range(n):
        u = user_ids[i]
        truth = test_truth.get(u, set())
        if not truth:
            continue
        row = scores[i]
        top = np.argpartition(-row, K)[:K]
        top = top[np.argsort(-row[top])]
        ranked = [item_ids[j] for j in top]
        hits = sum(1 for s in ranked if s in truth)
        precisions.append(hits / K)
        recalls.append(hits / len(truth))
        ndcgs.append(_ndcg_at_k(ranked, truth, K))

        if verbose and (i + 1) % log_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed else 0
            eta = (n - i - 1) / rate if rate else 0
            _log(f"    {i+1}/{n} users  rate={rate:.0f}/s  eta={eta:.0f}s")

    if verbose:
        _log(f"  done: {len(precisions)} users evaluated in {time.time()-t0:.1f}s")

    return {
        f"precision@{K}": float(np.mean(precisions)) if precisions else 0.0,
        f"recall@{K}": float(np.mean(recalls)) if recalls else 0.0,
        f"ndcg@{K}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "users_evaluated": len(precisions),
    }


def evaluate_all(
    P: np.ndarray,
    M: np.ndarray,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    train_seen: dict[str, set],
    test_truth: dict[str, set],
    K: int = 10,
    metrics: dict[str, Callable] | None = None,
    verbose: bool = True,
) -> dict[str, dict[str, float]]:
    """Прогоняет все метрики из METRICS на (P, M) и считает качество.

    Возвращает: {metric_name: {'precision@K': ..., 'recall@K': ..., 'ndcg@K': ...}}
    """
    metrics = metrics or METRICS
    out = {}
    total = len(metrics)
    t_total = time.time()
    for idx, (name, fn) in enumerate(metrics.items(), 1):
        if verbose:
            _log(f"[{idx}/{total}] metric '{name}': computing scores...")
        t0 = time.time()
        scores = fn(P, M)
        if verbose:
            _log(f"  scores computed in {time.time()-t0:.1f}s, shape={scores.shape}")
        out[name] = evaluate_scores(
            scores, user_ids, item_ids, train_seen, test_truth, K=K, verbose=verbose
        )
        if verbose:
            r = out[name]
            _log(f"  '{name}' result: P@{K}={r[f'precision@{K}']:.4f} "
                 f"R@{K}={r[f'recall@{K}']:.4f} NDCG@{K}={r[f'ndcg@{K}']:.4f}")
    if verbose:
        _log(f"ALL DONE in {time.time()-t_total:.1f}s")
    return out


# ----------------------------- утилиты -----------------------------

def to_dataframe(results: dict[str, dict[str, float]]):
    """Превращает результат evaluate_all в красивый pandas.DataFrame."""
    import pandas as pd
    return pd.DataFrame(results).T


def distance_concentration(M: np.ndarray, sample: int = 1000) -> dict:
    """Эмпирическая мера концентрации расстояний в HD-пространстве.

    Вычисляет (max - min) / mean для попарных L2-расстояний на случайной
    подвыборке из M. Чем ближе к 0 — тем сильнее проклятие размерности.
    """
    if len(M) > sample:
        idx = np.random.RandomState(42).choice(len(M), sample, replace=False)
        M = M[idx]
    D = cdist(M, M, metric="euclidean")
    np.fill_diagonal(D, np.nan)
    flat = D[~np.isnan(D)]
    return {
        "min": float(flat.min()),
        "max": float(flat.max()),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "contrast": float((flat.max() - flat.min()) / flat.mean()),
    }