"""HV regret, preference selection regret, bootstrap summaries."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from pareto import hypervolume_2d, normalize_cost, utility


def preference_winner(
    ids: Sequence[str],
    accuracies: dict[str, float],
    costs: dict[str, float],
    lam: float,
    cost_min: float,
    cost_max: float,
) -> str:
    best = None
    best_key = None
    for pid in ids:
        u = utility(accuracies[pid], normalize_cost(costs[pid], cost_min, cost_max), lam)
        key = (-u, pid)
        if best_key is None or key < best_key:
            best_key = key
            best = pid
    assert best is not None
    return best


def preference_selection_regret(
    winner_dev: str,
    accuracies_test: dict[str, float],
    costs_test: dict[str, float],
    lam: float,
    cost_min: float,
    cost_max: float,
) -> float:
    ids = list(accuracies_test.keys())
    oracle = preference_winner(ids, accuracies_test, costs_test, lam, cost_min, cost_max)
    u_oracle = utility(
        accuracies_test[oracle],
        normalize_cost(costs_test[oracle], cost_min, cost_max),
        lam,
    )
    u_dev = utility(
        accuracies_test[winner_dev],
        normalize_cost(costs_test[winner_dev], cost_min, cost_max),
        lam,
    )
    return float(u_oracle - u_dev)


def hv_regret(
    acc_dev: dict[str, float],
    cost_dev: dict[str, float],
    acc_test: dict[str, float],
    cost_test: dict[str, float],
    ref_a: float,
    ref_c_norm: float,
    cost_min: float,
    cost_max: float,
    selected_ids: Sequence[str] | None = None,
) -> float:
    ids = list(selected_ids) if selected_ids is not None else list(acc_dev.keys())
    pts_est = [
        (acc_dev[i], normalize_cost(cost_dev[i], cost_min, cost_max)) for i in ids
    ]
    pts_true = [
        (acc_test[i], normalize_cost(cost_test[i], cost_min, cost_max))
        for i in acc_test
    ]
    hv_true = hypervolume_2d(pts_true, ref_a, ref_c_norm)
    hv_est_front_on_test = hypervolume_2d(
        [
            (acc_test[i], normalize_cost(cost_test[i], cost_min, cost_max))
            for i in ids
        ],
        ref_a,
        ref_c_norm,
    )
    # Recovery quality: HV of the estimated-selected set evaluated on test
    # vs full test-oracle HV. `pts_est` is unused except documenting the front.
    _ = pts_est
    return float(hv_true - hv_est_front_on_test)


def bootstrap_mean_ci(values: Sequence[float], rng: np.random.Generator, n_boot: int = 1000) -> dict:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "sd": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, arr.size, arr.size)
        boots.append(float(arr[idx].mean()))
    boots_a = np.sort(np.asarray(boots))
    lo = float(np.quantile(boots_a, 0.025))
    hi = float(np.quantile(boots_a, 0.975))
    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "ci_lo": lo,
        "ci_hi": hi,
    }


def paired_bootstrap_reversal_rate(
    z_p: np.ndarray,
    z_q: np.ndarray,
    c_norm_p: float,
    c_norm_q: float,
    rng: np.random.Generator,
    n_boot: int,
    pid: str,
    qid: str,
    lam: float = 0.5,
) -> float:
    """Heuristic reversal rate of (preference at λ, Pareto relation). Not a CI."""
    n = int(z_p.size)
    if n == 0:
        return 1.0
    current = _pair_decision(z_p, z_q, c_norm_p, c_norm_q, pid, qid, lam)
    # Degenerate inputs: bootstrap cannot flip a constant paired difference.
    if n == 1 or np.all(z_p == z_q):
        return 0.0
    # Vectorized paired bootstrap (costs fixed; accuracy re-sampled with replacement).
    idx = rng.integers(0, n, size=(n_boot, n))
    a_p = z_p[idx].mean(axis=1)
    a_q = z_q[idx].mean(axis=1)
    u_p = lam * a_p - (1.0 - lam) * c_norm_p
    u_q = lam * a_q - (1.0 - lam) * c_norm_q
    # Winner by key (-u, id): smaller key wins.
    pref_is_p = (u_p > u_q) | ((u_p == u_q) & (pid < qid))
    cur_pref_p = current[0] == pid
    p_dom = (a_p >= a_q) & (c_norm_p <= c_norm_q) & ((a_p > a_q) | (c_norm_p < c_norm_q))
    q_dom = (a_q >= a_p) & (c_norm_q <= c_norm_p) & ((a_q > a_p) | (c_norm_q < c_norm_p))
    dom = np.where(p_dom, 1, np.where(q_dom, -1, 0))
    flips = int(((pref_is_p != cur_pref_p) | (dom != current[1])).sum())
    return float(flips / n_boot)


def _pair_decision(
    z_p: np.ndarray,
    z_q: np.ndarray,
    c_p: float,
    c_q: float,
    pid: str,
    qid: str,
    lam: float,
) -> tuple:
    a_p = float(np.mean(z_p))
    a_q = float(np.mean(z_q))
    u_p = utility(a_p, c_p, lam)
    u_q = utility(a_q, c_q, lam)
    pref = pid if (-u_p, pid) < (-u_q, qid) else qid
    p_dom = (a_p >= a_q and c_p <= c_q) and (a_p > a_q or c_p < c_q)
    q_dom = (a_q >= a_p and c_q <= c_p) and (a_q > a_p or c_q < c_p)
    if p_dom:
        dom = 1
    elif q_dom:
        dom = -1
    else:
        dom = 0
    return (pref, dom)
