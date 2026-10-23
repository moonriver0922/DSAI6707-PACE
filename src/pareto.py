"""Max-accuracy / min-cost Pareto, utility, and 2D hypervolume."""

from __future__ import annotations

from typing import Iterable, Sequence


def dominates_max_a_min_c(
    a1: float, c1: float, a2: float, c2: float
) -> bool:
    """True iff (a1,c1) Pareto-dominates (a2,c2) for max accuracy, min cost."""
    return (a1 >= a2 and c1 <= c2) and (a1 > a2 or c1 < c2)


def normalize_cost(cost: float, cost_min: float, cost_max: float) -> float:
    if cost_max <= cost_min:
        raise ValueError("cost_max must exceed cost_min")
    x = (cost - cost_min) / (cost_max - cost_min)
    return float(min(1.0, max(0.0, x)))


def utility(accuracy: float, cost_norm: float, lam: float) -> float:
    """U_λ = λ a - (1-λ) c̃. Range is [-(1-λ), λ]."""
    return float(lam * accuracy - (1.0 - lam) * cost_norm)


def nondominated_mask(accuracies: Sequence[float], costs: Sequence[float]) -> list[bool]:
    n = len(accuracies)
    out = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            if dominates_max_a_min_c(accuracies[j], costs[j], accuracies[i], costs[i]):
                dominated = True
                break
        out.append(not dominated)
    return out


def nondominated_ids(
    ids: Sequence[str], accuracies: dict[str, float], costs: dict[str, float]
) -> list[str]:
    acc = [accuracies[i] for i in ids]
    cost = [costs[i] for i in ids]
    mask = nondominated_mask(acc, cost)
    return [i for i, m in zip(ids, mask) if m]


def hypervolume_2d(
    points: Iterable[tuple[float, float]],
    ref_accuracy: float,
    ref_cost: float,
) -> float:
    """HV for max accuracy / min cost with a strictly worse reference point.

    `points` are (accuracy, cost). Reference must be worse: lower-or-equal
    accuracy and higher-or-equal cost than interesting points.
    """
    filtered = [
        (a, c)
        for a, c in points
        if a >= ref_accuracy and c <= ref_cost
    ]
    if not filtered:
        return 0.0
    # Keep only non-dominated
    keep = []
    for i, (a, c) in enumerate(filtered):
        if any(
            dominates_max_a_min_c(a2, c2, a, c)
            for j, (a2, c2) in enumerate(filtered)
            if i != j
        ):
            continue
        keep.append((a, c))
    keep.sort(key=lambda t: -t[0])  # accuracy descending
    hv = 0.0
    prev_c = ref_cost
    for a, c in keep:
        if c < prev_c:
            hv += (a - ref_accuracy) * (prev_c - c)
            prev_c = c
    return float(hv)


def crowding_distance(
    ids: Sequence[str], accuracies: dict[str, float], costs: dict[str, float]
) -> dict[str, float]:
    """Standard NSGA-II crowding on two objectives (max a, min c)."""
    dist = {i: 0.0 for i in ids}
    if len(ids) <= 2:
        for i in ids:
            dist[i] = float("inf")
        return dist
    for values, maximize in (
        (accuracies, True),
        (costs, False),
    ):
        ordered = sorted(ids, key=lambda i: values[i], reverse=maximize)
        dist[ordered[0]] = float("inf")
        dist[ordered[-1]] = float("inf")
        lo = values[ordered[-1]]
        hi = values[ordered[0]]
        span = abs(hi - lo)
        if span == 0:
            continue
        for k in range(1, len(ordered) - 1):
            dist[ordered[k]] += abs(values[ordered[k + 1]] - values[ordered[k - 1]]) / span
    return dist
