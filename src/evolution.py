"""Small-scale NSGA-II with controlled label-preserving prompt mutations (Gate 2)."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from pareto import crowding_distance, nondominated_ids
from state import SearchState


ROLE_PREFIXES = [
    "You are a careful news librarian. ",
    "You are a professional newspaper editor. ",
    "You are a wire-service copy taster. ",
    "You are an expert annotator for AG News. ",
]

FORMAT_SUFFIXES = [
    "\nReply with exactly one label. No extra words.",
    "\nOutput only the class name.",
    "\nAnswer as NAME only:",
    "\nLabel:",
]

DEFINITION_BLURBS = [
    "World=geopolitics; Sports=athletics; Business=markets; Sci/Tech=science/tech. ",
    "Use Sci/Tech only for science or technology. ",
    "Valid labels: World, Sports, Business, Sci/Tech. ",
]


def _ensure_placeholder(text: str) -> str:
    if "{text}" not in text:
        return text.rstrip() + "\nArticle: {text}\nLabel:"
    return text


def mutate_prompt(text: str, rng: np.random.Generator) -> str:
    """Rule-based, label-set-preserving mutation. No LLM calls."""
    t = text
    op = int(rng.integers(0, 5))
    if op == 0:
        # Swap / insert a role prefix
        for pref in ROLE_PREFIXES:
            if t.startswith(pref):
                t = t[len(pref) :]
                break
        t = ROLE_PREFIXES[int(rng.integers(0, len(ROLE_PREFIXES)))] + t
    elif op == 1:
        # Add or replace a short format constraint
        t = re.sub(r"\n(Reply with|Output only|Answer as|Label:).*$", "", t, flags=re.S)
        t = t.rstrip() + FORMAT_SUFFIXES[int(rng.integers(0, len(FORMAT_SUFFIXES)))]
    elif op == 2:
        # Insert a definition blurb near the top
        blurb = DEFINITION_BLURBS[int(rng.integers(0, len(DEFINITION_BLURBS)))]
        if "World" in t and "Sports" in t:
            t = blurb + t
        else:
            t = blurb + t
    elif op == 3:
        # Mild wording tweak
        t = t.replace("Article:", "News:", 1) if "Article:" in t else t.replace("News:", "Article:", 1)
        if t == text:
            t = t.replace("Classify", "Categorize", 1) if "Classify" in t else "Classify. " + t
    else:
        # Shorten: keep first and last sentence-ish chunks
        parts = [p for p in re.split(r"\n+", t) if p.strip()]
        if len(parts) >= 3:
            t = "\n".join([parts[0], "{text}" if "{text}" not in parts[0] else parts[1], parts[-1]])
            if "{text}" not in t:
                t = parts[0] + "\n{text}\n" + parts[-1]
        else:
            t = ROLE_PREFIXES[0] + t
    return _ensure_placeholder(t)


def crossover_prompts(a: str, b: str, rng: np.random.Generator) -> str:
    """Swap the instruction head of A with the format tail of B."""
    a_parts = a.split("{text}")
    b_parts = b.split("{text}")
    if len(a_parts) >= 2 and len(b_parts) >= 2:
        if rng.random() < 0.5:
            child = a_parts[0] + "{text}" + b_parts[-1]
        else:
            child = b_parts[0] + "{text}" + a_parts[-1]
    else:
        child = a if rng.random() < 0.5 else b
    return _ensure_placeholder(child)


@dataclass
class Individual:
    prompt_id: str
    text: str
    generation: int = 0
    parent_ids: tuple[str, ...] = ()


def nsga2_select(
    ids: list[str],
    accuracies: dict[str, float],
    costs: dict[str, float],
    k: int,
) -> list[str]:
    """Select k individuals via non-dominated sorting + crowding."""
    remaining = list(ids)
    selected: list[str] = []
    while remaining and len(selected) < k:
        front = nondominated_ids(remaining, accuracies, costs)
        if len(selected) + len(front) <= k:
            selected.extend(front)
            remaining = [i for i in remaining if i not in front]
            continue
        crowd = crowding_distance(front, accuracies, costs)
        front_sorted = sorted(front, key=lambda i: (-crowd[i], i))
        need = k - len(selected)
        selected.extend(front_sorted[:need])
        break
    return selected


def generate_offspring(
    parents: list[Individual],
    rng: np.random.Generator,
    n_children: int,
    id_prefix: str,
    gen: int,
) -> list[Individual]:
    children: list[Individual] = []
    for i in range(n_children):
        p1 = parents[int(rng.integers(0, len(parents)))]
        p2 = parents[int(rng.integers(0, len(parents)))]
        if rng.random() < 0.5:
            text = crossover_prompts(p1.text, p2.text, rng)
        else:
            text = p1.text
        text = mutate_prompt(text, rng)
        cid = f"{id_prefix}_g{gen}_c{i:02d}"
        children.append(
            Individual(prompt_id=cid, text=text, generation=gen, parent_ids=(p1.prompt_id, p2.prompt_id))
        )
    return children


@dataclass
class EvolutionConfig:
    population_size: int = 8
    n_generations: int = 4
    mutation_rate: float = 1.0  # always mutate after crossover in MVP


def run_nsga2_search(
    initial: list[Individual],
    evaluate_population: Callable[[list[Individual]], tuple[dict[str, float], dict[str, float]]],
    cfg: EvolutionConfig,
    seed: int,
) -> dict:
    """
    Shared evolutionary loop. `evaluate_population` must enforce the search
    evaluation budget externally (Uniform / Racing / DICE wrappers).
    """
    rng = np.random.default_rng(seed)
    pop = list(initial)[: cfg.population_size]
    history = []
    for gen in range(cfg.n_generations):
        acc, cost = evaluate_population(pop)
        ids = [ind.prompt_id for ind in pop]
        selected_ids = nsga2_select(ids, acc, cost, cfg.population_size)
        survivors = [ind for ind in pop if ind.prompt_id in selected_ids]
        # Keep stable order by selection rank
        order = {pid: i for i, pid in enumerate(selected_ids)}
        survivors.sort(key=lambda ind: order[ind.prompt_id])
        history.append(
            {
                "generation": gen,
                "population": [ind.prompt_id for ind in survivors],
                "accuracies": {i: acc[i] for i in selected_ids if i in acc},
                "costs": {i: cost[i] for i in selected_ids if i in cost},
            }
        )
        if gen == cfg.n_generations - 1:
            pop = survivors
            break
        children = generate_offspring(
            survivors, rng, cfg.population_size, id_prefix="evo", gen=gen + 1
        )
        # Evaluate combined pool then truncate (standard μ+λ style)
        combined = survivors + children
        acc2, cost2 = evaluate_population(combined)
        cids = [ind.prompt_id for ind in combined]
        keep = nsga2_select(cids, acc2, cost2, cfg.population_size)
        pop = [ind for ind in combined if ind.prompt_id in keep]
        order = {pid: i for i, pid in enumerate(keep)}
        pop.sort(key=lambda ind: order[ind.prompt_id])
    final_acc, final_cost = evaluate_population(pop)
    return {
        "final_population": pop,
        "accuracies": final_acc,
        "costs": final_cost,
        "history": history,
    }
