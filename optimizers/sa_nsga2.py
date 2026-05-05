"""Surrogate-assisted NSGA-II optimizer backed by real Backtrader evaluations."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from deap import base, creator, tools

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from backtests.run_backtest import run_walk_forward_backtest
from quant.features import build_feature_frame


SEARCH_SPACE = [
    ("tech_weight", 0.05, 0.65, "float"),
    ("vcp_weight", 0.00, 0.40, "float"),
    ("bb_weight", 0.00, 0.35, "float"),
    ("fund_weight", 0.00, 0.35, "float"),
    ("chip_weight", 0.00, 0.25, "float"),
    ("ml_weight", 0.00, 0.55, "float"),
    ("score_threshold", 0.45, 0.90, "float"),
    ("stop_loss_pct", 0.04, 0.12, "float"),
    ("risk_per_trade", 0.005, 0.030, "float"),
    ("exit_ma_period", 10, 60, "int"),
]

DEFAULT_CONFIG = {
    "tickers": ["2454.TW", "2330.TW", "AAPL", "NVDA"],
    "start": "2019-01-01",
    "end": None,
    "model_path": None,
    "population_size": 24,
    "generations": 6,
    "real_evals_per_gen": 8,
    "walk_forward_folds": 2,
    "initial_cash": 1_000_000.0,
    "commission": 0.002,
    "random_seed": 42,
    "output_path": "config/best_params.json",
}


if not hasattr(creator, "FitnessHybrid"):
    creator.create("FitnessHybrid", base.Fitness, weights=(1.0, 1.0, -1.0, -1.0))
if not hasattr(creator, "IndividualHybrid"):
    creator.create("IndividualHybrid", list, fitness=creator.FitnessHybrid)


def _random_gene(low: float, high: float, kind: str):
    if kind == "int":
        return random.randint(int(low), int(high))
    return random.uniform(float(low), float(high))


def _make_individual():
    return creator.IndividualHybrid([_random_gene(low, high, kind) for _, low, high, kind in SEARCH_SPACE])


def decode_individual(individual: list[float]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for value, (name, low, high, kind) in zip(individual, SEARCH_SPACE):
        clipped = min(max(float(value), float(low)), float(high))
        params[name] = int(round(clipped)) if kind == "int" else clipped
    return params


def encode_params(params: dict[str, Any]) -> list[float]:
    encoded = []
    for name, low, high, kind in SEARCH_SPACE:
        value = params.get(name, (low + high) / 2)
        encoded.append(float(min(max(float(value), float(low)), float(high))))
    return encoded


def _bounded_mutation(individual, indpb: float = 0.25):
    for idx, (_, low, high, kind) in enumerate(SEARCH_SPACE):
        if random.random() >= indpb:
            continue
        if kind == "int":
            individual[idx] = random.randint(int(low), int(high))
        else:
            span = high - low
            individual[idx] = min(max(individual[idx] + random.gauss(0, span * 0.12), low), high)
    return (individual,)


def _fitness_from_metrics(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(metrics.get("total_return_pct", -100.0)),
        float(metrics.get("risk_adjusted_return", metrics.get("sharpe_ratio", -5.0))),
        float(metrics.get("max_drawdown_pct", 100.0)),
        float(metrics.get("turnover", 1.0)),
    )


def _scalarize(fitness: tuple[float, float, float, float]) -> float:
    total_return, risk_adjusted, mdd, turnover = fitness
    return total_return + 12.0 * risk_adjusted - 1.5 * mdd - 100.0 * turnover


def _evaluate_real(individual, feature_frames, config: dict[str, Any]) -> dict[str, Any]:
    params = decode_individual(individual)
    metrics = run_walk_forward_backtest(
        feature_frames,
        {**params, "walk_forward_folds": int(config.get("walk_forward_folds", 1))},
        initial_cash=float(config.get("initial_cash", 1_000_000.0)),
        commission=float(config.get("commission", 0.002)),
    )
    fitness = _fitness_from_metrics(metrics)
    individual.fitness.values = fitness
    individual.is_surrogate = False
    return {
        "params": params,
        "genes": list(map(float, individual)),
        "fitness": fitness,
        "metrics": metrics,
        "surrogate": False,
    }


def _normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _expected_improvement(mu: float, sigma: float, best: float) -> float:
    if sigma <= 1e-9:
        return max(0.0, mu - best)
    z = (mu - best) / sigma
    return (mu - best) * _normal_cdf(z) + sigma * _normal_pdf(z)


def _fit_surrogates(archive: list[dict[str, Any]]):
    if len(archive) < 6:
        return None
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return None

    x = np.asarray([item["genes"] for item in archive], dtype=float)
    y = np.asarray([item["fitness"] for item in archive], dtype=float)
    x_scaler = StandardScaler()
    x_scaled = x_scaler.fit_transform(x)
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(nu=2.5) + WhiteKernel(
        noise_level=1e-4,
        noise_level_bounds=(1e-8, 1e1),
    )
    models = []
    for objective_idx in range(y.shape[1]):
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            random_state=17,
            n_restarts_optimizer=0,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model.fit(x_scaled, y[:, objective_idx])
        models.append(model)
    return x_scaler, models


def _predict_fitness(surrogates, individual) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    x_scaler, models = surrogates
    x_scaled = x_scaler.transform(np.asarray([list(map(float, individual))], dtype=float))
    means = []
    stds = []
    for model in models:
        mu, sigma = model.predict(x_scaled, return_std=True)
        means.append(float(mu[0]))
        stds.append(float(sigma[0]))
    return tuple(means), tuple(stds)


def _build_feature_frames(config: dict[str, Any]) -> dict[str, Any]:
    frames = {}
    weights = {name: config.get(name) for name, _, _, _ in SEARCH_SPACE if name.endswith("_weight")}
    for ticker in config.get("tickers", DEFAULT_CONFIG["tickers"]):
        frame = build_feature_frame(
            ticker,
            start=config.get("start"),
            end=config.get("end"),
            weights=weights,
            model_path=config.get("model_path"),
        )
        frames[ticker] = frame
    return frames


def _record_to_individual(record: dict[str, Any]):
    ind = creator.IndividualHybrid(record["genes"])
    ind.fitness.values = tuple(record["fitness"])
    return ind


def _pareto_records(archive: list[dict[str, Any]]) -> list[dict[str, Any]]:
    inds = [_record_to_individual(record) for record in archive]
    front = tools.sortNondominated(inds, len(inds), first_front_only=True)[0]
    front_genes = {tuple(round(x, 10) for x in ind) for ind in front}
    records = [record for record in archive if tuple(round(x, 10) for x in record["genes"]) in front_genes]
    records.sort(key=lambda item: (item["fitness"][2], -item["fitness"][1], -item["fitness"][0]))
    return records


def _minmax(values: list[float], invert: bool = False) -> list[float]:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return []
    if np.nanmax(arr) - np.nanmin(arr) < 1e-9:
        scaled = np.ones_like(arr) * 0.5
    else:
        scaled = (arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr))
    if invert:
        scaled = 1.0 - scaled
    return scaled.tolist()


def _choose_balanced(records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(records) == 1:
        return records[0]
    returns = _minmax([r["fitness"][0] for r in records])
    risk_adj = _minmax([r["fitness"][1] for r in records])
    drawdown = _minmax([r["fitness"][2] for r in records], invert=True)
    turnover = _minmax([r["fitness"][3] for r in records], invert=True)
    best_idx = 0
    best_score = -float("inf")
    for idx in range(len(records)):
        score = 0.25 * returns[idx] + 0.40 * risk_adj[idx] + 0.25 * drawdown[idx] + 0.10 * turnover[idx]
        if score > best_score:
            best_idx = idx
            best_score = score
    records[best_idx]["balanced_selection_score"] = best_score
    return records[best_idx]


def _json_safe_record(record: dict[str, Any]) -> dict[str, Any]:
    metrics = record.get("metrics", {}).copy()
    metrics.pop("runs", None)
    return {
        "params": record["params"],
        "fitness": list(map(float, record["fitness"])),
        "metrics": metrics,
        "balanced_selection_score": float(record.get("balanced_selection_score", 0.0)),
    }


def _write_best_params(best: dict[str, Any], pareto: list[dict[str, Any]], config: dict[str, Any]) -> None:
    params = best["params"].copy()
    payload = {
        **params,
        "hard_stop": params["stop_loss_pct"],
        "ma_period": params["exit_ma_period"],
        "model_path": config.get("model_path"),
        "optimizer": {
            "name": "sa_nsga2",
            "selection": "balanced",
            "objectives": ["total_return_pct", "risk_adjusted_return", "max_drawdown_pct", "turnover"],
            "best_metrics": _json_safe_record(best)["metrics"],
            "pareto_front": [_json_safe_record(record) for record in pareto],
        },
    }
    output_path = config.get("output_path", DEFAULT_CONFIG["output_path"])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run_sa_nsga2(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run surrogate-assisted NSGA-II and persist the balanced Pareto choice."""
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)

    seed = int(cfg.get("random_seed", 42))
    random.seed(seed)
    np.random.seed(seed)

    population_size = int(cfg.get("population_size", 24))
    if population_size % 4:
        population_size += 4 - population_size % 4
    generations = int(cfg.get("generations", 6))
    real_evals_per_gen = max(1, int(cfg.get("real_evals_per_gen", max(4, population_size // 3))))

    feature_frames = cfg.get("feature_frames") or _build_feature_frames(cfg)

    toolbox = base.Toolbox()
    toolbox.register("individual", _make_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxBlend, alpha=0.4)
    toolbox.register("mutate", _bounded_mutation, indpb=0.25)
    toolbox.register("select", tools.selNSGA2)

    population = toolbox.population(n=population_size)
    archive: list[dict[str, Any]] = []
    for individual in population:
        archive.append(_evaluate_real(individual, feature_frames, cfg))
    population = toolbox.select(population, len(population))

    for generation in range(1, generations + 1):
        offspring = tools.selTournamentDCD(population, len(population))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for first, second in zip(offspring[::2], offspring[1::2]):
            if random.random() <= 0.70:
                toolbox.mate(first, second)
                del first.fitness.values
                del second.fitness.values
        for individual in offspring:
            if random.random() <= 0.30:
                toolbox.mutate(individual)
                if individual.fitness.valid:
                    del individual.fitness.values

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        surrogates = _fit_surrogates(archive)
        if surrogates is None:
            selected_for_real = invalid
            surrogate_only = []
        else:
            best_scalar = max(_scalarize(tuple(record["fitness"])) for record in archive)
            ranked = []
            for ind in invalid:
                predicted, uncertainty = _predict_fitness(surrogates, ind)
                sigma = float(np.linalg.norm(np.asarray(uncertainty, dtype=float)))
                ranked.append((_expected_improvement(_scalarize(predicted), sigma, best_scalar), ind, predicted))
            ranked.sort(key=lambda item: item[0], reverse=True)
            selected_for_real = [item[1] for item in ranked[:real_evals_per_gen]]
            surrogate_only = ranked[real_evals_per_gen:]

        for individual in selected_for_real:
            archive.append(_evaluate_real(individual, feature_frames, cfg))
        for _, individual, predicted in surrogate_only:
            individual.fitness.values = predicted
            individual.is_surrogate = True

        population = toolbox.select(population + offspring, population_size)
        print(
            f"generation {generation}/{generations}: "
            f"real_archive={len(archive)} pareto={len(_pareto_records(archive))}"
        )

    pareto = _pareto_records(archive)
    best = _choose_balanced(pareto)
    _write_best_params(best, pareto, cfg)

    return {
        "best": _json_safe_record(best),
        "pareto_front": [_json_safe_record(record) for record in pareto],
        "archive_size": len(archive),
        "output_path": cfg.get("output_path", DEFAULT_CONFIG["output_path"]),
    }


def _load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run surrogate-assisted NSGA-II optimization.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--population-size", type=int, default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--real-evals-per-gen", type=int, default=None)
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--output-path", default=None)
    args = parser.parse_args()

    config = _load_config(args.config)
    for key, value in {
        "tickers": args.tickers,
        "start": args.start,
        "end": args.end,
        "model_path": args.model_path,
        "population_size": args.population_size,
        "generations": args.generations,
        "real_evals_per_gen": args.real_evals_per_gen,
        "walk_forward_folds": args.folds,
        "output_path": args.output_path,
    }.items():
        if value is not None:
            config[key] = value

    result = run_sa_nsga2(config)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
