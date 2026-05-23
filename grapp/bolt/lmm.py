"""BOLT-LMM-inf statistical functions and top-level driver."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from grapp.grg_calculator import GRGCalcInterface, _wrap_grg
from grapp.bolt.lmm_core import (
    DTYPE,
    BOLT_RANDOM_SEED,
    DEFAULT_NUM_CALIB_SNPS,
    DEFAULT_H2_EST_MC_TRIALS,
    DEFAULT_CG_TOL,
    DEFAULT_MAX_ITERS,
    _as_float,
    _array_module,
    _dot,
    BoostMt19937,
    BoostNormalDistribution,
    boost_uniform_int_0_2pow30,
    bolt_conj_grad_solve,
    BoltVariantStats,
    BoltLmmOps,
    CalibrationResult,
    CgStats,
    CovariateBasis,
    McScalingResult,
    VarianceFit,
    compute_bolt_variant_stats,
    compute_lmm_inf_stats,
    lmm_inf_stats_to_dataframe,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Variance component helpers
# ---------------------------------------------------------------------------

def log_delta_from_h2(ops: BoltLmmOps, h2: float) -> float:
    return math.log(float(ops.xfro2) / (float(ops.m_proj) * float(ops.dim)) * (1.0 - float(h2)) / float(h2))


def h2_from_log_delta(ops: BoltLmmOps, log_delta: float) -> float:
    return float(ops.xfro2) / (float(ops.xfro2) + float(ops.m_proj) * float(ops.dim) * math.exp(float(log_delta)))


def _sum_score_squares(ops: BoltLmmOps, vector) -> float:
    total = 0.0
    for chrom in ops.chroms:
        if not ops.model_stats_for(chrom):
            continue
        scores = ops.scores(chrom, vector)
        xp = _array_module(scores)
        total += _as_float(xp.sum(scores * scores))
    return float(total)


# ---------------------------------------------------------------------------
# MC component generation
# ---------------------------------------------------------------------------

def _generate_bolt_mc_components(
    ops: BoltLmmOps,
    y,
    *,
    trials: int,
    seed: int,
    rng_kind: str = "numpy",
) -> Tuple[List[Any], List[Any], Any]:
    if rng_kind not in ("numpy", "boost"):
        raise ValueError(f"unknown rng_kind {rng_kind!r}; expected 'numpy' or 'boost'")

    xp = ops.xp
    inv_sqrt_m = 1.0 / math.sqrt(float(ops.m_proj))

    weights_by_chrom: Dict[Any, np.ndarray] = {
        chrom: np.zeros((int(trials), len(ops.model_stats_for(chrom))), dtype=np.float64)
        for chrom in ops.chroms
        if ops.model_stats_for(chrom)
    }

    if rng_kind == "boost":
        # Scalar Boost-compatible draws: bit-matches the BOLT-LMM reference.
        rng = BoostMt19937(int(seed) + 1)
        randn = BoostNormalDistribution()
        for chrom in ops.chroms:
            model_stats = ops.model_stats_for(chrom)
            if not model_stats:
                continue
            chrom_weights = weights_by_chrom[chrom]
            for pos, stat in enumerate(model_stats):
                for trial in range(int(trials)):
                    chrom_weights[trial, pos] = randn(rng) * inv_sqrt_m
        noise = None  # generated per-trial below
    else:
        # Fast vectorized numpy draws (default). Order: all weights, then noise.
        gen = np.random.default_rng(int(seed) + 1)
        for chrom in ops.chroms:
            model_stats = ops.model_stats_for(chrom)
            if not model_stats:
                continue
            m_c = len(model_stats)
            weights_by_chrom[chrom] = (
                gen.standard_normal((int(trials), m_c)) * inv_sqrt_m
            )
        noise = gen.standard_normal((int(trials), int(ops.n)))

    g_rand: List[Any] = []
    e_rand: List[Any] = []

    for trial in range(int(trials)):
        g = xp.zeros((ops.n,), dtype=DTYPE)
        for chrom, chrom_w in weights_by_chrom.items():
            w = xp.asarray(chrom_w[trial], dtype=DTYPE)
            g = g + ops.apply_x(chrom, w)
        ops.project_inplace(g)
        g_rand.append(g)

    if rng_kind == "boost":
        for _trial in range(int(trials)):
            values = np.fromiter(
                (randn(rng) for _ in range(int(ops.n))),
                dtype=np.float64,
                count=int(ops.n),
            )
            e = xp.asarray(values, dtype=DTYPE)
            ops.project_inplace(e)
            e_rand.append(e)
    else:
        for trial in range(int(trials)):
            e = xp.asarray(noise[trial], dtype=DTYPE)
            ops.project_inplace(e)
            e_rand.append(e)

    y_dev = ops.project(xp.asarray(np.asarray(y, dtype=DTYPE).copy()))
    return g_rand, e_rand, y_dev


# ---------------------------------------------------------------------------
# MC scaling
# ---------------------------------------------------------------------------

def _compute_mc_scaling(
    ops: BoltLmmOps,
    y_dev,
    g_rand: Sequence[Any],
    e_rand: Sequence[Any],
    *,
    log_delta: float,
    rel_tol: float,
    max_iter: int,
    stats: CgStats,
) -> McScalingResult:
    trials = len(g_rand)
    delta = math.exp(float(log_delta))
    sqrt_delta = math.sqrt(delta)

    def h_into(src, dst) -> None:
        result = ops.apply_k(src, exclude_chrom=None)
        dst[...] = result + delta * src

    rand_beta: List[float] = []
    rand_eps: List[float] = []

    rhs_columns = []
    for g_t, e_t in zip(g_rand, e_rand):
        rhs = e_t * sqrt_delta + g_t
        ops.project_inplace(rhs)
        rhs_columns.append(rhs)
    rhs_columns.append(y_dev)

    z_columns = bolt_conj_grad_solve(
        [h_into for _ in rhs_columns],
        rhs_columns,
        rel_tol=rel_tol,
        max_iter=max_iter,
        stats=stats,
        project=ops.project,
    )
    for z_t in z_columns[:-1]:
        rand_beta.append(_sum_score_squares(ops, z_t))
        rand_eps.append(_dot(z_t, z_t))

    z_data = z_columns[-1]
    data_beta = _sum_score_squares(ops, z_data)
    data_eps = _dot(z_data, z_data)

    if min([data_beta, data_eps, *rand_beta, *rand_eps]) <= 0.0:
        raise RuntimeError("invalid BOLT MC-scaling objective component")

    rand_beta_total = float(sum(rand_beta))
    rand_eps_total = float(sum(rand_eps))
    f_reml = math.log((data_beta / data_eps) / (rand_beta_total / rand_eps_total))

    f_jacks: List[float] = []
    for jack in range(trials + 1):
        jack_rand_beta = 0.0
        jack_rand_eps = 0.0
        for trial in range(trials):
            if trial != jack:
                jack_rand_beta += rand_beta[trial]
                jack_rand_eps += rand_eps[trial]
        if jack_rand_beta <= 0.0 or jack_rand_eps <= 0.0:
            f_jacks.append(float("nan"))
        else:
            f_jacks.append(math.log((data_beta / data_eps) / (jack_rand_beta / jack_rand_eps)))
    f_jacks[-1] = f_reml

    f_rands_as_data: List[float] = []
    for trial in range(trials):
        f_rands_as_data.append(
            math.log((rand_beta[trial] / rand_eps[trial]) / (rand_beta_total / rand_eps_total))
        )

    sigma2_k = _dot(y_dev, z_data) / float(max(ops.dim, 1))
    return McScalingResult(
        log_delta=float(log_delta),
        f_jacks=tuple(float(v) for v in f_jacks),
        f_rands_as_data=tuple(float(v) for v in f_rands_as_data),
        sigma2_k=float(sigma2_k),
        all_hinv_y=z_data,
    )


# ---------------------------------------------------------------------------
# Variance-component fitting
# ---------------------------------------------------------------------------

def fit_bolt_variance_components(
    ops: BoltLmmOps,
    y,
    *,
    mc_trials: int,
    seed: int,
    rel_tol: float,
    max_iter: int,
    stats: CgStats,
    rng_kind: str = "numpy",
) -> VarianceFit:
    trials = max(2, int(mc_trials))
    g_rand, e_rand, y_dev = _generate_bolt_mc_components(
        ops, y, trials=trials, seed=int(seed), rng_kind=rng_kind
    )

    def evaluate(log_delta: float) -> McScalingResult:
        return _compute_mc_scaling(
            ops, y_dev, g_rand, e_rand,
            log_delta=float(log_delta),
            rel_tol=rel_tol,
            max_iter=max_iter,
            stats=stats,
        )

    prev = evaluate(log_delta_from_h2(ops, 0.25))
    cur = evaluate(log_delta_from_h2(ops, 0.125 if prev.f_reml < 0.0 else 0.5))
    best = prev if abs(prev.f_reml) <= abs(cur.f_reml) else cur
    if abs(prev.f_reml) < abs(cur.f_reml):
        prev, cur = cur, prev

    best_accepts_secant = False
    for _step in range(5):
        if abs(cur.f_reml - prev.f_reml) < 1e-300:
            break
        next_log_delta = (prev.log_delta * cur.f_reml - cur.log_delta * prev.f_reml) / (cur.f_reml - prev.f_reml)
        next_log_delta = float(np.clip(next_log_delta, -10.0, 10.0))
        if (not best_accepts_secant) and best.log_delta == cur.log_delta and abs(next_log_delta - cur.log_delta) < 0.01:
            break
        prev = cur
        cur = evaluate(next_log_delta)
        if (not best_accepts_secant) or abs(cur.f_reml) < abs(best.f_reml):
            best = cur
            best_accepts_secant = True

    logger.debug("Secant variance search complete")

    delta = math.exp(float(best.log_delta))
    sigma_g2 = float(best.sigma2_k)
    sigma_e2 = delta * sigma_g2
    h2 = h2_from_log_delta(ops, best.log_delta)
    return VarianceFit(
        log_delta=float(best.log_delta),
        sigma_g2=sigma_g2,
        sigma_e2=float(sigma_e2),
        h2=float(h2),
        delta=float(delta),
        all_hinv_y=best.all_hinv_y,
    )


# ---------------------------------------------------------------------------
# LOCO residuals
# ---------------------------------------------------------------------------

def solve_loco_hinv_y(
    ops: BoltLmmOps,
    y,
    *,
    fit: VarianceFit,
    rel_tol: float,
    max_iter: int,
    stats: CgStats,
) -> Dict[Any, Any]:
    xp = ops.xp
    y_dev = ops.project(xp.asarray(np.asarray(y, dtype=DTYPE)))
    chroms = ops.chroms
    matvecs = []
    rhs_columns = []
    for chrom in chroms:
        def h_into(src, dst, left_out=chrom) -> None:
            result = ops.apply_k(src, exclude_chrom=left_out)
            dst[...] = result + float(fit.delta) * src

        matvecs.append(h_into)
        rhs_columns.append(y_dev)
    solved = bolt_conj_grad_solve(
        matvecs, rhs_columns,
        rel_tol=rel_tol, max_iter=max_iter, stats=stats,
        project=ops.project,
    )
    return {chrom: value for chrom, value in zip(chroms, solved)}


# ---------------------------------------------------------------------------
# Calibration SNP selection
# ---------------------------------------------------------------------------

def select_bolt_calibration_snps(
    ops: BoltLmmOps,
    *,
    fit: VarianceFit,
    count: int,
    seed: int,
) -> Tuple[List[Tuple[Any, BoltVariantStats]], int]:
    num_calib = int(count)
    if num_calib < 2:
        raise ValueError("at least two calibration SNPs are required")

    ordered = ops.all_model_stats()  # List[(chrom, BoltVariantStats)]
    model_count = len(ordered)
    if model_count <= 0:
        raise ValueError("no eligible SNPs available for calibration")
    if num_calib > model_count:
        raise ValueError(
            f"requested {num_calib} calibration SNPs but only {model_count} model SNPs are available"
        )

    m_total = model_count
    m_first = [m_total] * (num_calib + 1)
    m_good = 0
    for m in range(m_total):
        block = num_calib * m_good // model_count
        if m_first[block] == m_total:
            m_first[block] = m
        m_good += 1
    if any(start == m_total for start in m_first[:-1]):
        raise RuntimeError("failed to build BOLT calibration SNP blocks")

    all_hinv_norm2 = _dot(fit.all_hinv_y, fit.all_hinv_y)
    if all_hinv_norm2 <= 0.0:
        raise RuntimeError("all-chromosome H^-1 y has nonpositive norm")

    # Compute GRAMMAR scores per chrom
    grammar_scores_by_chrom: Dict[Any, np.ndarray] = {}
    for chrom in ops.chroms:
        if not ops.model_stats_for(chrom):
            continue
        scores = ops.scores(chrom, fit.all_hinv_y)
        xp = _array_module(scores)
        if hasattr(xp, "asnumpy"):
            grammar_scores_by_chrom[chrom] = xp.asnumpy(scores)
        else:
            grammar_scores_by_chrom[chrom] = np.asarray(scores)

    rng = BoostMt19937(int(seed) + 321)
    selected: List[Tuple[Any, BoltVariantStats]] = []
    tried = 0
    for block in range(num_calib):
        block_start = int(m_first[block])
        block_end = int(m_first[block + 1])
        block_width = block_end - block_start
        if block_width <= 0:
            raise RuntimeError(f"empty calibration block {block}")
        attempts = 0
        while True:
            attempts += 1
            if attempts > 1_000_000:
                raise RuntimeError(f"could not select a calibration SNP from block {block}")
            m = block_start + boost_uniform_int_0_2pow30(rng) % block_width
            chrom, stat = ordered[m]
            tried += 1
            # position within this chrom's compact score array
            pos = ops._local_idx_to_pos[chrom][stat.local_idx]
            grammar_score = float(grammar_scores_by_chrom[chrom][pos])
            x_norm2 = float(stat.x_norm2)
            retro_stat = (grammar_score ** 2) / all_hinv_norm2 / x_norm2 * float(ops.dim)
            if retro_stat < 5.0:
                selected.append((chrom, stat))
                break
    return selected, tried


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate_lmm_inf(
    ops: BoltLmmOps,
    y,
    residuals: Dict[Any, Any],
    *,
    fit: VarianceFit,
    count: int,
    seed: int,
    rel_tol: float,
    max_iter: int,
    stats: CgStats,
) -> CalibrationResult:
    selected, tried = select_bolt_calibration_snps(ops, fit=fit, count=int(count), seed=int(seed))
    pro_stats: List[float] = []
    retro_stats: List[float] = []
    ratios: List[float] = []
    n_minus_c = float(max(ops.dim, 1))

    xp = ops.xp
    y_dev = ops.project(xp.asarray(np.asarray(y, dtype=DTYPE).copy()))
    chroms = ops.chroms
    rhs_columns = []
    matvecs = []

    for chrom in chroms:
        def h_into(src, dst, left_out=chrom) -> None:
            result = ops.apply_k(src, exclude_chrom=left_out)
            dst[...] = result + float(fit.delta) * src

        matvecs.append(h_into)
        rhs_columns.append(y_dev)

    selected_columns = []
    for sel_chrom, sel_stat in selected:
        x = ops.column(sel_chrom, sel_stat.local_idx)

        def h_into(src, dst, left_out=sel_chrom) -> None:
            result = ops.apply_k(src, exclude_chrom=left_out)
            dst[...] = result + float(fit.delta) * src

        matvecs.append(h_into)
        rhs_columns.append(x)
        selected_columns.append(x)

    solved_columns = bolt_conj_grad_solve(
        matvecs, rhs_columns,
        rel_tol=rel_tol, max_iter=max_iter, stats=stats,
        project=ops.project,
    )

    residuals.clear()
    for chrom, solved in zip(chroms, solved_columns[: len(chroms)]):
        residuals[chrom] = solved

    q_by_sel = {
        i: solved
        for i, solved in enumerate(solved_columns[len(chroms):])
    }
    x_by_sel = {i: x for i, x in enumerate(selected_columns)}

    h_norm2 = {chrom: _dot(value, value) for chrom, value in residuals.items()}
    phi_h_phi = {chrom: _dot(y_dev, value) for chrom, value in residuals.items()}

    for i, (sel_chrom, sel_stat) in enumerate(selected):
        x = x_by_sel[i]
        score_h = _dot(x, residuals[sel_chrom])
        x_norm2 = _dot(x, x)
        if h_norm2[sel_chrom] <= 0.0 or phi_h_phi[sel_chrom] <= 0.0:
            raise RuntimeError(f"invalid LOCO H^-1 y moments for chr{sel_chrom}")
        if x_norm2 <= 0.0:
            raise RuntimeError(f"selected calibration SNP has nonpositive projected norm: {sel_stat.local_idx}")
        retro = n_minus_c * score_h * score_h / (h_norm2[sel_chrom] * x_norm2)
        if retro <= 0.0:
            raise RuntimeError(f"selected calibration SNP has nonpositive retrospective stat: {sel_stat.local_idx}")
        q = q_by_sel[i]
        denom_h = _dot(x, q)
        if denom_h <= 0.0:
            raise RuntimeError(f"selected calibration SNP has nonpositive prospective denominator: {sel_stat.local_idx}")
        pro = n_minus_c * score_h * score_h / denom_h / phi_h_phi[sel_chrom]
        pro_stats.append(float(pro))
        retro_stats.append(float(retro))
        ratios.append(float(pro / retro))

    total_pro = float(sum(pro_stats))
    total_retro = float(sum(retro_stats))
    if total_pro <= 0.0 or total_retro <= 0.0:
        raise RuntimeError("calibration failed: prospective or retrospective sum is nonpositive")
    factor = total_pro / total_retro
    calibration_jacks = [
        (total_pro - pro) / (total_retro - retro)
        for pro, retro in zip(pro_stats, retro_stats)
    ]
    jack_count = len(calibration_jacks)
    jack_sum = float(sum(calibration_jacks))
    jack_sum2 = float(sum(v * v for v in calibration_jacks))
    calibration_std = math.sqrt(
        max(0.0, (jack_sum2 - jack_sum * jack_sum / jack_count) * (jack_count - 1) / jack_count)
    )
    ratio_of_medians = float(
        np.median(np.asarray(pro_stats, dtype=np.float64))
        / np.median(np.asarray(retro_stats, dtype=np.float64))
    )
    median_of_ratios = float(np.median(np.asarray(ratios, dtype=np.float64)))
    if calibration_std > 0.01:
        factor = ratio_of_medians
    if factor <= 0.0:
        raise RuntimeError(f"calibration factor is nonpositive: {factor}")

    vinv_scale_by_chrom = {}
    for chrom, residual in residuals.items():
        resid_norm2 = _dot(residual, residual)
        if resid_norm2 <= 0.0:
            raise RuntimeError(f"LOCO H^-1 y has nonpositive norm for chr{chrom}")
        resid_factor = math.sqrt(n_minus_c / resid_norm2 * factor)
        vinv_scale_by_chrom[chrom] = 1.0 / (resid_factor * float(fit.sigma_g2))

    return CalibrationResult(
        factor=float(factor),
        std=float(calibration_std),
        ratio_of_medians=ratio_of_medians,
        median_of_ratios=median_of_ratios,
        selected_snps=tuple(str(stat.local_idx) for _, stat in selected),
        tried_snps=int(tried),
        vinv_scale_by_chrom=vinv_scale_by_chrom,
    )


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def bolt_lmm_inf(
    chrom_grgs: List[Tuple[Any, GRGCalcInterface]],
    y: np.ndarray,
    covariates: CovariateBasis,
    *,
    num_calib_snps: int = DEFAULT_NUM_CALIB_SNPS,
    mc_trials: int = DEFAULT_H2_EST_MC_TRIALS,
    cg_tol: float = DEFAULT_CG_TOL,
    max_iter: int = DEFAULT_MAX_ITERS,
    seed: int = BOLT_RANDOM_SEED,
    threads: int = 1,
    rng_kind: str = "numpy",
) -> Tuple[VarianceFit, CalibrationResult, Dict, pd.DataFrame]:
    """
    Run BOLT-LMM-inf on one or more chromosomes.

    :param chrom_grgs: List of (chromosome_label, GRGCalcInterface), one per chromosome.
    :param y: Phenotype vector of length n_individuals.
    :param covariates: Orthonormal covariate basis (includes intercept).
    :param rng_kind: RNG for MC variance-component probes: "numpy" (fast, default)
        or "boost" (slower, bit-matches the BOLT-LMM reference).
    :returns: (VarianceFit, CalibrationResult, residuals_dict, results_dataframe)
    """
    cg_stats = CgStats()

    # Compute per-variant stats for each chromosome
    chrom_all_stats: List[List[BoltVariantStats]] = []
    grgs = [grg for _, grg in chrom_grgs]
    # TODO: check if CPU computations are non-trivial for multi-gpu
    scheduler = _wrap_grg(chrom_grgs[0][1]).make_scheduler(grgs, threads)
    futures = [scheduler.submit(grg, compute_bolt_variant_stats, grg, covariates, grg.num_individuals) for _, grg in chrom_grgs]
    for future in futures:
        stats = future.result()
        chrom_all_stats.append(stats)
    logger.debug("Variant statistics complete")

    # Build ops
    ops = BoltLmmOps(chrom_grgs, chrom_all_stats, covariates, threads=threads).setup()
    logger.debug("BoltLmmOps setup complete")

    # Fit variance components (variance fitting CG uses 10x looser tolerance,
    # matching grg-spmv convention — secant search is insensitive to CG precision)
    fit = fit_bolt_variance_components(
        ops, y,
        mc_trials=mc_trials,
        seed=seed,
        rng_kind=rng_kind,
        rel_tol=10.0 * cg_tol,
        max_iter=max_iter,
        stats=cg_stats,
    )
    logger.debug("Variance component fitting complete")

    # LOCO + calibration
    residuals: Dict[Any, Any] = {}
    calibration = calibrate_lmm_inf(
        ops, y, residuals,
        fit=fit,
        count=num_calib_snps,
        seed=seed,
        rel_tol=cg_tol,
        max_iter=max_iter,
        stats=cg_stats,
    )
    logger.debug("LOCO calibration complete")

    # Compute association statistics (fast numeric stats, then annotate)
    stats = compute_lmm_inf_stats(
        ops, chrom_grgs, chrom_all_stats, y, residuals, fit, calibration
    )
    logger.debug("Association statistics computation complete")

    results_df = lmm_inf_stats_to_dataframe(stats, chrom_grgs)
    logger.debug("Association results conversion complete")

    return fit, calibration, residuals, results_df
