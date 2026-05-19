"""BOLT-LMM-inf algorithmic primitives: dataclasses, RNGs, CG solver, and BoltLmmOps."""

from __future__ import annotations

import contextlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import chi2 as _scipy_chi2

import pygrgl

from grapp.grg_calculator import GRGCalcInterface, GRGSpMVCalculator, _wrap_grg
from grapp.linalg.ops_scipy import (
    SciPyStdXOperator,
    MultiSciPyLOCOStdXXTOperator,
)
from grapp.util.simple import (
    allele_counts, allele_counts_cupy,
    allele_frequencies, allele_frequencies_cupy,
)


DTYPE = np.dtype(np.float64)
BOLT_RANDOM_SEED = 12345
BOLT_BAD_SNP_STAT = -1e9
DEFAULT_NUM_CALIB_SNPS = 30
DEFAULT_H2_EST_MC_TRIALS = 3
DEFAULT_CG_TOL = 5e-4
DEFAULT_MAX_ITERS = 10_000
BOOST_NORMAL_HEADER = Path("/usr/include/boost/random/normal_distribution.hpp")
BOOST_EXPONENTIAL_HEADER = Path("/usr/include/boost/random/exponential_distribution.hpp")

_UP = pygrgl.TraversalDirection.UP


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _as_float(value: Any) -> float:
    if hasattr(value, "get"):
        return float(value.get())
    return float(value)


def _array_module(value):
    if type(value).__module__.split(".", 1)[0] == "cupy":
        import cupy as cp
        return cp
    return np


def _dot(left, right) -> float:
    xp = _array_module(left)
    return _as_float(xp.sum(left * right))

def _to_np(arr) -> np.ndarray:
    """Convert CuPy or NumPy array to NumPy."""
    return arr.get() if hasattr(arr, "get") else np.asarray(arr)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CgStats:
    solves: int = 0
    iterations: int = 0
    max_iterations: int = 0
    max_rel_resid: float = 0.0

    def add(self, iterations: int, rel_resid: float) -> None:
        iters = int(iterations)
        self.solves += 1
        self.iterations += iters
        self.max_iterations = max(self.max_iterations, iters)
        self.max_rel_resid = max(self.max_rel_resid, float(rel_resid))


@dataclass(frozen=True)
class VarianceFit:
    log_delta: float
    sigma_g2: float
    sigma_e2: float
    h2: float
    delta: float
    all_hinv_y: Any


@dataclass(frozen=True)
class McScalingResult:
    log_delta: float
    f_jacks: tuple
    f_rands_as_data: tuple
    sigma2_k: float
    all_hinv_y: Any

    @property
    def f_reml(self) -> float:
        return float(self.f_jacks[-1])


@dataclass(frozen=True)
class CalibrationResult:
    factor: float
    std: float
    ratio_of_medians: float
    median_of_ratios: float
    selected_snps: tuple
    tried_snps: int
    vinv_scale_by_chrom: dict


# ---------------------------------------------------------------------------
# CovariateBasis
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CovariateBasis:
    """BOLT-style orthonormal covariate basis, including the all-ones vector."""

    basis: np.ndarray
    covar_cols: tuple
    q_covar_cols: tuple
    covar_max_levels: int

    def __post_init__(self) -> None:
        arr = np.asarray(self.basis, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError("covariate basis must be two-dimensional")
        if arr.shape[0] < 1:
            raise ValueError("covariate basis must include at least one sample")
        object.__setattr__(self, "basis", np.ascontiguousarray(arr))

    @property
    def nused(self) -> int:
        return int(self.basis.shape[0])

    @property
    def cindep(self) -> int:
        return int(self.basis.shape[1])

    @property
    def dim(self) -> int:
        return int(self.nused - self.cindep)

    @classmethod
    def intercept_only(cls, n: int) -> "CovariateBasis":
        n_int = int(n)
        if n_int < 1:
            raise ValueError("sample count must be positive")
        basis = np.full((n_int, 1), 1.0 / math.sqrt(float(n_int)), dtype=np.float64)
        return cls(basis=basis, covar_cols=(), q_covar_cols=(), covar_max_levels=10)

    @classmethod
    def from_matrix(
        cls,
        matrix: np.ndarray,
        *,
        covar_cols: Sequence,
        q_covar_cols: Sequence,
        covar_max_levels: int,
    ) -> "CovariateBasis":
        covars = np.asarray(matrix, dtype=np.float64)
        if covars.ndim != 2:
            raise ValueError("covariate matrix must be two-dimensional")
        if covars.shape[0] < 1 or covars.shape[1] < 1:
            raise ValueError("covariate matrix must be non-empty")
        if covars.shape[1] > covars.shape[0]:
            raise ValueError("number of covariate columns cannot exceed sample count")
        u, s, _vt = np.linalg.svd(np.asfortranarray(covars), full_matrices=False)
        if s.size == 0 or s[0] <= 0.0:
            raise ValueError("covariate matrix is rank-deficient with no independent columns")
        rank = int(np.count_nonzero(s >= (s[0] * 1e-8)))
        return cls(
            basis=u[:, :rank],
            covar_cols=tuple(str(v) for v in covar_cols),
            q_covar_cols=tuple(str(v) for v in q_covar_cols),
            covar_max_levels=int(covar_max_levels),
        )

    def project_host(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        was_vector = arr.ndim == 1
        mat = arr.reshape((self.nused, 1)) if was_vector else arr
        if mat.shape[0] != self.nused:
            raise ValueError(f"vector has {mat.shape[0]} rows; expected {self.nused}")
        projected = mat - self.basis @ (self.basis.T @ mat)
        return projected[:, 0] if was_vector else projected

    def project_host_inplace(self, values: np.ndarray) -> np.ndarray:
        values[...] = self.project_host(values)
        return values

    def project_device(self, values):
        xp = _array_module(values)
        if xp is np:
            return self.project_host(values)
        q = xp.asarray(self.basis, dtype=DTYPE)
        arr = xp.asarray(values, dtype=DTYPE)
        was_vector = arr.ndim == 1
        mat = arr.reshape((self.nused, 1)) if was_vector else arr
        projected = mat - q @ (q.T @ mat)
        return projected[:, 0] if was_vector else projected

    def project_device_inplace(self, values):
        values[...] = self.project_device(values)
        return values


# ---------------------------------------------------------------------------
# Boost-compatible RNGs (needed for deterministic reproduction of BOLT results)
# ---------------------------------------------------------------------------

class BoostMt19937:
    """Small Boost.Random mt19937 clone for BOLT's calibration SNP sampler."""

    _n = 624
    _m = 397
    _r = 31
    _a = 0x9908B0DF
    _u = 11
    _d = 0xFFFFFFFF
    _s = 7
    _b = 0x9D2C5680
    _t = 15
    _c = 0xEFC60000
    _l = 18
    _f = 1812433253
    _mask = 0xFFFFFFFF
    _upper_mask = 0x80000000
    _lower_mask = 0x7FFFFFFF

    def __init__(self, seed: int):
        self.x = [0] * self._n
        self.i = self._n
        self.seed(seed)

    def seed(self, value: int) -> None:
        self.x[0] = int(value) & self._mask
        for idx in range(1, self._n):
            prev = self.x[idx - 1]
            self.x[idx] = (self._f * (prev ^ (prev >> 30)) + idx) & self._mask
        self.i = self._n
        self._normalize_state()

    def _normalize_state(self) -> None:
        y0 = self.x[self._m - 1] ^ self.x[self._n - 1]
        if y0 & (1 << 31):
            y0 = ((y0 ^ self._a) << 1) | 1
        else:
            y0 <<= 1
        self.x[0] = (self.x[0] & self._upper_mask) | (y0 & self._lower_mask)
        if not any(self.x):
            self.x[0] = 1 << 31

    def _twist(self) -> None:
        for idx in range(0, self._n - self._m):
            y = (self.x[idx] & self._upper_mask) | (self.x[idx + 1] & self._lower_mask)
            self.x[idx] = (self.x[idx + self._m] ^ (y >> 1) ^ ((self.x[idx + 1] & 1) * self._a)) & self._mask
        for idx in range(self._n - self._m, self._n - 1):
            y = (self.x[idx] & self._upper_mask) | (self.x[idx + 1] & self._lower_mask)
            self.x[idx] = (self.x[idx - (self._n - self._m)] ^ (y >> 1) ^ ((self.x[idx + 1] & 1) * self._a)) & self._mask
        y = (self.x[self._n - 1] & self._upper_mask) | (self.x[0] & self._lower_mask)
        self.x[self._n - 1] = (self.x[self._m - 1] ^ (y >> 1) ^ ((self.x[0] & 1) * self._a)) & self._mask
        self.i = 0

    def __call__(self) -> int:
        if self.i == self._n:
            self._twist()
        z = self.x[self.i]
        self.i += 1
        z ^= (z >> self._u) & self._d
        z ^= (z << self._s) & self._b
        z ^= (z << self._t) & self._c
        z ^= z >> self._l
        return z & self._mask


def boost_uniform_int_0_2pow30(rng: BoostMt19937) -> int:
    """Match boost::uniform_int<>(0, 1<<30) for a 32-bit mt19937 engine."""
    max_value = 1 << 30
    bucket_size = 3
    while True:
        result = rng() // bucket_size
        if result <= max_value:
            return int(result)


_BOOST_TABLES: Dict[Tuple[Path, str], Tuple[float, ...]] = {}


def _boost_table(header: Path, table_name: str) -> Tuple[float, ...]:
    key = (Path(header), str(table_name))
    cached = _BOOST_TABLES.get(key)
    if cached is not None:
        return cached
    text = Path(header).read_text(encoding="utf-8")
    match = re.search(rf"{re.escape(table_name)}\[\d+\]\s*=\s*\{{(?P<body>.*?)\}};", text, flags=re.S)
    if match is None:
        raise RuntimeError(f"could not find Boost.Random {table_name} in {header}")
    values = tuple(float(item.strip()) for item in match.group("body").replace("\n", " ").split(",") if item.strip())
    _BOOST_TABLES[key] = values
    return values


def boost_uniform_01(rng: BoostMt19937) -> float:
    """Match boost::random::uniform_01<double> for boost::mt19937."""
    return float(rng()) / 4294967296.0


def boost_generate_int_float_pair_8(rng: BoostMt19937) -> Tuple[float, int]:
    """Match boost::random::detail::generate_int_float_pair<double, 8>."""
    first = int(rng())
    bucket = first & 0xFF
    r = float(first >> 8) / 16777216.0
    second = int(rng())
    r += float(second & ((1 << 29) - 1))
    r /= float(1 << 29)
    return r, bucket


class BoostExponentialDistribution:
    """Boost.Random exponential_distribution<> clone for the normal tail path."""

    def __init__(self, lambda_arg: float = 1.0):
        self.lambda_arg = float(lambda_arg)
        if self.lambda_arg <= 0.0:
            raise ValueError("lambda_arg must be positive")
        self._table_x = _boost_table(BOOST_EXPONENTIAL_HEADER, "table_x")
        self._table_y = _boost_table(BOOST_EXPONENTIAL_HEADER, "table_y")

    def __call__(self, rng: BoostMt19937) -> float:
        table_x = self._table_x
        table_y = self._table_y
        shift = 0.0
        while True:
            r, i = boost_generate_int_float_pair_8(rng)
            x = r * table_x[i]
            if x < table_x[i + 1]:
                return (shift + x) / self.lambda_arg
            if i == 0:
                shift += table_x[1]
                continue
            y01 = boost_uniform_01(rng)
            y = table_y[i] + y01 * (table_y[i + 1] - table_y[i])
            y_above_ubound = (table_x[i] - table_x[i + 1]) * y01 - (table_x[i] - x)
            y_above_lbound = y - (table_y[i + 1] + (table_x[i + 1] - x) * table_y[i + 1])
            if y_above_ubound < 0.0 and (y_above_lbound < 0.0 or y < math.exp(-x)):
                return (x + shift) / self.lambda_arg


class BoostNormalDistribution:
    """Boost.Random normal_distribution<> clone used by BOLT's MC scaling."""

    def __init__(self, mean: float = 0.0, sigma: float = 1.0):
        self.mean = float(mean)
        self.sigma = float(sigma)
        if self.sigma < 0.0:
            raise ValueError("sigma must be nonnegative")
        self._table_x = _boost_table(BOOST_NORMAL_HEADER, "table_x")
        self._table_y = _boost_table(BOOST_NORMAL_HEADER, "table_y")

    def __call__(self, rng: BoostMt19937) -> float:
        unit = self._unit(rng)
        return unit * self.sigma + self.mean

    def _unit(self, rng: BoostMt19937) -> float:
        table_x = self._table_x
        table_y = self._table_y
        while True:
            r, bucket = boost_generate_int_float_pair_8(rng)
            sign = (bucket & 1) * 2 - 1
            i = bucket >> 1
            x = r * table_x[i]
            if x < table_x[i + 1]:
                return x * sign
            if i == 0:
                return self._tail(rng) * sign
            y01 = boost_uniform_01(rng)
            y = table_y[i] + y01 * (table_y[i + 1] - table_y[i])
            if table_x[i] >= 1.0:
                y_above_ubound = (table_x[i] - table_x[i + 1]) * y01 - (table_x[i] - x)
                y_above_lbound = y - (table_y[i] + (table_x[i] - x) * table_y[i] * table_x[i])
            else:
                y_above_lbound = (table_x[i] - table_x[i + 1]) * y01 - (table_x[i] - x)
                y_above_ubound = y - (table_y[i] + (table_x[i] - x) * table_y[i] * table_x[i])
            if y_above_ubound < 0.0 and (y_above_lbound < 0.0 or y < math.exp(-(x * x / 2.0))):
                return x * sign

    def _tail(self, rng: BoostMt19937) -> float:
        tail_start = self._table_x[1]
        exp_x = BoostExponentialDistribution(tail_start)
        exp_y = BoostExponentialDistribution()
        while True:
            x = exp_x(rng)
            y = exp_y(rng)
            if 2.0 * y > x * x:
                return x + tail_start


# ---------------------------------------------------------------------------
# CG solver
# ---------------------------------------------------------------------------

def bolt_conj_grad_solve(
    matvecs: Sequence[Any],
    rhs_columns: Sequence[Any],
    *,
    rel_tol: float,
    max_iter: int,
    stats: Optional[CgStats] = None,
    project,
) -> List[Any]:
    """Mirrors BOLT-LMM_v2.5 Bolt::conjGradSolve."""
    if len(matvecs) != len(rhs_columns):
        raise ValueError("matvec and RHS counts differ")
    if not rhs_columns:
        return []
    xp = _array_module(rhs_columns[0])
    b_cols = [project(xp.asarray(rhs, dtype=DTYPE).copy()) for rhs in rhs_columns]
    b = xp.column_stack(b_cols)
    x = xp.zeros_like(b)
    r = b.copy()
    p = r.copy()
    hp = xp.empty_like(b)
    r2_orig = xp.sum(r * r, axis=0)
    r2_old = r2_orig.copy()
    rels = xp.sqrt(r2_old / r2_orig)
    for it in range(1, int(max_iter) + 1):
        for col, matvec in enumerate(matvecs):
            matvec(p[:, col], hp[:, col])
        denom = xp.sum(p * hp, axis=0)
        alpha = r2_old / denom
        x += p * alpha.reshape((1, -1))
        r -= hp * alpha.reshape((1, -1))
        r = project(r)
        r2_new = xp.sum(r * r, axis=0)
        rels = xp.sqrt(r2_new / r2_orig)
        if not bool(_as_float(xp.any(rels > float(rel_tol)))):
            if stats is not None:
                rel_values = xp.asnumpy(rels) if hasattr(xp, "asnumpy") else np.asarray(rels)
                for rel in rel_values:
                    stats.add(it, float(rel))
            return [project(x[:, idx].copy()) for idx in range(x.shape[1])]
        beta = r2_new / r2_old
        p *= beta.reshape((1, -1))
        p += r
        r2_old = r2_new
    if stats is not None:
        rel_values = xp.asnumpy(rels) if hasattr(xp, "asnumpy") else np.asarray(rels)
        for rel in rel_values:
            stats.add(int(max_iter), float(rel))
    return [project(x[:, idx].copy()) for idx in range(x.shape[1])]


# ---------------------------------------------------------------------------
# BoltVariantStats
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BoltVariantStats:
    """Per-variant statistics needed for BOLT-LMM-inf."""
    local_idx: int
    mean: float
    mean_center_norm2: float
    proj_norm2: float
    norm_scale: float
    x_norm2: float

    @property
    def is_model_variant(self) -> bool:
        return (
            float(self.mean_center_norm2) > 0.0
            and float(self.proj_norm2) >= 0.1
            and float(self.norm_scale) > 0.0
            and float(self.x_norm2) > 0.0
        )


# ---------------------------------------------------------------------------
# BoltLmmOps
# ---------------------------------------------------------------------------

class BoltLmmOps:
    """
    GRG-backed BOLT-LMM-inf linear algebra using grapp's standardized operators.

    Accepts one GRG per chromosome for LOCO. Both K_all and LOCO variants are
    served by a single Multi-XXT operator (``MultiSciPyLOCOStdXXTOperator`` for
    SciPy, ``MultiCuPyStdXXTOperator`` for CuPy) that natively skips a
    chromosome via ``matvec_loco(exclude_op_idx=...)``.
    """

    def __init__(
        self,
        chrom_grgs: List[Tuple[Any, GRGCalcInterface]],
        chrom_stats: List[List[BoltVariantStats]],
        covariates: CovariateBasis,
        threads: int = 1,
    ):
        if len(chrom_grgs) != len(chrom_stats):
            raise ValueError("chrom_grgs and chrom_stats must have the same length")
        self._chrom_grgs = chrom_grgs
        self._chrom_stats = chrom_stats
        self._covariates = covariates
        self._threads = threads

        self._n: int = 0
        self._m_proj: int = 0
        self._m_proj_by_chrom: Dict[Any, int] = {}
        self._xfro2: float = 0.0

        self._x_ops: Dict[Any, Any] = {}
        self._k_all_op: Any = None
        self._chrom_to_op_idx: Dict[Any, int] = {}
        self._model_stats_by_chrom: Dict[Any, List[BoltVariantStats]] = {}
        self._local_idx_to_pos: Dict[Any, Dict[int, int]] = {}
        self._is_cupy: bool = False
        self._xp = np

    def setup(self) -> "BoltLmmOps":
        grgs_list = [grg for _, grg in self._chrom_grgs]
        self._n = int(grgs_list[0].num_individuals)

        if self._covariates.nused != self._n:
            raise ValueError(
                f"covariate sample count {self._covariates.nused} != GRG individual count {self._n}"
            )

        # Detect GPU backend
        self._is_cupy = bool(
            grgs_list
            and isinstance(grgs_list[0], GRGSpMVCalculator)
            and getattr(grgs_list[0], "use_cupy", False)
        )
        if self._is_cupy:
            import cupy
            self._xp = cupy

        active_grgs: List[Any] = []
        active_freqs: List[np.ndarray] = []
        active_vars: List[np.ndarray] = []

        for (chrom, grg), stats in zip(self._chrom_grgs, self._chrom_stats):
            model_stats = stats
            self._model_stats_by_chrom[chrom] = model_stats
            self._local_idx_to_pos[chrom] = {
                s.local_idx: pos for pos, s in enumerate(model_stats)
            }

            if not model_stats:
                continue

            mcn2 = np.array([s.mean_center_norm2 for s in model_stats], dtype=np.float64)
            var_c = mcn2 / float(self._n - 1)

            freqs_c = allele_frequencies_cupy(grg) if self._is_cupy else allele_frequencies(grg)

            m_c = len(model_stats)
            self._m_proj += m_c
            self._m_proj_by_chrom[chrom] = m_c
            self._xfro2 += float(sum(s.x_norm2 for s in model_stats))

            if self._is_cupy:
                from grapp.linalg.ops_cupy import CuPyStdXOperator
                self._x_ops[chrom] = CuPyStdXOperator(
                    grg, _UP, freqs_c,
                    custom_variance=var_c,
                )
            else:
                self._x_ops[chrom] = SciPyStdXOperator(
                    grg, _UP, freqs_c,
                    custom_variance=var_c,
                )

            self._chrom_to_op_idx[chrom] = len(active_grgs)
            active_grgs.append(grg)
            active_freqs.append(freqs_c)
            active_vars.append(var_c)

        if self._m_proj <= 0:
            raise ValueError("no eligible model variants")

        if self._is_cupy:
            from grapp.linalg.ops_cupy import MultiCuPyStdXXTOperator
            self._k_all_op = MultiCuPyStdXXTOperator(
                active_grgs, active_freqs,
                custom_variance=active_vars,
                threads=self._threads,
            )
        else:
            self._k_all_op = MultiSciPyLOCOStdXXTOperator(
                active_grgs, active_freqs,
                custom_variance=active_vars,
                threads=self._threads,
            )

        return self

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n(self) -> int:
        return self._n

    @property
    def dim(self) -> int:
        return self._covariates.dim

    @property
    def m_proj(self) -> int:
        return self._m_proj

    @property
    def xfro2(self) -> float:
        return self._xfro2

    @property
    def chroms(self) -> List[Any]:
        return [chrom for chrom, _ in self._chrom_grgs]

    @property
    def xp(self):
        return self._xp

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------

    def project(self, v):
        return self._covariates.project_device(v)

    def project_inplace(self, v):
        return self._covariates.project_device_inplace(v)

    # ------------------------------------------------------------------
    # Per-chromosome operations
    # ------------------------------------------------------------------

    def _device_ctx(self, chrom):
        if not self._is_cupy:
            return contextlib.nullcontext()
        import cupy as cp
        dev = getattr(self._x_ops[chrom], "_device", None)
        return cp.cuda.Device(dev) if dev is not None else contextlib.nullcontext()

    def scores(self, chrom, v) -> np.ndarray:
        """X^T @ project(v) for model variants of this chromosome (compact array)."""
        v_proj = self.project(v)
        return self._x_ops[chrom].rmatvec(v_proj)

    def apply_x(self, chrom, w) -> np.ndarray:
        """project(X @ w) for model variants of this chromosome."""
        with self._device_ctx(chrom):
            w_dev = self._xp.asarray(w, dtype=DTYPE)
        result = self._x_ops[chrom].matvec(w_dev)
        return self._covariates.project_device(result)

    def column(self, chrom, local_idx: int) -> np.ndarray:
        """The projected column x_i of the standardized X for one model variant."""
        pos = self._local_idx_to_pos[chrom][int(local_idx)]
        m = len(self._model_stats_by_chrom[chrom])
        with self._device_ctx(chrom):
            w = self._xp.zeros(m, dtype=DTYPE)
            w[pos] = 1.0
        return self.apply_x(chrom, w)

    # ------------------------------------------------------------------
    # Kinship
    # ------------------------------------------------------------------

    def apply_k(self, v, *, exclude_chrom=None) -> np.ndarray:
        """(1/m_loco) * X_loco @ X_loco^T @ project(v), then project."""
        if exclude_chrom is not None and exclude_chrom not in self._chrom_to_op_idx:
            raise KeyError(
                f"exclude_chrom={exclude_chrom!r} not in active chroms "
                f"{sorted(self._chrom_to_op_idx.keys())}"
            )
        v_dev = self._xp.asarray(v, dtype=DTYPE)
        v_proj = self._covariates.project_device(v_dev)
        exclude_idx = (
            self._chrom_to_op_idx[exclude_chrom]
            if exclude_chrom is not None else None
        )
        xxt_v = self._k_all_op.matvec_loco(v_proj, exclude_op_idx=exclude_idx)

        m = self._m_proj
        if exclude_chrom is not None:
            m -= self._m_proj_by_chrom[exclude_chrom]
        if m <= 0:
            return self._xp.zeros(self._n, dtype=DTYPE)
        result = xxt_v / float(m)
        self.project_inplace(result)
        return result

    # ------------------------------------------------------------------
    # Accessor for stats
    # ------------------------------------------------------------------

    def model_stats_for(self, chrom) -> List[BoltVariantStats]:
        return self._model_stats_by_chrom.get(chrom, [])

    def all_model_stats(self) -> List[Tuple[Any, BoltVariantStats]]:
        result = []
        for chrom, _ in self._chrom_grgs:
            for s in self._model_stats_by_chrom.get(chrom, []):
                result.append((chrom, s))
        return result


# ---------------------------------------------------------------------------
# Per-variant statistics from GRG (replaces PLINK-based attach_bed_stats)
# ---------------------------------------------------------------------------

def compute_bolt_variant_stats(
    grg: GRGCalcInterface,
    covariates: CovariateBasis,
    n_individuals: int,
) -> List[BoltVariantStats]:
    """
    Compute BOLT-LMM-inf per-variant statistics from a GRG.

    Replaces the PLINK-based attach_bed_stats + attach_projected_bed_stats pipeline.
    Uses GRG traversals to compute mean_center_norm2 and proj_norm2 without
    reading BED files.

    Note that this is lightweight and is done sequentially currently, not using
    the linear operators.
    """
    grg = _wrap_grg(grg)
    n = int(n_individuals)

    use_cupy = getattr(grg, 'use_cupy', False)
    if use_cupy:
        import cupy as cp
        xp = cp
        dev_ctx = lambda: cp.cuda.Device(grg.device)
    else:
        xp = np
        dev_ctx = contextlib.nullcontext

    # Allele counts and missingness
    if use_cupy:
        acount_raw, miss_raw = allele_counts_cupy(grg, return_missing=True)
        acount = _to_np(acount_raw).astype(np.float64)
        miss = _to_np(miss_raw).astype(np.float64)
    else:
        acount_raw, miss_raw = allele_counts(grg, return_missing=True)
        acount = np.asarray(acount_raw, dtype=np.float64)
        miss = np.asarray(miss_raw, dtype=np.float64)

    # Effective sample count and diploid mean per variant
    n_eff = n - miss / grg.ploidy
    with np.errstate(divide="ignore", invalid="ignore"):
        diploid_mean = np.where(n_eff > 0, acount / n_eff, 0.0)

    # sumsq_g = diag(X_indiv^T X_indiv)_j = sum_i g_ij^2, g_ij in {0,1,2}.
    # init="xtx" with by_individual=True computes the individual-level squared sum,
    # matching the native BED-based computation (sumsq_lut[g].sum() = sum_i g_ij^2).
    with dev_ctx():
        inp = xp.ones((1, grg.num_individuals), dtype=np.float64)

    sumsq_g = _to_np(grg.matmul(
        inp,
        pygrgl.TraversalDirection.UP,
        by_individual=True,
        init="xtx",
    )).squeeze().astype(np.float64)

    #TODO: this is not good with missing data!

    # mean_center_norm2 = sum_i (x_ij - mean_j)^2 (using n_eff for mean)
    mean_center_norm2 = sumsq_g - acount * diploid_mean

    # norm_scale uses n_individuals (not n_eff), matching BOLT's Bessel correction
    norm_scale = np.where(
        mean_center_norm2 > 0.0,
        np.sqrt((n - 1.0) / np.maximum(mean_center_norm2, 1e-300)),
        0.0,
    )

    # proj_norm2: subtract covariate contribution via c UP traversals
    # proj_norm2_i = mean_center_norm2_i - sum_k (X_i^T q_k - mean_i * sum(q_k))^2
    sum_sq_proj = np.zeros(grg.num_mutations, dtype=np.float64)
    Q = covariates.basis  # (n_individuals, cindep)
    for k in range(covariates.cindep):
        q_k = Q[:, k].astype(np.float64)
        sum_qk = float(np.sum(q_k))
        with dev_ctx():
            q_dev = xp.asarray(q_k).reshape(1, -1)
        raw_scores = _to_np(grg.matmul(
            q_dev,
            pygrgl.TraversalDirection.UP,
            by_individual=True,
        )).squeeze().astype(np.float64)
        score_k = raw_scores - diploid_mean * sum_qk
        sum_sq_proj += score_k * score_k

    proj_norm2 = np.maximum(0.0, mean_center_norm2 - sum_sq_proj)
    x_norm2 = proj_norm2 * norm_scale * norm_scale

    result = []
    for local_idx in range(grg.num_mutations):
        result.append(BoltVariantStats(
            local_idx=local_idx,
            mean=float(diploid_mean[local_idx]),
            mean_center_norm2=float(mean_center_norm2[local_idx]),
            proj_norm2=float(proj_norm2[local_idx]),
            norm_scale=float(norm_scale[local_idx]),
            x_norm2=float(x_norm2[local_idx]),
        ))
    return result


# ---------------------------------------------------------------------------
# Test-statistic output
# ---------------------------------------------------------------------------

def compute_lmm_inf_results(
    ops: BoltLmmOps,
    chrom_grgs: List[Tuple[Any, GRGCalcInterface]],
    chrom_all_stats: List[List[BoltVariantStats]],
    y,
    residuals: Dict[Any, Any],
    fit: VarianceFit,
    calibration: CalibrationResult,
) -> pd.DataFrame:
    """
    Compute per-variant BOLT-LMM-inf and linear-regression statistics.

    Returns a DataFrame with the standard BOLT-LMM output columns. Reuses
    ``ops.scores(chrom, v)`` (compact model-variant scores) — non-model
    variants get fixed placeholder stats, so they never need a real score.
    Backend-correct: the operator dispatch inside ``ops.scores`` follows
    ``ops._is_gpu``.
    """
    y_dev = ops.project(ops.xp.asarray(np.asarray(y, dtype=DTYPE).copy()))
    y_norm2 = _dot(y_dev, y_dev)
    if y_norm2 <= 0.0:
        raise RuntimeError("phenotype has nonpositive projected norm")

    rows = []
    for (chrom, grg), all_stats in zip(chrom_grgs, chrom_all_stats):
        vinv_scale = float(calibration.vinv_scale_by_chrom[chrom])
        if vinv_scale <= 0.0:
            raise RuntimeError(f"nonpositive VinvScaleFactor for chr{chrom}: {vinv_scale}")

        # Compact (model-variant-only) scores via the existing per-chrom op.
        # ops.scores internally projects; passing already-projected vectors is
        # safe because the orthogonal projection is idempotent.
        linreg_scores_compact = ops.scores(chrom, y_dev)
        lmm_scores_compact    = ops.scores(chrom, residuals[chrom])

        xp_s = _array_module(linreg_scores_compact)
        if hasattr(xp_s, "asnumpy"):
            linreg_scores_compact = xp_s.asnumpy(linreg_scores_compact)
            lmm_scores_compact    = xp_s.asnumpy(lmm_scores_compact)
        else:
            linreg_scores_compact = np.asarray(linreg_scores_compact)
            lmm_scores_compact    = np.asarray(lmm_scores_compact)

        pos_by_local = ops._local_idx_to_pos[chrom]
        freqs_full = _to_np(allele_frequencies_cupy(grg)) if ops._is_cupy else allele_frequencies(grg)

        for stat in all_stats:
            local_idx = stat.local_idx
            mut = grg.get_mutation_by_id(local_idx)
            snp_id = f"{chrom}:{mut.position}:{mut.allele}:{mut.ref_allele}"
            a1freq = float(freqs_full[local_idx])

            if not stat.is_model_variant:
                rows.append({
                    "SNP_ID": snp_id, "CHROM": chrom, "BP": mut.position,
                    "ALLELE1": mut.allele, "ALLELE0": mut.ref_allele, "A1FREQ": a1freq,
                    "CHISQ_LINREG": BOLT_BAD_SNP_STAT, "P_LINREG": 1.0,
                    "BETA": 0.0, "SE": float("nan"),
                    "CHISQ_BOLT_LMM_INF": BOLT_BAD_SNP_STAT, "P_BOLT_LMM_INF": 1.0,
                })
                continue

            pos = pos_by_local[local_idx]
            linreg_score = float(linreg_scores_compact[pos])
            lmm_score    = float(lmm_scores_compact[pos])

            ns  = float(stat.norm_scale)
            pn2 = float(stat.proj_norm2)
            xn2 = float(stat.x_norm2)

            linreg_chi2 = (linreg_score * linreg_score) / y_norm2 / xn2 * float(ops.dim)
            linreg_p    = float(_scipy_chi2.sf(linreg_chi2, df=1))

            h_score_raw    = lmm_score / ns
            vinv_score_raw = h_score_raw / float(fit.sigma_g2)
            lmm_chi2       = ((vinv_score_raw / vinv_scale) ** 2) / pn2
            beta           = vinv_score_raw / (pn2 * vinv_scale * vinv_scale)
            se             = 1.0 / (math.sqrt(pn2) * vinv_scale)
            lmm_p          = float(_scipy_chi2.sf(lmm_chi2, df=1))

            rows.append({
                "SNP_ID": snp_id, "CHROM": chrom, "BP": mut.position,
                "ALLELE1": mut.allele, "ALLELE0": mut.ref_allele, "A1FREQ": a1freq,
                "CHISQ_LINREG": linreg_chi2, "P_LINREG": linreg_p,
                "BETA": beta, "SE": se,
                "CHISQ_BOLT_LMM_INF": lmm_chi2, "P_BOLT_LMM_INF": lmm_p,
            })

    return pd.DataFrame(rows)
