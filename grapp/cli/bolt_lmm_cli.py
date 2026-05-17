import argparse
import os
import re
import sys

import numpy as np
import pandas

from grapp.assoc import read_pheno, read_plink_covariates
from grapp.assoc.bolt_lmm import bolt_lmm_inf
from grapp.cli.util import pandas_to_tsv
from grapp.grg_calculator import load_grg_calculator
from grapp.linalg.bolt_lmm_core import CovariateBasis


def add_options(subparser):
    subparser.add_argument(
        "grg_input",
        nargs="+",
        help="One GRG file per chromosome (order determines chromosome labels).",
    )
    subparser.add_argument(
        "-p",
        "--phenotypes",
        required=True,
        help="Phenotype file (PLINK/GCTA/GRG format).",
    )
    subparser.add_argument(
        "-c",
        "--covariates",
        default=None,
        help="Covariate file (PLINK .txt format).",
    )
    subparser.add_argument(
        "--covar-cols",
        nargs="*",
        default=(),
        help="Names of categorical covariate columns.",
    )
    subparser.add_argument(
        "--q-covar-cols",
        nargs="*",
        default=(),
        help="Names of quantitative covariate columns.",
    )
    subparser.add_argument(
        "--covar-max-levels",
        type=int,
        default=10,
        help="Maximum number of levels for categorical covariates (default: 10).",
    )
    subparser.add_argument(
        "--num-calib-snps",
        type=int,
        default=30,
        help="Number of calibration SNPs (default: 30).",
    )
    subparser.add_argument(
        "--mc-trials",
        type=int,
        default=3,
        help="Number of MC trials for variance component estimation (default: 3).",
    )
    subparser.add_argument(
        "--cg-tol",
        type=float,
        default=5e-4,
        help="Conjugate gradient convergence tolerance (default: 5e-4).",
    )
    subparser.add_argument(
        "--max-iter",
        type=int,
        default=10_000,
        help="Maximum CG iterations (default: 10000).",
    )
    subparser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed (default: 12345).",
    )
    subparser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="Thread count for parallel multi-chromosome operator (default: 1).",
    )
    subparser.add_argument(
        "-o",
        "--out-file",
        default=None,
        help="Output TSV file (default: <first_grg>.bolt_lmm.tsv).",
    )


def _chrom_label_from_path(path: str, fallback: int) -> int:
    """Extract chromosome number from a filename like chr19.grg, fallback to index."""
    basename = os.path.basename(path)
    m = re.search(r"chr(\d+)", basename, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return fallback


def run(args):
    grg_files = args.grg_input
    chrom_grgs = []
    for idx, path in enumerate(grg_files, start=1):
        label = _chrom_label_from_path(path, idx)
        grg = load_grg_calculator(path)
        chrom_grgs.append((label, grg))

    y = read_pheno(args.phenotypes)
    n = chrom_grgs[0][1].num_individuals
    assert len(y) == n, (
        f"Phenotype file has {len(y)} rows; GRG has {n} individuals"
    )

    if args.covariates is not None:
        covar_mat = read_plink_covariates(args.covariates)
        covariates = CovariateBasis.from_matrix(
            np.column_stack([np.ones(n), covar_mat]),
            covar_cols=args.covar_cols,
            q_covar_cols=args.q_covar_cols,
            covar_max_levels=args.covar_max_levels,
        )
    else:
        covariates = CovariateBasis.intercept_only(n)

    fit, calibration, _residuals, results_df = bolt_lmm_inf(
        chrom_grgs,
        y,
        covariates,
        num_calib_snps=args.num_calib_snps,
        mc_trials=args.mc_trials,
        cg_tol=args.cg_tol,
        max_iter=args.max_iter,
        seed=args.seed,
        threads=args.jobs,
    )

    print(
        f"BOLT-LMM-inf: h2={fit.h2:.4f} sigma_g2={fit.sigma_g2:.4g} "
        f"sigma_e2={fit.sigma_e2:.4g} calibration_factor={calibration.factor:.4f}",
        file=sys.stderr,
    )

    out_file = args.out_file
    if out_file is None:
        out_file = f"{os.path.basename(grg_files[0])}.bolt_lmm.tsv"
    pandas_to_tsv(out_file, results_df)
    print(f"Wrote results to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_options(parser)
    args = parser.parse_args()
    run(args)
