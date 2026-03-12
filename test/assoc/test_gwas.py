import math
import numpy
import os
import pandas as pd
import sys
import tempfile
import unittest
from collections import defaultdict

from grapp.assoc import linear_assoc_no_covar, linear_assoc_covar, read_pheno
from grapp.linalg import PCs
from grapp.util.filter import grg_save_samples
from grapp.backends import HAS_MKL, HAS_CUSPARSE

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(THIS_DIR, ".."))
from testing_utils import construct_grg, complete_sample_sets

CLEANUP = True
INPUT_DIR = os.path.join(THIS_DIR, "input")


class _GWASTestBase:

    BACKEND_CLASS = None
    BACKEND_KWARGS = {}

    @classmethod
    def setUpClass(cls):
        cls.grg_filename = construct_grg("test-200-samples.vcf.gz", "test.gwas.grg")
        cls.grg = cls.BACKEND_CLASS(cls.grg_filename, **cls.BACKEND_KWARGS)
        cls.pheno_path = os.path.join(INPUT_DIR, "phenotypes.txt")
        assert os.path.isfile(cls.pheno_path)

        baseline_path = os.path.join(INPUT_DIR, "gwas.baseline.txt")
        cls.gwas = pd.read_csv(baseline_path, delimiter="\t")

        covar_baseline_path = os.path.join(INPUT_DIR, "covar.baseline.txt")
        cls.covar = pd.read_csv(covar_baseline_path, delimiter="\t")

        cls.Y = read_pheno(cls.pheno_path)

    def test_gwas_no_covar_vs_grg(self):
        df_py = linear_assoc_no_covar(self.grg, self.Y)
        df_grg = self.gwas

        # Check same number of rows
        self.assertEqual(
            len(df_py),
            len(df_grg),
            f"Mismatch in row count: Python has {len(df_py)}, GRG has {len(df_grg)}",
        )

        # Columns to compare
        columns = ["COUNT", "BETA", "SE", "T", "P"]
        atol = 1e-4
        rtol = 1e-3

        for i in range(len(df_py)):
            acount = df_py.iloc[i]["COUNT"]
            should_be_nan = acount == 0 or acount == self.grg.num_samples
            for col in columns:
                val_py = df_py.iloc[i][col]
                val_grg = df_grg.iloc[i][col]
                diff = abs(val_py - val_grg)
                rel_err = diff / (abs(val_grg) + 1e-8)
                fail_msg = (
                    f"Row {i}, column '{col}' mismatch:\n"
                    f"Python: {val_py}, GRG: {val_grg}, "
                    f"abs_diff={diff}, rel_err={rel_err}"
                )
                if should_be_nan and col != "COUNT":
                    assert math.isnan(val_py), fail_msg
                    continue
                assert not math.isnan(val_py), fail_msg
                assert diff <= atol or rel_err <= rtol, fail_msg

    def test_gwas_covar(self):
        C = PCs(self.grg, 10, unitvar=False).to_numpy()
        df_nonstd = linear_assoc_covar(self.grg, self.Y, C)
        self.assertFalse(numpy.any(numpy.isinf(df_nonstd["BETA"].to_numpy())))
        # Compare against the baseline of known values. This is just a test to make sure
        # nothing changes.
        numpy.testing.assert_allclose(df_nonstd["BETA"], self.covar["BETA"])
        numpy.testing.assert_allclose(df_nonstd["P"], self.covar["P"], rtol=0.01)
        df_nonstd_regress = linear_assoc_covar(self.grg, self.Y, C, method="regress")

        qr_nans = df_nonstd["BETA"].isna().sum()
        reg_nans = df_nonstd_regress["BETA"].isna().sum()
        self.assertLess(abs(qr_nans - reg_nans), 10)

        # These methods are not identical. But we expect them to be "pretty close", which
        # we measure as 90% of SNPs having a relative difference of less than 0.2 between
        # the two methods.
        total = self.grg.num_mutations - max(qr_nans, reg_nans)
        pretty_close = 0
        reldiff = (df_nonstd["BETA"] - df_nonstd_regress["BETA"]).abs() / df_nonstd[
            "BETA"
        ]
        for r in reldiff:
            if not math.isnan(r) and not math.isinf(r) and r <= 0.2:
                pretty_close += 1
        self.assertGreater(pretty_close / total, 0.9)

        df_std = linear_assoc_covar(self.grg, self.Y, C, standardize=True)
        self.assertEqual(len(df_nonstd), len(df_std))
        self.assertEqual(self.grg.num_mutations, len(df_std))

    def test_gwas_no_covar_dists(self):
        """
        The binomial method and the sample method should be very similar on neutral simulated data
        of large sample size. For small sample sizes (like this test), the deviation can be quite
        large.
        """
        df_sample = linear_assoc_no_covar(self.grg, self.Y)
        df_binomial = linear_assoc_no_covar(self.grg, self.Y, dist="binomial")

        sample_nans = df_sample["BETA"].isna().sum()
        binomial_nans = df_binomial["BETA"].isna().sum()
        self.assertEqual(sample_nans, binomial_nans)

        relerr = (df_binomial["BETA"] - df_sample["BETA"]) / df_sample["BETA"]
        small_errors = numpy.where((relerr < 0.20) & (relerr > -0.20))
        total = self.grg.num_mutations - sample_nans
        small_err_proportion = len(small_errors[0]) / total
        self.assertGreater(
            small_err_proportion, 0.99
        )  # 99% of errors are less than 20% relative error

    def test_gwas_no_covar_missing_Y(self):
        """
        When there are missing phenotypes (Y), they are represented as NaN. In this case, we need to scale
        the variance term in the GWAS appropriately. The calculation should be numerically close to removing
        those individuals from the GRG first, and then performing a GWAS.
        """
        # The individual list needs to be ordered, because GRG will retain the original order when down sampling.
        keep_indivs, ignore_indivs, keep_samples, ignore_samples = complete_sample_sets(
            self.grg, [22, 33, 54, 166, 167]
        )

        # Setup the missing data in the phenotype.
        Y_miss = self.Y.copy()
        Y_miss[ignore_indivs] = math.nan
        Y_kept = self.Y[keep_indivs]

        # Create the filtered GRG by just removing the individuals with missing Y
        filt_name = "test.gwas.missingY.grg"
        grg_save_samples(self.grg, filt_name, keep_samples)
        filt_grg = self.BACKEND_CLASS(filt_name, load_up_edges=False)
        self.assertEqual(filt_grg.num_individuals, Y_kept.shape[0])

        # Use binomial estimates
        true_sample_df = linear_assoc_no_covar(filt_grg, Y_kept, dist="binomial")
        mask_sample_df = linear_assoc_no_covar(self.grg, Y_miss, dist="binomial")
        true_nans = true_sample_df["BETA"].isna().sum()
        mask_nans = mask_sample_df["BETA"].isna().sum()
        self.assertEqual(true_nans, mask_nans)
        numpy.testing.assert_allclose(true_sample_df["BETA"], mask_sample_df["BETA"])

        # Use sample estimates (default dist)
        true_sample_df = linear_assoc_no_covar(filt_grg, Y_kept)
        mask_sample_df = linear_assoc_no_covar(self.grg, Y_miss)
        true_nans = true_sample_df["BETA"].isna().sum()
        mask_nans = mask_sample_df["BETA"].isna().sum()
        self.assertGreaterEqual(true_nans, mask_nans)
        true_beta = true_sample_df["BETA"].to_numpy()
        mask_beta = mask_sample_df["BETA"].to_numpy()
        relative_err_thresh = 0.25
        num_exceeded = 0
        num_within = 0
        for i in range(true_beta.shape[0]):
            # The masked version is estimating the diag(X^T X) value from the whole graph, so
            # the downsampled GRG can have freq=0 for the mutation, but the masked version will just
            # estimate a really small value.
            if math.isnan(true_beta[i]) and not math.isnan(mask_beta[i]):
                self.assertEqual(mask_sample_df.iloc[i]["COUNT"], 0)
            elif (
                abs((true_beta[i] - mask_beta[i]) / mask_beta[i]) > relative_err_thresh
            ):
                num_exceeded += 1
            else:
                num_within += 1
        # No more than 2% of beta shoulds exceed the relative error threshold
        self.assertLess(num_exceeded / (num_within + num_exceeded), 0.02)

    def test_read_pheno(self):
        # grg_pheno_sim emits two columns, with a header, tab-separated.
        GRG_PHENO_OUTPUT = """person_id\tphenotypes
0\t0.20630173530879656
1\t-1.433337296521211
2\t0.002598344575590285
3\t0.03609182039687409"""

        with tempfile.TemporaryDirectory() as tmpdirname:
            phen_file = os.path.join(tmpdirname, "test.phen")
            with open(phen_file, "w") as f:
                f.write(GRG_PHENO_OUTPUT)
            Y = read_pheno(phen_file)
            numpy.testing.assert_allclose(
                Y,
                [
                    0.20630173530879656,
                    -1.433337296521211,
                    0.002598344575590285,
                    0.03609182039687409,
                ],
            )

    @classmethod
    def tearDownClass(cls):
        if CLEANUP:
            os.remove(cls.grg_filename)

class TestGWAS_ImmutableGRG(_GWASTestBase, unittest.TestCase):
    from grapp.backends import IMMUTABLE_GRG
    BACKEND_CLASS = IMMUTABLE_GRG
    BACKEND_KWARGS = {"load_up_edges": True}


@unittest.skipUnless(HAS_MKL, "MKL not available")
class TestGWAS_SPMV_MKL(_GWASTestBase, unittest.TestCase):
    from grapp.backends.spmv import SPMV_GRG_MKL
    BACKEND_CLASS = SPMV_GRG_MKL
    BACKEND_KWARGS = {"load_up_edges": True, "nthreads": 64}
    @unittest.skip("save_subset not supported for SPMV_GRG")
    def test_gwas_no_covar_missing_Y(self):
        pass

@unittest.skipUnless(HAS_CUSPARSE, "cuSPARSE not available")
class TestGWAS_SPMV_cuSparse(_GWASTestBase, unittest.TestCase):
    from grapp.backends.spmv import SPMV_GRG_cuSparse
    BACKEND_CLASS = SPMV_GRG_cuSparse
    BACKEND_KWARGS = {"load_up_edges": True}
    @unittest.skip("save_subset not supported for SPMV_GRG")
    def test_gwas_no_covar_missing_Y(self):
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Stress Test: Compare backends against IMMUTABLE_GRG ground truth
# ══════════════════════════════════════════════════════════════════════════════
# Controlled by environment variables:
#   GRAPP_GWAS_STRESS_TEST_INPUT  - GRG input file (VCF or GRG)
#   GRAPP_GWAS_STRESS_TEST_PHENO  - Phenotype file (optional, random if not set)
#   GRAPP_GWAS_STRESS_TEST_RUNS   - Number of test iterations (default: 100)
# ══════════════════════════════════════════════════════════════════════════════

GWAS_STRESS_INPUT = os.environ.get("GRAPP_STRESS_TEST_INPUT")
GWAS_STRESS_PHENO = os.environ.get("GRAPP_STRESS_TEST_PHENO")
GWAS_STRESS_RUNS = int(os.environ.get("GRAPP_STRESS_TEST_RUNS", "100"))


class _GWASStressTestBase:
    """Base class for stress tests comparing backends against IMMUTABLE_GRG."""

    BACKEND_CLASS = None
    BACKEND_KWARGS = {}

    @classmethod
    def setUpClass(cls):
        from grapp.backends import IMMUTABLE_GRG
        
        # Construct GRG from input file
        input_file = GWAS_STRESS_INPUT
        is_grg = input_file.endswith('.grg')
        
        if is_grg:
            cls.grg_filename = input_file
        else:
            # Construct from VCF
            cls.grg_filename = construct_grg(
                input_file,
                output_file=f"{os.path.basename(input_file)}.gwas_stress.grg",
                is_test_input=False
            )
        
        # Create ground truth backend (IMMUTABLE_GRG)
        cls.grg_truth = IMMUTABLE_GRG(cls.grg_filename, load_up_edges=True)
        
        # Create test backend
        cls.grg_test = cls.BACKEND_CLASS(cls.grg_filename, **cls.BACKEND_KWARGS)
        
        # Load or generate phenotypes
        if GWAS_STRESS_PHENO and os.path.exists(GWAS_STRESS_PHENO):
            cls.Y = read_pheno(GWAS_STRESS_PHENO)
        else:
            # Generate random phenotypes
            numpy.random.seed(12345)
            cls.Y = numpy.random.randn(cls.grg_truth.num_individuals)
        
        # Compute ground truth results ONCE
        print(f"\n{'='*80}")
        print(f"GWAS STRESS TEST CONFIGURATION:")
        print(f"  Input file:      {cls.grg_filename}")
        print(f"  Num samples:     {cls.grg_truth.num_samples}")
        print(f"  Num individuals: {cls.grg_truth.num_individuals}")
        print(f"  Num mutations:   {cls.grg_truth.num_mutations}")
        print(f"  Phenotype file:  {GWAS_STRESS_PHENO or 'random'}")
        print(f"  Test runs:       {GWAS_STRESS_RUNS}")
        print(f"  Backend:         {cls.BACKEND_CLASS.__name__}")
        print(f"{'='*80}")
        
        # Pre-compute ground truth for GWAS without covariates
        print("Computing ground truth GWAS (no covariates)...", flush=True)
        cls.truth_gwas_no_covar = linear_assoc_no_covar(cls.grg_truth, cls.Y)
        
        # Pre-compute ground truth for GWAS with covariates (10 PCs)
        print("Computing ground truth GWAS (with covariates)...", flush=True)
        numpy.random.seed(42)
        cls.C = PCs(cls.grg_truth, 10, unitvar=False).to_numpy()
        cls.truth_gwas_covar = linear_assoc_covar(cls.grg_truth, cls.Y, cls.C)
        
        print("Ground truth computed.\n", flush=True)

    def test_stress_gwas_no_covar(self):
        """Run GWAS without covariates multiple times and verify consistency."""
        for run in range(GWAS_STRESS_RUNS):
            # Compute with test backend
            test_gwas = linear_assoc_no_covar(self.grg_test, self.Y)
            
            # Verify same number of mutations
            self.assertEqual(len(test_gwas), len(self.truth_gwas_no_covar))
            
            # Compare columns: COUNT, BETA, SE, T, P (vectorized comparison)
            for col in ["COUNT", "BETA", "SE", "T", "P"]:
                truth_col = self.truth_gwas_no_covar[col].to_numpy()
                test_col = test_gwas[col].to_numpy()
                
                # Check that NaNs match
                truth_nans = numpy.isnan(truth_col)
                test_nans = numpy.isnan(test_col)
                if not numpy.array_equal(truth_nans, test_nans):
                    # Find first mismatch for error message
                    mismatch_idx = numpy.where(truth_nans != test_nans)[0][0]
                    self.fail(f"Run {run + 1}: Row {mismatch_idx}, col {col}: "
                             f"NaN mismatch (truth={truth_col[mismatch_idx]}, test={test_col[mismatch_idx]})")
                
                # Compare non-NaN values (vectorized)
                valid_mask = ~truth_nans
                if valid_mask.any():
                    numpy.testing.assert_allclose(
                        truth_col[valid_mask],
                        test_col[valid_mask],
                        rtol=1e-3,
                        atol=1e-5,
                        err_msg=f"Run {run + 1}: {col} values differ"
                    )
            
            if (run + 1) % 10 == 0:  # Print every 10 runs
                print(f"  ✓ GWAS (no covar) run {run + 1}/{GWAS_STRESS_RUNS} passed", flush=True)

    def test_stress_gwas_with_covar(self):
        """Run GWAS with covariates multiple times and verify consistency."""
        for run in range(GWAS_STRESS_RUNS):
            # Use same seed as ground truth
            numpy.random.seed(42)
            
            # Compute with test backend
            test_gwas = linear_assoc_covar(self.grg_test, self.Y, self.C)
            
            # Verify same number of mutations
            self.assertEqual(len(test_gwas), len(self.truth_gwas_covar))
            
            # Compare BETA and P values (main columns of interest)
            for col in ["BETA", "P"]:
                truth_col = self.truth_gwas_covar[col].to_numpy()
                test_col = test_gwas[col].to_numpy()
                
                # Count NaNs - should be similar
                truth_nans = numpy.isnan(truth_col).sum()
                test_nans = numpy.isnan(test_col).sum()
                self.assertLess(abs(truth_nans - test_nans), 10,
                              f"Run {run + 1}: {col} has different NaN counts: truth={truth_nans}, test={test_nans}")
                
                # Compare non-NaN values
                valid_mask = ~(numpy.isnan(truth_col) | numpy.isnan(test_col))
                if valid_mask.any():
                    numpy.testing.assert_allclose(
                        truth_col[valid_mask],
                        test_col[valid_mask],
                        rtol=1e-3,
                        atol=1e-5,
                        err_msg=f"Run {run + 1}: {col} values differ"
                    )
            
            if (run + 1) % 10 == 0:  # Print every 10 runs
                print(f"  ✓ GWAS (with covar) run {run + 1}/{GWAS_STRESS_RUNS} passed", flush=True)

    @classmethod
    def tearDownClass(cls):
        # Only clean up if we constructed the GRG (not if using existing .grg)
        if CLEANUP and not GWAS_STRESS_INPUT.endswith('.grg'):
            if os.path.exists(cls.grg_filename):
                os.remove(cls.grg_filename)


@unittest.skipUnless(GWAS_STRESS_INPUT and HAS_MKL, "GWAS stress test disabled or MKL not available")
class TestGWASStress_SPMV_MKL(_GWASStressTestBase, unittest.TestCase):
    from grapp.backends.spmv import SPMV_GRG_MKL
    BACKEND_CLASS = SPMV_GRG_MKL
    BACKEND_KWARGS = {"load_up_edges": True, "nthreads": 64}


@unittest.skipUnless(GWAS_STRESS_INPUT and HAS_CUSPARSE, "GWAS stress test disabled or cuSPARSE not available")
class TestGWASStress_SPMV_cuSparse(_GWASStressTestBase, unittest.TestCase):
    from grapp.backends.spmv import SPMV_GRG_cuSparse
    BACKEND_CLASS = SPMV_GRG_cuSparse
    BACKEND_KWARGS = {"load_up_edges": True}

