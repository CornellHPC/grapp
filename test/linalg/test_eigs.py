from grapp.linalg import (
    MatrixSelection,
    PCs,
    eigs as grg_eigs,
    get_pcs_propca,
    sort_by_eigvalues,
)

import numpy
import os
import unittest
from scipy.sparse.linalg import eigs as scipy_eigs
import sys

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(THIS_DIR, ".."))
from testing_utils import construct_grg, grg2X, standardize_X
from grapp.backends import HAS_MKL, HAS_CUSPARSE

JOBS = 4
CLEANUP = True

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
INPUT_DIR = os.path.join(THIS_DIR, "input")


class _PCATestBase:

    BACKEND_CLASS = None
    BACKEND_KWARGS = {}
    GRG_TO_DENSE_METHOD = None

    @classmethod
    def setUpClass(cls):
        cls.grg_filename = construct_grg("test-200-samples.vcf.gz", "test.pca.grg")
        # Up edges needed for grg2X
        cls.grg = cls.BACKEND_CLASS(cls.grg_filename, **cls.BACKEND_KWARGS)

    def test_eigvals(self):
        X_stand = standardize_X(self.GRG_TO_DENSE_METHOD(self.grg, diploid=True))

        D = X_stand.T @ X_stand
        evals, evects = scipy_eigs(D, k=15, which="LR")

        # grg_eigs does this for us, because scipy _does not guarantee_ that these are in the
        # correct order.
        sort_by_eigvalues(evals, evects)

        grg_evals, grg_evects = grg_eigs(MatrixSelection.XTX, self.grg, 15)
        numpy.testing.assert_array_almost_equal(
            numpy.flip(numpy.sort(grg_evals)), grg_evals, 10
        )
        numpy.testing.assert_array_almost_equal(evals, grg_evals, 3)
        for ev, gev in zip(evects.T, grg_evects.T):
            # Vectors may differ by sign.
            self.assertTrue(numpy.allclose(ev, gev) or numpy.allclose(-ev, gev))

    # Just make sure PCA succeeds
    def test_pca_smoketest(self):
        # Not all recent versions of scipy support passing in a random number generator,
        # so we need to do this instead for testing purposes
        numpy.random.seed(42)
        scores = PCs(self.grg, k=20).to_numpy()
        pca_expect = numpy.loadtxt(os.path.join(INPUT_DIR, "pca.expected.txt"))
        numpy.testing.assert_allclose(numpy.abs(scores), numpy.abs(pca_expect))

    # Just make sure PCA succeeds
    def test_propca_smoketest(self):
        scores, _, _ = get_pcs_propca(self.grg, k=20, convergence_lim=1e-5)
        propca_expect = numpy.loadtxt(os.path.join(INPUT_DIR, "propca.expected.txt"))
        numpy.testing.assert_allclose(
            numpy.abs(scores), numpy.abs(propca_expect), atol=0.005
        )

    @classmethod
    def tearDownClass(cls):
        if CLEANUP:
            os.remove(cls.grg_filename)

class TestPCA_ImmutableGRG(_PCATestBase, unittest.TestCase):
    from grapp.backends import IMMUTABLE_GRG
    BACKEND_CLASS = IMMUTABLE_GRG
    BACKEND_KWARGS = {"load_up_edges": True}
    
    @staticmethod
    def GRG_TO_DENSE_METHOD(grg, diploid=True):
        return grg2X(grg, diploid)

@unittest.skipUnless(HAS_MKL, "MKL not available")
class TestPCA_SPMV_MKL(_PCATestBase, unittest.TestCase):
    from grapp.backends.spmv import SPMV_GRG_MKL
    BACKEND_CLASS = SPMV_GRG_MKL
    BACKEND_KWARGS = {"load_up_edges": True, "nthreads": 64}

    @staticmethod
    def GRG_TO_DENSE_METHOD(grg, diploid=True):
        return grg2X(grg._grg, diploid)

@unittest.skipUnless(HAS_CUSPARSE, "cuSPARSE not available")
class TestPCA_SPMV_cuSparse(_PCATestBase, unittest.TestCase):
    from grapp.backends.spmv import SPMV_GRG_cuSparse
    BACKEND_CLASS = SPMV_GRG_cuSparse
    BACKEND_KWARGS = {"load_up_edges": True}

    @staticmethod
    def GRG_TO_DENSE_METHOD(grg, diploid=True):
        return grg2X(grg._grg, diploid)


# ══════════════════════════════════════════════════════════════════════════════
# Stress Test: Compare backends against IMMUTABLE_GRG ground truth
# ══════════════════════════════════════════════════════════════════════════════
# Controlled by environment variables:
#   GRAPP_STRESS_TEST_INPUT  - GRG input file (VCF or GRG)
#   GRAPP_STRESS_TEST_RUNS   - Number of test iterations (default: 10)
# ══════════════════════════════════════════════════════════════════════════════

STRESS_INPUT = os.environ.get("GRAPP_STRESS_TEST_INPUT")
STRESS_RUNS = int(os.environ.get("GRAPP_STRESS_TEST_RUNS", "100"))


class _PCAStressTestBase:
    """Base class for stress tests comparing backends against IMMUTABLE_GRG."""

    BACKEND_CLASS = None
    BACKEND_KWARGS = {}

    @classmethod
    def setUpClass(cls):
        from grapp.backends import IMMUTABLE_GRG
        
        # Construct GRG from input file
        input_file = STRESS_INPUT
        is_grg = input_file.endswith('.grg')
        
        if is_grg:
            cls.grg_filename = input_file
        else:
            # Construct from VCF
            cls.grg_filename = construct_grg(
                input_file,
                output_file=f"{os.path.basename(input_file)}.stress.grg",
                is_test_input=False
            )
        
        # Create ground truth backend (IMMUTABLE_GRG)
        cls.grg_truth = IMMUTABLE_GRG(cls.grg_filename, load_up_edges=True)
        
        # Create test backend
        cls.grg_test = cls.BACKEND_CLASS(cls.grg_filename, **cls.BACKEND_KWARGS)
        
        # Compute ground truth results ONCE
        print(f"\n{'='*80}")
        print(f"STRESS TEST CONFIGURATION:")
        print(f"  Input file:      {cls.grg_filename}")
        print(f"  Num samples:     {cls.grg_truth.num_samples}")
        print(f"  Num mutations:   {cls.grg_truth.num_mutations}")
        print(f"  Test runs:       {STRESS_RUNS}")
        print(f"  Backend:         {cls.BACKEND_CLASS.__name__}")
        print(f"{'='*80}")
        
        # Pre-compute ground truth for PCA
        print("Computing ground truth PCA...", flush=True)
        cls.k_pca = min(20, cls.grg_truth.num_mutations - 1)
        numpy.random.seed(42)  # Fixed seed for ground truth
        cls.truth_pca_scores = PCs(cls.grg_truth, k=cls.k_pca).to_numpy()
        
        # Pre-compute ground truth for eigenvalues
        print("Computing ground truth eigenvalues...", flush=True)
        cls.k_eigs = min(15, cls.grg_truth.num_mutations - 1)
        numpy.random.seed(100)  # Fixed seed for ground truth
        cls.truth_evals, cls.truth_evects = grg_eigs(
            MatrixSelection.XTX, cls.grg_truth, cls.k_eigs
        )
        print("Ground truth computed.\n", flush=True)

    def test_stress_pca_consistency(self):
        """Run PCA multiple times and verify consistency with ground truth."""
        for run in range(STRESS_RUNS):
            # Use same seed as ground truth for deterministic comparison
            numpy.random.seed(42)
            
            # Compute with test backend
            test_scores = PCs(self.grg_test, k=self.k_pca).to_numpy()
            
            # Compare against pre-computed ground truth (allow sign flips on eigenvectors)
            numpy.testing.assert_allclose(
                numpy.abs(self.truth_pca_scores),
                numpy.abs(test_scores),
                rtol=1e-3,
                atol=1e-5,
                err_msg=f"Run {run + 1}/{STRESS_RUNS} failed"
            )
            
            print(f"  ✓ PCA run {run + 1}/{STRESS_RUNS} passed", flush=True)

    def test_stress_eigenvalues(self):
        """Run eigenvalue computation multiple times and verify consistency."""
        for run in range(STRESS_RUNS):
            # Use same seed as ground truth for deterministic comparison
            numpy.random.seed(100)
            
            # Compute with test backend
            test_evals, test_evects = grg_eigs(MatrixSelection.XTX, self.grg_test, self.k_eigs)
            
            # Compare eigenvalues against pre-computed ground truth
            numpy.testing.assert_allclose(
                self.truth_evals,
                test_evals,
                rtol=1e-3,
                atol=1e-5,
                err_msg=f"Eigenvalues differ at run {run + 1}/{STRESS_RUNS}"
            )
            
            # Compare eigenvectors (allow sign flips)
            for i, (truth_ev, test_ev) in enumerate(zip(self.truth_evects.T, test_evects.T)):
                self.assertTrue(
                    numpy.allclose(truth_ev, test_ev, rtol=1e-3, atol=1e-5) or
                    numpy.allclose(-truth_ev, test_ev, rtol=1e-3, atol=1e-5),
                    f"Eigenvector {i} differs at run {run + 1}/{STRESS_RUNS}"
                )
            
            print(f"  ✓ Eigenvalue run {run + 1}/{STRESS_RUNS} passed", flush=True)

    @classmethod
    def tearDownClass(cls):
        # Only clean up if we constructed the GRG (not if using existing .grg)
        if CLEANUP and not STRESS_INPUT.endswith('.grg'):
            if os.path.exists(cls.grg_filename):
                os.remove(cls.grg_filename)


@unittest.skipUnless(STRESS_INPUT and HAS_MKL, "Stress test disabled or MKL not available")
class TestPCAStress_SPMV_MKL(_PCAStressTestBase, unittest.TestCase):
    from grapp.backends.spmv import SPMV_GRG_MKL
    BACKEND_CLASS = SPMV_GRG_MKL
    BACKEND_KWARGS = {"load_up_edges": True, "nthreads": 64}


@unittest.skipUnless(STRESS_INPUT and HAS_CUSPARSE, "Stress test disabled or cuSPARSE not available")
class TestPCAStress_SPMV_cuSparse(_PCAStressTestBase, unittest.TestCase):
    from grapp.backends.spmv import SPMV_GRG_cuSparse
    BACKEND_CLASS = SPMV_GRG_cuSparse
    BACKEND_KWARGS = {"load_up_edges": True}
