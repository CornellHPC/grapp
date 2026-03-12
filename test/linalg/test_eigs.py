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
