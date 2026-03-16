from grapp.linalg.ops_scipy import (
    MultiSciPyStdXOperator,
    MultiSciPyStdXTXOperator,
    MultiSciPyXOperator,
    MultiSciPyXTXOperator,
    SciPyStdXOperator,
    SciPyXXTOperator,
    SciPyStdXTXOperator,
    SciPyXOperator,
    SciPyXTXOperator,
)
from grapp.backends import IMMUTABLE_GRG, HAS_MKL, HAS_CUSPARSE
from grapp.util import allele_frequencies
from grapp.util.filter import grg_save_samples
import itertools
import numpy
import os
from grapp import Direction
import sys
import unittest

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(THIS_DIR, ".."))
from testing_utils import (
    construct_grg,
    standardize_X,
    grg2X,
    split_and_load,
    complete_sample_sets,
)

CLEANUP = True
JOBS = 4

# Absolute error tolerated between numpy and GRG methods.
ABSOLUTE_TOLERANCE = 1e-10


class TestLinearOperators(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grg_filename = construct_grg("test-200-samples.vcf.gz", "test.linop.grg")
        cls.grg = IMMUTABLE_GRG(cls.grg_filename, load_up_edges=True)

        numpy.random.seed(42)

    def test_simple_op(self):
        """
        The simple operator works on the genotype matrix without any modification. In this case,
        the diploid and haploid formulations should product the same result.
        """
        K = 20  # Use 20 random vectors for test.
        random_input = numpy.random.standard_normal((K, self.grg.num_mutations)).T

        X_hap = grg2X(self.grg, diploid=False)
        numpy_hap_result = numpy.matmul(X_hap, random_input)

        X_dip = grg2X(self.grg, diploid=True)
        self.assertNotEqual(X_hap.shape, X_dip.shape)
        self.assertEqual(numpy.max(X_hap), 1)
        self.assertEqual(numpy.max(X_dip), 2)
        numpy_dip_result = numpy.matmul(X_dip, random_input)
        self.assertAlmostEqual(numpy.sum(numpy_hap_result), numpy.sum(numpy_dip_result))

        grg_hap_op = SciPyXOperator(
            self.grg, Direction.UP, haploid=True
        )
        grg_hap_result = grg_hap_op._matmat(random_input)
        numpy.testing.assert_allclose(grg_hap_result, numpy_hap_result)

    def test_XXT(self):
        K = 20  # Use 20 random vectors for test.
        random_input = numpy.random.standard_normal((K, self.grg.num_samples)).T

        X_hap = grg2X(self.grg, diploid=False)
        numpy_hap_result = numpy.matmul(X_hap @ X_hap.T, random_input)

        grg_hap_op = SciPyXXTOperator(self.grg, haploid=True)
        grg_hap_result = grg_hap_op._matmat(random_input)
        numpy.testing.assert_allclose(grg_hap_result, numpy_hap_result)

        random_input = numpy.random.standard_normal((K, self.grg.num_individuals)).T
        X_dip = grg2X(self.grg, diploid=True)
        numpy_dip_result = numpy.matmul(X_dip @ X_dip.T, random_input)
        grg_dip_op = SciPyXXTOperator(self.grg, haploid=False)
        grg_dip_result = grg_dip_op._matmat(random_input)
        numpy.testing.assert_allclose(grg_dip_result, numpy_dip_result)

    def test_standardized_op_X(self):
        K = 20  # Use 20 random vectors for test.
        random_input = numpy.random.standard_normal((K, self.grg.num_mutations)).T

        X = grg2X(self.grg, diploid=True)
        X_stand = standardize_X(X)
        numpy_result = numpy.matmul(X_stand, random_input)

        freqs = allele_frequencies(self.grg)
        grg_op = SciPyStdXOperator(self.grg, Direction.UP, freqs)
        grg_result = grg_op._matmat(random_input)

        self.assertFalse(numpy.any(numpy.isinf(grg_result)))
        self.assertFalse(numpy.any(numpy.isinf(numpy_result)))
        self.assertFalse(numpy.any(numpy.isnan(grg_result)))
        self.assertFalse(numpy.any(numpy.isnan(numpy_result)))
        numpy.testing.assert_allclose(grg_result, numpy_result, atol=ABSOLUTE_TOLERANCE)

    def test_standardized_op_XT(self):
        K = 20  # Use 20 random vectors for test.
        random_input = numpy.random.standard_normal((K, self.grg.num_individuals)).T

        X = grg2X(self.grg, diploid=True)
        XT_stand = standardize_X(X).T
        numpy_result = numpy.matmul(XT_stand, random_input)

        freqs = allele_frequencies(self.grg)
        grg_op = SciPyStdXOperator(self.grg, Direction.DOWN, freqs)
        grg_result = grg_op._matmat(random_input)

        self.assertFalse(numpy.any(numpy.isinf(grg_result)))
        self.assertFalse(numpy.any(numpy.isinf(numpy_result)))
        self.assertFalse(numpy.any(numpy.isnan(grg_result)))
        self.assertFalse(numpy.any(numpy.isnan(numpy_result)))
        numpy.testing.assert_allclose(grg_result, numpy_result, atol=ABSOLUTE_TOLERANCE)

    def test_standardized_op_XtX(self):
        K = 20  # Number of random vectors for test.
        random_input = numpy.random.standard_normal((K, self.grg.num_mutations)).T

        X = grg2X(self.grg, diploid=True)
        X_stand = standardize_X(X)
        XtX = X_stand.T @ X_stand
        # (NxM)x(MxK) == NxK
        numpy_result = numpy.matmul(XtX, random_input)

        freqs = allele_frequencies(self.grg)
        grg_op = SciPyStdXTXOperator(self.grg, freqs)
        grg_result = grg_op._matmat(random_input)

        self.assertFalse(numpy.any(numpy.isinf(grg_result)))
        self.assertFalse(numpy.any(numpy.isinf(numpy_result)))
        self.assertFalse(numpy.any(numpy.isnan(grg_result)))
        self.assertFalse(numpy.any(numpy.isnan(numpy_result)))
        numpy.testing.assert_allclose(grg_result, numpy_result, atol=ABSOLUTE_TOLERANCE)

        # Reversed result should be identical, because XtX.T == XtX
        numpy_result = numpy.matmul(XtX.T, random_input)
        grg_result = grg_op._rmatmat(random_input)

        self.assertFalse(numpy.any(numpy.isinf(grg_result)))
        self.assertFalse(numpy.any(numpy.isinf(numpy_result)))
        self.assertFalse(numpy.any(numpy.isnan(grg_result)))
        self.assertFalse(numpy.any(numpy.isnan(numpy_result)))
        numpy.testing.assert_allclose(grg_result, numpy_result, atol=ABSOLUTE_TOLERANCE)

    def test_multi_ops(self):
        """
        Test that the operators that work with multiple GRGs produce the same result
        as ones that work with a single GRG.
        """

        # Split the graph and get the multiple GRGs, for testing all of the below.
        test_dir = "test.multi_ops.split"
        grgs = split_and_load(self.grg_filename, test_dir, 1_000_000, JOBS, CLEANUP)

        #### Direction == UP
        K = 10
        random_input = numpy.random.standard_normal((K, self.grg.num_mutations)).T

        # Result on the full graph.
        grg_op = SciPyXOperator(self.grg, Direction.UP, haploid=False)
        full_dip_result = grg_op._matmat(random_input)
        # Result on the split graph
        multi_op = MultiSciPyXOperator(
            grgs, Direction.UP, haploid=False, threads=JOBS
        )
        self.assertEqual(multi_op.shape, grg_op.shape)
        split_dip_result = multi_op._matmat(random_input)
        # Test equality
        numpy.testing.assert_allclose(full_dip_result, split_dip_result)

        #### Direction == DOWN
        random_input = numpy.random.standard_normal((K, self.grg.num_individuals)).T
        # Reverse from above
        full_dip_result = grg_op._rmatmat(random_input)
        split_dip_result = multi_op._rmatmat(random_input)
        numpy.testing.assert_allclose(full_dip_result, split_dip_result)

        # Result on the full graph.
        grg_op = SciPyXOperator(self.grg, Direction.DOWN, haploid=False)
        full_dip_result = grg_op._matmat(random_input)
        # Result on the split graph
        multi_op = MultiSciPyXOperator(
            grgs, Direction.DOWN, haploid=False, threads=JOBS
        )
        self.assertEqual(multi_op.shape, grg_op.shape)
        split_dip_result = multi_op._matmat(random_input)
        # Test equality
        numpy.testing.assert_allclose(full_dip_result, split_dip_result)

        # Vector version
        vec_result = grg_op._matvec(random_input[:, 1])
        split_vec_result = multi_op._matvec(random_input[:, 1])
        numpy.testing.assert_allclose(
            vec_result, split_vec_result, atol=ABSOLUTE_TOLERANCE
        )

        #### XTX non-standardized
        random_input = numpy.random.standard_normal((K, self.grg.num_mutations)).T
        # Result on the full graph.
        grg_op = SciPyXTXOperator(self.grg, haploid=False)
        full_dip_result = grg_op._matmat(random_input)
        # Result on the split graph
        multi_op = MultiSciPyXTXOperator(grgs, haploid=False, threads=JOBS)
        self.assertEqual(multi_op.shape, grg_op.shape)
        split_dip_result = multi_op._matmat(random_input)
        # Test equality
        numpy.testing.assert_allclose(full_dip_result, split_dip_result)

        # Vector version
        vec_result = grg_op._matvec(random_input[:, 1])
        split_vec_result = multi_op._matvec(random_input[:, 1])
        numpy.testing.assert_allclose(
            vec_result, split_vec_result, atol=ABSOLUTE_TOLERANCE
        )

        #### X standardized (UP)
        random_input = numpy.random.standard_normal((K, self.grg.num_mutations)).T
        # Result on the full graph.
        freqs = allele_frequencies(self.grg)
        grg_op = SciPyStdXOperator(
            self.grg, Direction.UP, freqs, haploid=False
        )
        full_dip_result = grg_op._matmat(random_input)
        # Result on the split graph
        freq_list = list(map(allele_frequencies, grgs))
        multi_op = MultiSciPyStdXOperator(
            grgs, Direction.UP, freq_list, haploid=False, threads=JOBS
        )
        self.assertEqual(multi_op.shape, grg_op.shape)
        split_dip_result = multi_op._matmat(random_input)
        # Test equality
        numpy.testing.assert_allclose(full_dip_result, split_dip_result)

        # Vector version
        vec_result = grg_op._matvec(random_input[:, 1])
        split_vec_result = multi_op._matvec(random_input[:, 1])
        numpy.testing.assert_allclose(
            vec_result, split_vec_result, atol=ABSOLUTE_TOLERANCE
        )

        #### XTX standardized
        random_input = numpy.random.standard_normal((K, self.grg.num_mutations)).T
        # Result on the full graph.
        grg_op = SciPyStdXTXOperator(self.grg, freqs, haploid=False)
        full_dip_result = grg_op._matmat(random_input)
        # Result on the split graph
        multi_op = MultiSciPyStdXTXOperator(
            grgs, freq_list, haploid=False, threads=JOBS
        )
        self.assertEqual(multi_op.shape, grg_op.shape)
        split_dip_result = multi_op._matmat(random_input)
        # Test equality
        numpy.testing.assert_allclose(
            full_dip_result, split_dip_result, atol=ABSOLUTE_TOLERANCE
        )

        # Vector version
        vec_result = grg_op._matvec(random_input[:, 1])
        split_vec_result = multi_op._matvec(random_input[:, 1])
        numpy.testing.assert_allclose(
            vec_result, split_vec_result, atol=ABSOLUTE_TOLERANCE
        )

        ### Test with a contiguous mutation filter
        total_muts = sum([g.num_mutations for g in grgs])
        keep_mutations = list(range(total_muts // 2))
        random_input = numpy.random.standard_normal((K, len(keep_mutations))).T
        grg_op = SciPyXOperator(
            self.grg,
            Direction.UP,
            haploid=False,
            mutation_filter=keep_mutations,
        )
        full_dip_result = grg_op._matmat(random_input)
        multi_op = MultiSciPyXOperator(
            grgs,
            Direction.UP,
            haploid=False,
            mutation_filter=keep_mutations,
            threads=JOBS,
        )
        self.assertEqual(multi_op.shape, grg_op.shape)
        split_dip_result = multi_op._matmat(random_input)
        numpy.testing.assert_allclose(full_dip_result, split_dip_result)

        # Vector version
        vec_result = grg_op._matvec(random_input[:, 1])
        split_vec_result = multi_op._matvec(random_input[:, 1])
        numpy.testing.assert_allclose(
            vec_result, split_vec_result, atol=ABSOLUTE_TOLERANCE
        )

        ### Test with a scattered mutation filter
        total_muts = sum([g.num_mutations for g in grgs])
        keep_mutations = [i * 2 for i in range(total_muts // 2)]
        random_input = numpy.random.standard_normal((K, len(keep_mutations))).T
        freqs = allele_frequencies(self.grg)
        freq_list = list(map(allele_frequencies, grgs))

        grg_op = SciPyStdXTXOperator(
            self.grg,
            freqs,
            haploid=False,
            mutation_filter=keep_mutations,
        )
        full_dip_result = grg_op._matmat(random_input)
        multi_op = MultiSciPyStdXTXOperator(
            grgs,
            freq_list,
            haploid=False,
            mutation_filter=keep_mutations,
            threads=JOBS,
        )
        self.assertEqual(multi_op.shape, grg_op.shape)
        split_dip_result = multi_op._matmat(random_input)
        numpy.testing.assert_allclose(
            full_dip_result, split_dip_result, atol=ABSOLUTE_TOLERANCE
        )
        # Vector version
        vec_result = grg_op._matvec(random_input[:, 1])
        split_vec_result = multi_op._matvec(random_input[:, 1])
        numpy.testing.assert_allclose(
            vec_result, split_vec_result, atol=ABSOLUTE_TOLERANCE
        )

        # Reverse direction
        # random_input = numpy.random.standard_normal((K, self.grg.num_individuals)).T
        rev_result = grg_op._rmatmat(random_input)
        split_rev_result = multi_op._rmatmat(random_input)
        numpy.testing.assert_allclose(
            rev_result, split_rev_result, atol=ABSOLUTE_TOLERANCE
        )

    def test_filtering(self):
        """
        Test the operators with filters enabled.
        """
        keep_mutations = list(range(self.grg.num_mutations // 2))

        K = 20  # Use 20 random vectors for test.
        random_mutvals = numpy.random.standard_normal((K, len(keep_mutations))).T
        random_mutvec = numpy.random.standard_normal(len(keep_mutations))
        random_sampvals = numpy.random.standard_normal((K, self.grg.num_individuals)).T

        X = grg2X(self.grg, diploid=True)
        X_std = standardize_X(X)
        X_dip = X[:, keep_mutations]
        X_dip_std = X_std[:, keep_mutations]
        freqs = allele_frequencies(self.grg)

        ### Non-standardized X operator
        # UP
        grg_dip_op = SciPyXOperator(
            self.grg, Direction.UP, mutation_filter=keep_mutations
        )
        numpy_dip_result = numpy.matmul(X_dip, random_mutvec)
        grg_dip_result = grg_dip_op._matvec(random_mutvec).squeeze()
        numpy.testing.assert_allclose(grg_dip_result, numpy_dip_result)
        numpy_dip_result = numpy.matmul(X_dip, random_mutvals)
        grg_dip_result = grg_dip_op._matmat(random_mutvals)
        numpy.testing.assert_allclose(grg_dip_result, numpy_dip_result)
        grg_dip_multi_op = MultiSciPyXOperator(
            [self.grg], Direction.UP, mutation_filter=keep_mutations
        )
        grg_dip_multi_result = grg_dip_multi_op._matmat(random_mutvals)
        numpy.testing.assert_allclose(grg_dip_multi_result, numpy_dip_result)

        # DOWN
        grg_dip_op = SciPyXOperator(
            self.grg, Direction.DOWN, mutation_filter=keep_mutations
        )
        numpy_dip_result = numpy.matmul(X_dip.T, random_sampvals)
        grg_dip_result = grg_dip_op._matmat(random_sampvals)
        numpy.testing.assert_allclose(grg_dip_result, numpy_dip_result)

        ### Non-standardized XTX operator
        grg_dip_op = SciPyXTXOperator(self.grg, mutation_filter=keep_mutations)
        numpy_dip_result = numpy.matmul(numpy.matmul(X_dip.T, X_dip), random_mutvec)
        grg_dip_result = grg_dip_op._matvec(random_mutvec).squeeze()
        numpy.testing.assert_allclose(grg_dip_result, numpy_dip_result)
        numpy_dip_result = numpy.matmul(numpy.matmul(X_dip.T, X_dip), random_mutvals)
        grg_dip_result = grg_dip_op._matmat(random_mutvals)
        numpy.testing.assert_allclose(grg_dip_result, numpy_dip_result)

        ### Standardized X operator
        # UP
        grg_op = SciPyStdXOperator(
            self.grg,
            Direction.UP,
            freqs,
            mutation_filter=keep_mutations,
        )
        numpy_dip_result = numpy.matmul(X_dip_std, random_mutvec)
        grg_dip_result = grg_op._matvec(random_mutvec).squeeze()
        numpy.testing.assert_allclose(grg_dip_result, numpy_dip_result)
        numpy_dip_result = numpy.matmul(X_dip_std, random_mutvals)
        grg_dip_result = grg_op._matmat(random_mutvals)
        numpy.testing.assert_allclose(grg_dip_result, numpy_dip_result)
        # DOWN
        numpy_dip_result = numpy.matmul(X_dip_std.T, random_sampvals)
        grg_op = SciPyStdXOperator(
            self.grg,
            Direction.DOWN,
            freqs,
            mutation_filter=keep_mutations,
        )
        grg_dip_result = grg_op._matmat(random_sampvals)
        numpy.testing.assert_allclose(grg_dip_result, numpy_dip_result)

    def test_missing(self):
        # Properties of the input data.
        MISSING_INDIVS = 21
        MISSING_SAMPLES = 25
        grg_filename = construct_grg("test-200-samples.miss.igd", "test.linop.miss.grg")
        grg = IMMUTABLE_GRG(grg_filename, load_up_edges=False)

        # X is the explicit genotype matrix, with allele frequency used for missing items. So the
        # only non-0,1,2 values should be missing items.
        X = grg2X(grg, diploid=True)
        self.assertEqual(
            len(numpy.where((X > 0) & (X != 1) & (X != 2))[0]), MISSING_INDIVS
        )

        # Create the operator, using the allele frequencies as the mean value for each Mutation
        freqs = allele_frequencies(grg, adjust_missing=True)
        X_op = SciPyXOperator(grg, Direction.UP, miss_values=freqs)

        #### UP direction (AX) ####
        K = 7
        rv = numpy.random.standard_normal((K, self.grg.num_individuals))
        # Using the explicit genotype matrix vs. GRG operator should produce identical results.
        numpy_result = rv @ X
        grg_result = rv @ X_op
        numpy.testing.assert_allclose(numpy_result, grg_result)

        #### DOWN direction (AX^T) ####
        rv = numpy.random.standard_normal((K, self.grg.num_mutations))
        # Using the explicit genotype matrix vs. GRG operator should produce identical results.
        numpy_result = rv @ X.T
        grg_result = rv @ X_op.T
        numpy.testing.assert_allclose(numpy_result, grg_result)

        # Just a sanity check: using the non-missingness-adjusted operator should cause failure.
        X_nomiss_op = SciPyXOperator(grg, Direction.UP)
        grg_result = rv @ X_nomiss_op.T
        self.assertFalse(numpy.allclose(numpy_result, grg_result))

    def test_ignore_samples(self):
        """
        We downsample a GRG explicitly, and then use an operator's mask to ignore the same individuals,
        and expect the matrix multiplication should produce the same result.
        """
        keep_indivs, ignore_indivs, keep_samples, ignore_samples = complete_sample_sets(
            self.grg, [4, 9, 101, 177]
        )

        filt_name = "test.ignore_samples.grg"
        grg_save_samples(self.grg, filt_name, keep_samples)
        filt_grg = IMMUTABLE_GRG(filt_name, load_up_edges=False)

        K = 17
        Y = numpy.random.standard_normal((K, self.grg.num_individuals))
        sub_Y = Y[:, keep_indivs]

        # Non-standardized operator
        truth_op = SciPyXOperator(filt_grg, Direction.UP)
        truth_matrix = sub_Y @ truth_op
        mask_op = SciPyXOperator(
            self.grg, Direction.UP, mask_samples=ignore_indivs
        )
        mask_matrix = Y @ mask_op
        numpy.testing.assert_allclose(truth_matrix, mask_matrix)

        # Standardized operator
        truth_freqs = allele_frequencies(filt_grg)
        mask_freqs = allele_frequencies(self.grg, mask_samples=ignore_samples)
        numpy.testing.assert_allclose(truth_freqs, mask_freqs, atol=ABSOLUTE_TOLERANCE)

        truth_op = SciPyStdXOperator(
            filt_grg, Direction.UP, truth_freqs
        )
        truth_matrix = sub_Y @ truth_op
        mask_op = SciPyStdXOperator(
            self.grg,
            Direction.UP,
            mask_freqs,
            mask_samples=ignore_indivs,
        )
        mask_matrix = Y @ mask_op
        numpy.testing.assert_allclose(
            truth_matrix, mask_matrix, atol=ABSOLUTE_TOLERANCE
        )

    @classmethod
    def tearDownClass(cls):
        if CLEANUP:
            os.remove(cls.grg_filename)

MULTI_INPUT_DIR = os.environ.get("GRAPP_MULTI_TEST_INPUT_DIR")


class _MultiLinearOperatorsTestBase:
    BACKEND_CLASS = None
    BACKEND_SUFFIX = None
    BACKEND_KWARGS = {}
    GROUND_TRUTH_CLASS = IMMUTABLE_GRG

    @classmethod
    def setUpClass(cls):
        if not MULTI_INPUT_DIR:
            raise unittest.SkipTest("GRAPP_MULTI_TEST_INPUT_DIR not set")

        grg_files = sorted(
            f for f in os.listdir(MULTI_INPUT_DIR) if f.endswith(".grg")
        )
        suffix_files = sorted(
            f for f in os.listdir(MULTI_INPUT_DIR) if f.endswith(cls.BACKEND_SUFFIX)
        ) if cls.BACKEND_SUFFIX else []

        assert len(grg_files) != 0 and len(grg_files) == len(suffix_files), (
            f"Mismatch or zero files: {len(grg_files)} .grg files, "
            f"{len(suffix_files)} {cls.BACKEND_SUFFIX!r} files in {MULTI_INPUT_DIR}"
        )

        if cls.BACKEND_SUFFIX != ".grg":
            import warnings
            warnings.warn(
                f"BACKEND_SUFFIX ({cls.BACKEND_SUFFIX!r}) differs from '.grg'. "
                "Ground truth and backend file lists are matched by sort order — "
                "ensure both sets contain the same samples in the same order.",
                UserWarning,
            )

        cls.ground_truth_files = [os.path.join(MULTI_INPUT_DIR, f) for f in grg_files]
        cls.backend_files = [os.path.join(MULTI_INPUT_DIR, f) for f in suffix_files]

        cls.ground_truth_grgs = [cls.GROUND_TRUTH_CLASS(f) for f in cls.ground_truth_files]
        cls.backend_grgs = [cls.BACKEND_CLASS(f, **cls.BACKEND_KWARGS) for f in cls.backend_files]

    def test_multi_X_operator(self):
        """
        MultiSciPyXOperator on the backend files must produce the same result
        as the same operator on the ground-truth IMMUTABLE_GRG files.
        """
        K = 10
        num_mutations = sum(g.num_mutations for g in self.ground_truth_grgs)
        num_individuals = self.ground_truth_grgs[0].num_individuals

        truth_op = MultiSciPyXOperator(self.ground_truth_grgs, Direction.UP, haploid=False)
        backend_op = MultiSciPyXOperator(self.backend_grgs, Direction.UP, haploid=False, mode="sequential")
        self.assertEqual(truth_op.shape, backend_op.shape)

        # _matmat UP
        random_input = numpy.random.standard_normal((K, num_mutations)).T
        numpy.testing.assert_allclose(
            backend_op._matmat(random_input),
            truth_op._matmat(random_input),
            atol=ABSOLUTE_TOLERANCE,
        )

        # _matvec UP
        numpy.testing.assert_allclose(
            backend_op._matvec(random_input[:, 0]),
            truth_op._matvec(random_input[:, 0]),
            atol=ABSOLUTE_TOLERANCE,
        )

        # _rmatmat (DOWN via rmatmat)
        random_input_down = numpy.random.standard_normal((K, num_individuals)).T
        numpy.testing.assert_allclose(
            backend_op._rmatmat(random_input_down),
            truth_op._rmatmat(random_input_down),
            atol=ABSOLUTE_TOLERANCE,
        )

        # Direction == DOWN
        truth_op_down = MultiSciPyXOperator(self.ground_truth_grgs, Direction.DOWN, haploid=False)
        backend_op_down = MultiSciPyXOperator(self.backend_grgs, Direction.DOWN, haploid=False, mode="sequential")
        self.assertEqual(truth_op_down.shape, backend_op_down.shape)

        numpy.testing.assert_allclose(
            backend_op_down._matmat(random_input_down),
            truth_op_down._matmat(random_input_down),
            atol=ABSOLUTE_TOLERANCE,
        )
        numpy.testing.assert_allclose(
            backend_op_down._matvec(random_input_down[:, 0]),
            truth_op_down._matvec(random_input_down[:, 0]),
            atol=ABSOLUTE_TOLERANCE,
        )

    def test_multi_XTX_operator(self):
        """
        MultiSciPyXTXOperator on the backend files must produce the same result
        as the same operator on the ground-truth IMMUTABLE_GRG files.
        """
        K = 10
        num_mutations = sum(g.num_mutations for g in self.ground_truth_grgs)

        truth_op = MultiSciPyXTXOperator(self.ground_truth_grgs, haploid=False)
        backend_op = MultiSciPyXTXOperator(self.backend_grgs, haploid=False, mode="sequential")
        self.assertEqual(truth_op.shape, backend_op.shape)

        random_input = numpy.random.standard_normal((K, num_mutations)).T

        # _matmat
        numpy.testing.assert_allclose(
            backend_op._matmat(random_input),
            truth_op._matmat(random_input),
            atol=ABSOLUTE_TOLERANCE,
        )

        # _matvec
        numpy.testing.assert_allclose(
            backend_op._matvec(random_input[:, 0]),
            truth_op._matvec(random_input[:, 0]),
            atol=ABSOLUTE_TOLERANCE,
        )

        # _rmatmat (XTX is symmetric, but still exercise the path)
        numpy.testing.assert_allclose(
            backend_op._rmatmat(random_input),
            truth_op._rmatmat(random_input),
            atol=ABSOLUTE_TOLERANCE,
        )

    def test_multi_std_X_operator(self):
        """
        MultiSciPyStdXOperator on the backend files must produce the same result
        as the same operator on the ground-truth IMMUTABLE_GRG files.

        We use truth_freqs for both operators so that any frequency-computation
        differences between backends don't confound the operator comparison.
        """
        K = 10
        num_mutations = sum(g.num_mutations for g in self.ground_truth_grgs)
        num_individuals = self.ground_truth_grgs[0].num_individuals

        # Use truth_freqs for both so we're testing the operator, not frequency computation.
        truth_freqs = [allele_frequencies(g) for g in self.ground_truth_grgs]

        truth_op = MultiSciPyStdXOperator(self.ground_truth_grgs, Direction.UP, truth_freqs, haploid=False)
        backend_op = MultiSciPyStdXOperator(self.backend_grgs, Direction.UP, truth_freqs, haploid=False, mode="sequential")
        self.assertEqual(truth_op.shape, backend_op.shape)

        # _matmat UP
        random_input = numpy.random.standard_normal((K, num_mutations)).T
        numpy.testing.assert_allclose(
            backend_op._matmat(random_input),
            truth_op._matmat(random_input),
            atol=ABSOLUTE_TOLERANCE,
        )

        # _matvec UP
        numpy.testing.assert_allclose(
            backend_op._matvec(random_input[:, 0]),
            truth_op._matvec(random_input[:, 0]),
            atol=ABSOLUTE_TOLERANCE,
        )

        # _rmatmat (DOWN via rmatmat)
        random_input_down = numpy.random.standard_normal((K, num_individuals)).T
        numpy.testing.assert_allclose(
            backend_op._rmatmat(random_input_down),
            truth_op._rmatmat(random_input_down),
            atol=ABSOLUTE_TOLERANCE,
        )

        # Direction == DOWN
        truth_op_down = MultiSciPyStdXOperator(self.ground_truth_grgs, Direction.DOWN, truth_freqs, haploid=False)
        backend_op_down = MultiSciPyStdXOperator(self.backend_grgs, Direction.DOWN, truth_freqs, haploid=False, mode="sequential")
        self.assertEqual(truth_op_down.shape, backend_op_down.shape)

        numpy.testing.assert_allclose(
            backend_op_down._matmat(random_input_down),
            truth_op_down._matmat(random_input_down),
            atol=ABSOLUTE_TOLERANCE,
        )
        numpy.testing.assert_allclose(
            backend_op_down._matvec(random_input_down[:, 0]),
            truth_op_down._matvec(random_input_down[:, 0]),
            atol=ABSOLUTE_TOLERANCE,
        )

    def test_multi_std_XTX_operator(self):
        """
        MultiSciPyStdXTXOperator on the backend files must produce the same result
        as the same operator on the ground-truth IMMUTABLE_GRG files.

        We use truth_freqs for both operators so that any frequency-computation
        differences between backends don't confound the operator comparison.

        Note: the tolerance here is looser than the single-pass X tests. MKL and
        cuSparse use internal parallelism in their SpMV kernels, which can reorder
        floating-point additions non-deterministically. StdXTX chains two SpMV calls
        (X then X^T), compounding the error. The standardization (dividing by σ and
        subtracting the mean) further amplifies small differences, pushing errors to
        ~1e-8 — above the default 1e-10 tolerance used for single-pass operators.
        Setting MKL threads to 1 will give deterministic results, while still differing
        from the pygrgl implementation.
        """
        # XTX_TOLERANCE accounts for error amplification from two chained SpMV calls.
        XTX_TOLERANCE = 1e-6

        K = 10
        num_mutations = sum(g.num_mutations for g in self.ground_truth_grgs)

        # Use truth_freqs for both so we're testing the operator, not frequency computation.
        truth_freqs = [allele_frequencies(g) for g in self.ground_truth_grgs]

        truth_op = MultiSciPyStdXTXOperator(self.ground_truth_grgs, truth_freqs, haploid=False)
        backend_op = MultiSciPyStdXTXOperator(self.backend_grgs, truth_freqs, haploid=False, mode="sequential")
        self.assertEqual(truth_op.shape, backend_op.shape)

        random_input = numpy.random.standard_normal((K, num_mutations)).T

        # _matmat
        numpy.testing.assert_allclose(
            backend_op._matmat(random_input),
            truth_op._matmat(random_input),
            atol=XTX_TOLERANCE,
        )

        # _matvec
        numpy.testing.assert_allclose(
            backend_op._matvec(random_input[:, 0]),
            truth_op._matvec(random_input[:, 0]),
            atol=XTX_TOLERANCE,
        )

        # _rmatmat (StdXTX is symmetric, but still exercise the path)
        numpy.testing.assert_allclose(
            backend_op._rmatmat(random_input),
            truth_op._rmatmat(random_input),
            atol=XTX_TOLERANCE,
        )


@unittest.skipUnless(MULTI_INPUT_DIR and HAS_MKL, "GRAPP_MULTI_TEST_INPUT_DIR not set or MKL not available")
class TestMultiOps_SPMV_MKL(_MultiLinearOperatorsTestBase, unittest.TestCase):
    from grapp.backends.spmv import SPMV_GRG_MKL
    BACKEND_CLASS = SPMV_GRG_MKL
    BACKEND_SUFFIX = ".grg"
    BACKEND_KWARGS = {"nthreads": 32}


@unittest.skipUnless(MULTI_INPUT_DIR and HAS_CUSPARSE, "GRAPP_MULTI_TEST_INPUT_DIR not set or cuSPARSE not available")
class TestMultiOps_SPMV_cuSparse(_MultiLinearOperatorsTestBase, unittest.TestCase):
    from grapp.backends.spmv import SPMV_GRG_cuSparse
    BACKEND_CLASS = SPMV_GRG_cuSparse
    BACKEND_SUFFIX = ".grg"
    BACKEND_KWARGS = {}
