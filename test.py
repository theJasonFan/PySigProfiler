import numpy as np
from sigprofiler import SigProfiler
import unittest

class TestSigProfiler(unittest.TestCase):
    '''
    Class for simple unit tests
    '''
    def setUp(self):
        self.N, self.M = 10, 7
        self.X = np.random.randint(1, 10, size=(self.N, self.M))
        self.X_std = np.random.standard_normal(size=(self.N, self.M))
        self.bootstrap_test_n = 1000
        self.tol = 1e-9

    def test_nmf(self):
        np.random.seed(828828)
        sp = SigProfiler()
        K = 7
        P, E, err= sp.nmf_fit(self.X, K)
        self.assertEqual(P.shape, (self.M, K))
        self.assertEqual(E.shape, (self.N, K))

        # Check that P is indeed a p-dist
        self.assertTrue(np.allclose(np.sum(P, axis=1), 
                       np.ones_like(np.sum(P, axis=1))))
        #self.assertLess(err, 1e-4)

    def test_bootstrap(self):
        sp = SigProfiler()
        X_ = sp.bootstrap(self.X)

        # Test that rowsums are preserved
        self.assertTrue(np.all(np.sum(self.X, axis=1) == np.sum(X_, axis=1)))

        # Test that bootstrapping works in expectation
        Xs = [sp.bootstrap(self.X) for _ in range(self.bootstrap_test_n)]
        X_expected = np.mean(Xs, axis=0)
        self.assertLess(np.sum(X_expected - self.X), self.tol)

class TestSigProfiler_integration(unittest.TestCase):
    '''
    Class for integration tests
    Note: it's hard to test for correctness of boostrapped nmf procedure
    '''
    def setUp(self):
        self.N, self.M = 10, 7
        self.X = np.random.randint(1, 10, size=(self.N, self.M))
        self.X_std = np.random.standard_normal(size=(self.N, self.M))
        self.bootstrap_test_n = 10
        self.tol = 1e-9

    def test_compute_bootstrapped_nmf_runs(self):
        sp = SigProfiler(bootstrap_n=self.bootstrap_test_n)
        sp.compute_bootstrapped_nmf_runs(self.X, 2)
        self.assertEqual(len(sp.Ps[2]), self.bootstrap_test_n)
        self.assertEqual(len(sp.Es[2]), self.bootstrap_test_n)
        self.assertEqual(len(sp.errs[2]), self.bootstrap_test_n)

    def test_compute_consensus_signatures(self):
        sp = SigProfiler(bootstrap_n=self.bootstrap_test_n, verbose=2)
        sp.compute_bootstrapped_nmf_runs(self.X, 2)
        sp.compute_consensus_signatures(K=2)

    def test_fit(self):
        sp = SigProfiler(rank_range=[2,3],
                         bootstrap_n=self.bootstrap_test_n, 
                         verbose=2)
        P = sp.fit(self.X)
        # Check that P is indeed a p-dist
        self.assertTrue(np.allclose(np.sum(P, axis=0), 
                       np.ones_like(np.sum(P, axis=0))))

if __name__ == '__main__':
    unittest.main()