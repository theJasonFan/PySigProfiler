from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import silhouette_score
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from collections import defaultdict
import numpy as np
import warnings

class SigProfiler(object):
    '''
    Python implementation of SigProfiler detailed in:
    Alexandrev et. 2013, Deciphering Signatures of Mutational Processes 
        Operative in Human Cancer

        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3588146
    
    NOTE: The algorithm used in the clustering step is *not* the same as
    the one used by the original MatLab SigProfiler. The original performs 
    clustering under the cosine metric. This performs clustering under the
    euclidean metric. However, the minimized objective under both objectives
    are just (monotonic) linear functions of each other. This implementation
    should be 'close enough'. 
    '''
    def __init__(self,
                 rank_range=[2],
                 bootstrap_n=100,
                 nmf_max_iter=1000,
                 nmf_tol=1e-9,
                 nmf_beta_loss='frobenius',
                 nmf_solver='mu',
                 nmf_random_state=None,
                 km_max_iter=1000,
                 km_tol = 1e-9,
                 verbose=0
                 ):
        '''
        Parameters
        ----------
        rank_range : list of int
            The possible ranks (# of mutation signatures) to search for
        
        bootstrap_n : int
            The number of bootstrapped samples to generate mutational 
            signatures from
        
        nmf_max_iter : int
            The number of iterations any run of the NMF algorithm is allowed
            to run for. (Note: a warning will be raised if NMF has not 
            converged)

        nmf_beta_loss : float or string, default 'frobenius'
            String must be in {'frobenius', 'kullback-leibler', 'itakura-saito'}.
            Beta divergence to be minimized, measuring the distance between X
            and the dot product WH. Note that values different from 'frobenius'
            (or 2) and 'kullback-leibler' (or 1) lead to significantly slower
            fits. Note that for beta_loss <= 0 (or 'itakura-saito'), the input
            matrix X cannot contain zeros. Used only in 'mu' solver.

        nmf_tol : float
            Tolerence for the stopping criterion for NMF runs
        
        nmf_solver : 'mu' | 'cd'
            Method used to solve NMF
            - 'mu': multiplicative update
            - 'cd': co-ordinate descent

        km_max_iter : int
            The number of iterations any run of the KMeans algorithm used for 
            finding consensus signatures is allowed to run for.

        nmf_tol : float
            Tolerence for the stopping criterion for KMeans

        verbose:
            Verbosity of output. 0 for no outputs.

        Attributes
        ----------
        P_centroids: dict int: array, [n_components, n_features]
            Dictionary of consensus signatures for each given rank

        E_centroids: dict int: array, [n_samples, n_components]
            Dictionary of consensus exposures for each given rank

        silhouette_scores: dict int: float
            Silhouette widths of clustering of signatures for each given rank

        reconstruction_errs: dict int: float
            Frobenius reconstructions for with respect to consensus signatures
            for each given rank

        __X: array, [n_samples, n_features]
            Data array used to fit signatures

        P: array, [n_components, n_features]
            The 'best' consensus signature
        E: array, [n_samples, n_components]
            The exposures correspoding to best signatures
        '''

        self.rank_range = rank_range
        self.bootstrap_n = bootstrap_n
        self.nmf_max_iter = nmf_max_iter
        self.nmf_beta_loss = nmf_beta_loss
        self.nmf_solver = nmf_solver
        self.nmf_tol = nmf_tol
        self.nmf_random_state = None
        self.km_max_iter = km_max_iter
        self.km_tol = km_tol
        self.verbose = verbose

        # Book-keeping
        self.Ps = defaultdict(list)
        self.Es = defaultdict(list)
        self.errs = defaultdict(list)
        self.P_centroids = {}
        self.E_centroids = {}
        self.silhouette_scores = {}

        # Fitted params
        self.__X = None
        self.P = None
        self.E = None
        self.K = None

    def fit(self, X):
        '''
        Fit model and return 'best' signatures
        '''
        self.__X = X
        # 1) iterate over rank range
        for k in self.rank_range:

            # 2) generate bootstrapped samples
            self.compute_bootstrapped_nmf_runs(self.__X, k)

            # 3) cluster anc compute consensus signatures
            self.compute_consensus_signatures(k)

        # 4) Select K by using most 'reproducible' rank
        self.K = max(self.silhouette_scores, key=self.silhouette_scores.get)
        if self.verbose >= 1:
            print('* best consensus rank:', self.K)
        self.P = self.P_centroids[self.K]
        return self.P
        
    def compute_consensus_signatures(self, K):
        '''
        Compute and save consensus signatures for rank K
        '''
        Ps = self.Ps[K]
        km = KMeans(n_clusters=K, max_iter=self.km_max_iter, tol=self.km_tol)
        Ps = np.vstack(Ps)
        print(Ps.shape)
        km.fit(Ps)
        print(km.n_iter_)
        P_centroid = km.cluster_centers_
        labels = km.labels_
        print(labels)

        cosine_dists = cosine_distances(Ps)
        self.silhouette_scores[K] = silhouette_score(cosine_dists, labels, 
                                                     metric='precomputed')
        self.P_centroids[K] = P_centroid / np.sum(P_centroid, axis=0)
    
    def compute_bootstrapped_nmf_runs(self, X, K):
        '''
        Compute and save NMF run wit rank K with bootstrapped data matrix
        '''
        if self.verbose >= 1:
            print('* computing nmf of %d bootstrapped samples  with K=%d' \
                  % (self.bootstrap_n, K))
        for i in range(self.bootstrap_n):
            if self.verbose >= 2:
                print('\t - computing nmf for bootstrapped sample %d with K=%d' % (i, K))
            X_ = self.bootstrap(X)
            P_i, E_i, err_i = self.nmf_fit(X_, K)
            self.Ps[K].append(P_i)
            self.Es[K].append(E_i)
            self.errs[K].append(err_i)

    def bootstrap(self, X):
        '''
        Return bootstrapped data matrix.
        
        For a given sample with N mutations, N mutations are drawn with
        probability proportional to the number of observed mutations in each category.
        '''
        assert(X.dtype == int)
        # X has shape (N, M)

        row_sums = np.sum(X, axis=1, keepdims=True)

        ps = X / row_sums
        X_ = [np.random.multinomial(n, ps[i]) 
              for i, n in enumerate(row_sums)]
        return np.asarray(X_, dtype=int)

    def nmf_fit(self, X, K):
        '''
        Compute NMF with rank K on given matrix X
        '''
        nmf = NMF(n_components=K,
                   solver=self.nmf_solver,
                   beta_loss=self.nmf_beta_loss,
                   tol=self.nmf_tol,
                   max_iter=self.nmf_max_iter,
                   verbose=max(self.verbose - 2, 0))
        E = nmf.fit_transform(X)
        P = nmf.components_
        norm = np.sum(P, axis=1, keepdims=True)
        E *= norm.reshape(-1)
        P /= norm
        err = nmf.reconstruction_err_
        if nmf.n_iter_ >= (self.nmf_max_iter - 1):
            warnings.warn('NMF ran for max iterations')
        return P, E, err