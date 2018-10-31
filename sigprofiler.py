from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import silhouette_score
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from collections import defaultdict
import numpy as np
import warnings

class SigProfiler(object):
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
        assert(X.dtype == int)
        # X has shape (N, M)

        row_sums = np.sum(X, axis=1, keepdims=True)

        ps = X / row_sums
        X_ = [np.random.multinomial(n, ps[i]) 
              for i, n in enumerate(row_sums)]
        return np.asarray(X_, dtype=int)

    def nmf_fit(self, X, K):
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