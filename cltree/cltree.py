import numpy

import numba

import scipy.sparse
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import depth_first_order

from spn import LOG_ZERO


@numba.njit
def safe_log(x):
    """
    Assuming x to be a scalar
    """
    if x > 0.0:
        return numpy.log(x)
    else:
        return LOG_ZERO


@numba.njit
def compute_mutual_information(feature_vals,
                               log_probs,
                               log_joint_probs,
                               m_i_table):
    """
    WRITEME
    """
    n_features = feature_vals.shape[0]
    for i in range(n_features):
        for j in range(i + 1, n_features):
            # if i != j:
            for val_i in range(feature_vals[i]):
                for val_j in range(feature_vals[j]):
                    log_joint_i_j = log_joint_probs[i, j, val_i, val_j]
                    m_i_table[i, j] = (m_i_table[i, j] +
                                       numpy.exp(log_joint_i_j) *
                                       (log_joint_i_j -
                                        log_probs[i, val_i] -
                                        log_probs[j, val_j]))
                    m_i_table[j, i] = m_i_table[i, j]
    return m_i_table


@numba.njit
def compute_log_probs(freqs,
                      probs,
                      log_probs,
                      n_instances,
                      alpha=0.0):
    """
    WRITEME
    """
    n_features = freqs.shape[0]

    #
    # smoothing, if needed
    # probs = (freqs + 2 * alpha) / (n_instances + 4 * alpha)

    for i in range(n_features):
        #
        # smoothing if needed
        probs[i] = (freqs[i] + 2 * alpha) / (n_instances + 4 * alpha)
        #
        # going to logs
        log_probs[i, 0] = safe_log(1 - probs[i])
        log_probs[i, 1] = safe_log(probs[i])
    return log_probs


@numba.njit
def compute_joint_bin_freqs(joint_freqs,
                            co_occs,
                            n_instances):
    """
    Assuming features to be binary
    TODO: generalize this to categorical vars
    """
    n_features = co_occs.shape[0]

    for i in range(n_features):
        for j in range(i + 1, n_features):
            joint_freqs[i, j, 1, 1] = co_occs[i, j]
            joint_freqs[i, j, 0, 1] = co_occs[j, j] - co_occs[i, j]
            joint_freqs[i, j, 1, 0] = co_occs[i, i] - co_occs[i, j]
            joint_freqs[i, j, 0, 0] = (n_instances -
                                       joint_freqs[i, j, 1, 1] -
                                       joint_freqs[i, j, 0, 1] -
                                       joint_freqs[i, j, 1, 0])

            # saving for symmetry
            joint_freqs[j, i, 1, 1] = joint_freqs[i, j, 1, 1]
            joint_freqs[j, i, 0, 1] = joint_freqs[i, j, 1, 0]
            joint_freqs[j, i, 1, 0] = joint_freqs[i, j, 0, 1]
            joint_freqs[j, i, 0, 0] = joint_freqs[i, j, 0, 0]

    return joint_freqs


@numba.njit
def compute_log_joint_bin_probs(joint_freqs,
                                log_joint_probs,
                                n_instances,
                                alpha=0.0):
    """
    Assuming features to be binary
    TODO: generalize this to categorical vars
    """
    n_features = joint_freqs.shape[0]

    for i in range(n_features):
        for j in range(i + 1, n_features):

            log_joint_probs[i, j, 1, 1] = \
                safe_log((joint_freqs[i, j, 1, 1] + alpha) /
                         (n_instances + 4.0 * alpha))
            log_joint_probs[i, j, 0, 1] = \
                safe_log((joint_freqs[i, j, 0, 1] + alpha) /
                         (n_instances + 4.0 * alpha))
            log_joint_probs[i, j, 1, 0] = \
                safe_log((joint_freqs[i, j, 1, 0] + alpha) /
                         (n_instances + 4.0 * alpha))
            log_joint_probs[i, j, 0, 0] = \
                safe_log((joint_freqs[i, j, 0, 0] + alpha) /
                         (n_instances + 4.0 * alpha))

            # saving for symmetry
            log_joint_probs[j, i, 1, 1] = log_joint_probs[i, j, 1, 1]
            log_joint_probs[j, i, 0, 1] = log_joint_probs[i, j, 1, 0]
            log_joint_probs[j, i, 1, 0] = log_joint_probs[i, j, 0, 1]
            log_joint_probs[j, i, 0, 0] = log_joint_probs[i, j, 0, 0]

    return log_joint_probs


@numba.njit
def compute_log_cond_probs(feature_vals,
                           log_probs,
                           log_joint_probs,
                           log_cond_probs):
    """
    FIXME:
    when both log_probs are LOG_ZERO then result is 0->prob of 1
    plus, the safe_log function shall be used once only, at the end
    of all computations
    """
    n_features = feature_vals.shape[0]
    for i in range(n_features):
        for j in range(n_features):
            if i != j:
                for val_i in range(feature_vals[i]):
                    for val_j in range(feature_vals[j]):
                        log_cond_probs[i, j, val_i, val_j] = \
                            (log_joint_probs[i, j, val_i, val_j] -
                             log_probs[j, val_j])

    return log_cond_probs


@numba.njit
def compute_log_factors(tree,
                        feature_vals,
                        log_probs,
                        log_joint_probs,
                        log_factors):
    """
    FIXME:
    when both log_probs are LOG_ZERO then result is 0->prob of 1
    plus, the safe_log function shall be used once only, at the end
    of all computations

    This shall compute only the conditioned factors
    """
    n_features = feature_vals.shape[0]

    #
    # for the root we have a redundant representation
    log_factors[0, 0, 0] = log_probs[0, 0]
    log_factors[0, 0, 1] = log_probs[0, 0]
    log_factors[0, 1, 0] = log_probs[0, 1]
    log_factors[0, 1, 1] = log_probs[0, 1]
    #
    # for the rest
    for feature_id, parent_id in zip(range(1, n_features), tree[1:]):
        for feature_val in range(feature_vals[feature_id]):
            for parent_val in range(feature_vals[parent_id]):
                log_factors[feature_id, feature_val, parent_val] = \
                    (log_joint_probs[feature_id, parent_id, feature_val, parent_val] -
                     log_probs[parent_id, parent_val])

    return log_factors

from spn import MARG_IND


def tree_2_factor_matrix(tree,
                         factors):
    """
    factors = n_features x n_factors
    (n_features x n_features for trees)
    """

    n_features = factors.shape[0]
    print(factors.shape)
    #
    # setting the root node
    factors[0, 0] = True

    for i, feature in zip(range(1, n_features), tree[1:]):
        factors[i, i] = True
        factors[feature, i] = True

    return factors


def instantiate_factors(tree,
                        feature_vals,
                        evidence,
                        factors,
                        ev_factors):
    """
    ev_factors = factors
    """

    n_features = tree.shape[0]
    #
    # evidence for root
    if evidence[0] != MARG_IND:
        ev_factors[0] = factors[0][evidence[0]]

    for i, parent in zip(range(1, n_features), tree[1:]):
        if evidence[i] != MARG_IND:
            child_evidence = [evidence[i]]
        else:
            child_evidence = list(range(feature_vals[i]))

        if evidence[parent] != MARG_IND:
            parent_evidence = [evidence[parent]]
        else:
            parent_evidence = list(range(feature_vals[parent]))

        ev_factors[i] = factors[i][child_evidence, parent_evidence]
    return ev_factors


def marginalize(features,
                feature_vals,
                factors,
                factor_matrix,
                tree,
                evidence):
    """
    ex: evidence = [0, 1, MARG_IND, 0, MARG_IND]
    ordering = [False, True, False] the second elem is a leaf
    """
    n_features = len(tree)
    #
    # getting vars to marginalize over
    # marg_vars = ()
    print(evidence == MARG_IND)
    print(features[evidence == MARG_IND])
    sum_out_features = features[evidence == MARG_IND]
    print(sum_out_features)

    log_prob = 0.0
    remaining_factors = numpy.ones(n_features, dtype=bool)
    #
    # for each feature to marginalize
    for m_feature in sum_out_features:
        print('Consider feature', m_feature)
        #
        #
        sum_prob = 0.0
        for val in range(feature_vals[m_feature]):
            print('Sum over', m_feature, val)
            #
            # getting the factors it appears into
            log_prod_prob = log_prob
            factor_list = factor_matrix[m_feature]
            for f in features[factor_list]:
                factor = factors[f]
                print('Considering Factor', f)
                if factor.shape[0] > 1:
                    log_prod_prob += factor[val]
                else:
                    log_prod_prob += factor[0]
                print('log_prod_prob', log_prod_prob)
                #
                # erasing the factor from consideration
                remaining_factors[f] = False
            sum_prob += numpy.exp(log_prod_prob)
            print('sum prob', sum_prob)

        factor_matrix[:, ~remaining_factors] = False
        #
        #
        log_prob = numpy.log(sum_prob)
        print('log prob', log_prob)
    #
    # for the remaining factors (they shall be instantiated)
    print('Remaining Factors', remaining_factors)
    for f in features[remaining_factors]:
        factor = factors[f]
        log_prob += factor[0]
    return log_prob


# def minimum_spanning_tree(X, copy=True):
#     """
#     X are edge weights of fully connected graph
#     """

#     if copy:
#         X = X.copy()

#     if X.shape[0] != X.shape[1]:
#         raise ValueError("X needs to be square matrix of edge weights")

#     n_vertices = X.shape[0]
#     spanning_edges = []

#     # initialize with node 0:
#     visited_vertices = [0]
#     num_visited = 1

#     # exclude self connections:
#     diag_indices = numpy.arange(n_vertices)
#     X[diag_indices, diag_indices] = numpy.inf

#     while num_visited != n_vertices:
#         new_edge = numpy.argmin(X[visited_vertices], axis=None)
#         # 2d encoding of new_edge from flat, get correct indices
#         new_edge = divmod(new_edge, n_vertices)
#         # print('new_edge', new_edge)
#         # print(visited_vertices[new_edge[0]])
#         new_edge = [visited_vertices[new_edge[0]], new_edge[1]]
#         # add edge to tree
#         spanning_edges.append(new_edge)
#         visited_vertices.append(new_edge[1])
#         # remove all edges inside current tree
#         X[visited_vertices, new_edge[1]] = numpy.inf
#         X[new_edge[1], visited_vertices] = numpy.inf
#         num_visited += 1
#    return numpy.vstack(spanning_edges)


# @numba.jit
# def minimum_spanning_tree_numba(X,
#                                 visited_vertices,
#                                 spanning_edges,
#                                 diag_indices):
#     """
#     X are edge weights of fully connected graph
#     """

#     # if X.shape[0] != X.shape[1]:
#     #     raise ValueError("X needs to be square matrix of edge weights")

#     n_vertices = X.shape[0]
#     # spanning_edges = []

#     # initialize with node 0:
#     visited_vertices[0] = True
#     num_visited = 1

#     # exclude self connections:
#     # diag_indices = numpy.arange(n_vertices)
#     # X[diag_indices, diag_indices] = numpy.inf
#     for i in range(n_vertices):
#         X[i, i] = numpy.inf

#     while num_visited != n_vertices:
#         new_edge = numpy.argmin(X[visited_vertices])
#         # 2d encoding of new_edge from flat, get correct indices
#         # new_edge = divmod(new_edge, n_vertices)

#         new_edge_0 = new_edge // n_vertices
#         new_edge_1 = new_edge % n_vertices

#         # print('new_edge', (new_edge_0, new_edge_1))
#         # print(visited_vertices.nonzero()[0][new_edge_0])
#         new_edge_0 = visited_vertices.nonzero()[0][new_edge_0]
#         # add edge to tree
#         # spanning_edges.append(new_edge)
#         spanning_edges[num_visited - 1, 0] = new_edge_0
#         spanning_edges[num_visited - 1, 1] = new_edge_1

#         # visited_vertices.append(new_edge[1])
#         visited_vertices[new_edge_1] = True
#         # remove all edges inside current tree
#         X[visited_vertices, new_edge_1] = numpy.inf
#         X[new_edge_1, visited_vertices] = numpy.inf
#         num_visited += 1
#     return numpy.vstack(spanning_edges)


@numba.njit
def eval_instance(instance,
                  features,
                  tree,
                  log_marg_probs,
                  log_cond_probs):
    """
    WRITEME
    """
    ll = log_marg_probs[0, instance[features[0]]]
    for feature_id in range(1, tree.shape[0]):
        feature = features[feature_id]
        parent_id = tree[feature_id]
        ll += log_cond_probs[feature_id, parent_id,
                             instance[feature],
                             instance[features[parent_id]]]
    return ll


@numba.njit
def eval_instance_fact(instance,
                       features,
                       tree,
                       factors):
    """
    WRITEME
    """
    ll = 0.0
    for feature_id in range(tree.shape[0]):
        feature = features[feature_id]
        parent_id = tree[feature_id]
        ll += factors[feature_id,
                      instance[feature],
                      instance[features[parent_id]]]

    return ll


class CLTree:

    """
    A class for modeling a Chow-Liu tree
    """

    def __init__(self,
                 data,
                 features=None,
                 factors=None,
                 tree=None,
                 n_feature_vals=2,
                 feature_vals=None,
                 alpha=0.1,
                 sparse=True,
                 mem_free=True):
        """
        WRITEME
        """

        #
        # learning it from data
        if data is not None:
            self._learn_from_data(data,
                                  features,
                                  n_feature_vals,
                                  feature_vals,
                                  alpha,
                                  sparse,
                                  mem_free)
        elif (features is not None and
              feature_vals is not None and
              tree is not None and
              factors is not None):
            self._build_from_factors(features,
                                     feature_vals,
                                     tree,
                                     factors)
        else:
            raise ValueError('Invalid CL Tree initialization')

    def _learn_from_data(self,
                         data,
                         features=None,
                         n_feature_vals=2,
                         feature_vals=None,
                         alpha=0.1,
                         sparse=True,
                         mem_free=True):
        """
        Chow and Liu learning algorithm
        """
        #
        # this trick helps for sparse matrices
        # TODO: check if this cond is needed or the sparse dot is equal to
        # the dense one performance-wise
        if sparse:
            self._data = scipy.sparse.csr_matrix(data)
        else:
            self._data = data

        self._alpha = alpha
        self._n_features = data.shape[1]
        self._n_instances = data.shape[0]

        self.features = features

        #
        # assuming homogeneous features this could be restrictive
        # TODO: extend the whole code to categorical non homogeneous features
        self._feature_vals = feature_vals

        if self._feature_vals is None:
            self._feature_vals = \
                numpy.array([n_feature_vals
                             for i in range(self._n_features)])

        #
        # getting the max to pre-allocate the memory
        self._n_feature_vals = n_feature_vals
        if self._n_feature_vals is None:
            self._n_feature_vals = max(self._feature_vals)

        if self.features is None:
            self.features = numpy.array([i for i in range(self._n_features)])

        #
        # pre-allocating arrays for freqs and probs
        # self._marg_freqs = numpy.zeros(self._n_features)
        self._joint_freqs = numpy.zeros((self._n_features,
                                         self._n_features,
                                         self._n_feature_vals,
                                         self._n_feature_vals))
        self._log_marg_probs = numpy.zeros((self._n_features,
                                            self._n_feature_vals))
        self._log_joint_probs = numpy.zeros((self._n_features,
                                             self._n_features,
                                             self._n_feature_vals,
                                             self._n_feature_vals))
        self._log_cond_probs = numpy.zeros((self._n_features,
                                            self._n_features,
                                            self._n_feature_vals,
                                            self._n_feature_vals))
        self._mutual_info = numpy.zeros((self._n_features,
                                         self._n_features))

        #
        # computing freqs and probs (and smoothing)
        co_occ_matrix = self._data.T.dot(self._data)
        #
        # marginal frequencies
        if sparse:
            co_occ_matrix = numpy.array(co_occ_matrix.todense())
            self._marg_freqs = co_occ_matrix.diagonal()
        else:
            self._marg_freqs = co_occ_matrix.diagonal()

        self._log_marg_probs = self.log_marg_probs(self._marg_freqs,
                                                   self._log_marg_probs)
        #
        # joint estimation
        self._joint_freqs = self.joint_freqs(self._joint_freqs,
                                             co_occ_matrix)
        self._log_joint_probs = self.log_joint_probs(self._joint_freqs,
                                                     self._log_joint_probs)
        #
        # conditional estimation
        self._log_cond_probs = self.log_cond_probs(self._log_marg_probs,
                                                   self._log_joint_probs,
                                                   self._log_cond_probs)
        self._mutual_info = self.mutual_information(self._log_marg_probs,
                                                    self._log_joint_probs,
                                                    self._mutual_info)

        #
        # computing the MST (this way we are not overwriting mutual_info)
        # this can be useful for testing but not for efficiency
        # mst = minimum_spanning_tree(-self._mutual_info, copy=copy_mi)
        mst = minimum_spanning_tree(-(self._mutual_info + 1))
        dfs_tree = depth_first_order(mst, directed=False, i_start=0)

        #
        # representing the CLTree as a sequence of parents ids
        self._tree = numpy.zeros(self._n_features, dtype=int)

        # self._tree[0] = -1
        # the root is its parent
        self._tree[0] = 0

        for feature in range(1, self._n_features):
            self._tree[feature] = dfs_tree[1][feature]

        #
        # computing the factored represetation
        self._factors = numpy.zeros((self._n_features,
                                     self._n_feature_vals,
                                     self._n_feature_vals))
        self._factors = self.log_factors(self._log_marg_probs,
                                         self._log_joint_probs,
                                         self._factors)

        #
        # removing references,this is optional for test purposes
        if mem_free:
            self._mutual_info = None
            self._joint_freqs = None
            self._log_marg_probs = None
            self._log_joint_probs = None
            self._log_cond_probs = None
            self._marg_freqs = None
            self._data = None

    def _build_from_factors(self,
                            features,
                            feature_vals,
                            tree,
                            factors):
        """
        Building a node from (already learned) info
        about the features, the tree representation and
        the factors values
        """
        #
        # just storing information
        self._features = features
        self._n_features = len(features)
        self._feature_vals = feature_vals

        self._tree = tree
        self._factors = factors

    #
    #
    # numba ops wrappers

    def joint_freqs(self, joint_freqs, co_occs):
        """
        WRITEME
        """
        return compute_joint_bin_freqs(joint_freqs,
                                       co_occs,
                                       self._n_instances)

    def log_marg_probs(self, freqs, log_probs):
        """
        WRITEME
        """
        probs = numpy.zeros(freqs.shape[0])
        return compute_log_probs(freqs,
                                 probs,
                                 log_probs,
                                 self._n_instances,
                                 self._alpha)

    def log_joint_probs(self, joint_freqs, log_joint_probs):
        """
        WRITEME
        """
        return compute_log_joint_bin_probs(joint_freqs,
                                           log_joint_probs,
                                           self._n_instances,
                                           self._alpha)

    def log_cond_probs(self,
                       log_marg_probs,
                       log_joint_probs,
                       log_cond_probs):
        """
        WRITEME
        """
        return compute_log_cond_probs(self._feature_vals,
                                      log_marg_probs,
                                      log_joint_probs,
                                      log_cond_probs)

    def log_factors(self,
                    log_probs,
                    log_joint_probs,
                    log_factors):
        """
        WRITEME
        """
        return compute_log_factors(self._tree,
                                   self._feature_vals,
                                   log_probs,
                                   log_joint_probs,
                                   log_factors)

    def mutual_information(self,
                           log_marg_probs,
                           log_joint_probs,
                           mutual_info):
        """
        WRITEME
        """
        return compute_mutual_information(self._feature_vals,
                                          log_marg_probs,
                                          log_joint_probs,
                                          mutual_info)

    # def smooth(self, alpha):
    #     """
    #     Recomputing logs with different smooths
    #     This may be totally useless
    #     """
    #     #
    #     # saving alpha since wrapper are using the local value
    #     self._alpha = alpha
    #     #
    #     # recompute and smooth (is this too side-effect prone?)
    #     self._log_marg_probs = self.log_marg_probs(self._marg_freqs,
    #                                                self._log_marg_probs)

    #     self._log_joint_probs = self.log_joint_probs(self._joint_freqs,
    #                                                  self._log_joint_probs)
    #     self._log_cond_probs = self.log_cond_probs(self._log_marg_probs,
    #                                                self._log_joint_probs,
    #                                                self._log_cond_probs)

    def eval(self, instance):
        """
        WRITEME
        """
        return eval_instance(instance,
                             self.features,
                             self._tree,
                             self._log_marg_probs,
                             self._log_cond_probs)

    def eval_fact(self, instance):
        """
        WRITEME
        """
        return eval_instance_fact(instance,
                                  self.features,
                                  self._tree,
                                  self._factors)

    def __repr__(self):
        """
        WRITEME
        """
        return ('CLTree on features {0}:\n\t{1}'.format(self.features,
                                                        self._tree))

    def tree_repr(self):
        """
        Internal representation of the tree
        """
        return ("{0}".format(self._tree))

    def factors_repr(self):
        """
        String representation for internal factors
        TODO: this assumes vars are binary
        """
        return ("{0}".format(self._factors[0, :]))
