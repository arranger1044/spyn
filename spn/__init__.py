import sys

# marginalize indicator
MARG_IND = -1

# log of zero const, to avoid -inf
# numpy.exp(LOG_ZERO) = 0
LOG_ZERO = -1e3


def IS_LOG_ZERO(log_val):
    """
    checks for a value to represent the logarithm of 0.
    The identity to be verified is that:
    IS_LOG_ZERO(x) && exp(x) == 0
    according to the constant LOG_ZERO
    """
    return (log_val <= LOG_ZERO)


# defining a numerical correction for 0
EPSILON = sys.float_info.min

# size for integers
INT_TYPE = 'int8'

# seed for random generators
RND_SEED = 31

# negative infinity for worst log-likelihood
NEG_INF = -sys.float_info.max

# abstract class definition
from abc import ABCMeta
from abc import abstractmethod


class AbstractSpn(metaclass=ABCMeta):

    """
    WRITEME
    """
    # __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, input_layer=None, layers=[]):
        """
        WRITEME
        """

    @abstractmethod
    def eval(self, input):
        """
        WRITEME
        """
    #
    # layer setting routines
    #
    @abstractmethod
    def set_input_layer(self, layer):
        """
        WRITEME
        """

    @abstractmethod
    def set_layers(self, layers):
        """
        WRITEME
        """

    @abstractmethod
    def add_layer(self, layer, pos=None):
        """
        WRITEME
        """

    @abstractmethod
    def fit(self, train, valid, test, algo, options):
        """
        WRITEME
        """

    def __repr__(self):
        """
        Printing an SPN summary
        WRITEME
        """
        layer_strings = [msg for msg in map(str, self._layers)]
        layer_strings.reverse()
        layer_strings.append(str(self._input_layer))
        stats = '\n'.join(layer_strings)
        return stats

    def top_down_layers(self):
        """
        WRITEME
        """
        for layer in reversed(self._layers):
            yield layer

    def input_layer(self):
        """
        WRITEME
        """
        return self._input_layer

    def smooth_leaves(self, alpha):
        """
        Laplacian smoothing of the probability values
        of the leaf nodes (if the leaf represents a univariate distribution)
        """
        self._input_layer.smooth_probs(alpha)

    def n_layers(self):
        """
        WRITEME
        """
        return len(self._layers) + 1

    def n_nodes(self):
        """
        WRITEME
        """
        nodes = self._input_layer.n_nodes()
        for layer in self._layers:
            nodes += layer.n_nodes()
        return nodes

    def n_edges(self):
        """
        WRITEME
        """
        #
        # adding input layer too, it may contain cltrees
        edges = self._input_layer.n_edges()
        for layer in self._layers:
            edges += layer.n_edges()

        return edges

    def n_leaves(self):
        """
        WRITEME
        """
        return self._input_layer.n_nodes()

    def n_weights(self):
        """
        WRITEME
        """
        weights = 0
        for layer in self._layers:
            weights += layer.n_weights()
        return weights

    def stats(self):
        """
        WRITEME
        """
        # total stats
        stats = '*************************\n'\
            '* levels:\t{0}\t*\n'\
            '* nodes:\t{1}\t*\n'\
            '* edges:\t{2}\t*\n'\
            '* weights:\t{3}\t*\n'\
            '*************************'.format(self.n_layers(),
                                               self.n_nodes(),
                                               self.n_edges(),
                                               self.n_weights())
        return stats
