import numpy

try:
    from time import perf_counter
except:
    from time import time
    perf_counter = time

from spn import LOG_ZERO


class DataSlice(object):

    """
    A little util class for storing
    the sets of indexes for the instances and features
    considered
    """

    class_counter = 0

    @classmethod
    def reset_id_counter(cls):
        """
        WRITEME
        """
        DataSlice.class_counter = 0

    @classmethod
    def whole_slice(cls,
                    n_instances,
                    n_features):

        instances = numpy.arange(n_instances, dtype=numpy.uint32)
        features = numpy.arange(n_features, dtype=numpy.uint32)
        return DataSlice(instances, features)

    def __init__(self,
                 instances=None,
                 features=None):
        #
        # ensuring them to be numpy ndarrays
        self.instance_ids = numpy.array(instances)
        self.feature_ids = numpy.array(features)

        self.id = DataSlice.class_counter
        self.children = []
        self.weights = []
        self.ll = LOG_ZERO
        #
        # this is fugly, what do I need the oop for?
        self.type = None
        #
        # adding some members for the ll correction
        self.lls = None  # this shall be a numpy array
        self.parent = None
        self.w = 1.0
        DataSlice.class_counter += 1

    def __hash__(self):
        return hash(self.id)

    # def __eq__(self, other):
    #     return self.id == other.id

    # def __ne__(self, other):
    #     return not self.__eq__(other)

    def add_child(self, data_slice_child, data_slice_weight=None):
        self.children.append(data_slice_child)
        if data_slice_weight is not None:
            self.weights.append(data_slice_weight)
            # adding a reference to the weight
            data_slice_child.w = data_slice_weight
        # adding a reference to a child
        data_slice_child.parent = self

    #
    # commenting this out for error prevention
    # def add_children(self, data_slice_children, data_slice_weights=None):
    #     self.children.extend(data_slice_children)
    #     if data_slice_weights is not None:
    #         self.weigths.extend(data_slice_weights)

    def n_instances(self):
        return len(self.instance_ids)

    def n_features(self):
        return len(self.feature_ids)

    def __repr__(self):
        return ("[id: {id} ll: {ll} i :{instances} f :{features}\n {lls}]".
                format(id=self.id,
                       ll=self.ll,
                       instances=self.instance_ids,
                       features=self.feature_ids,
                       lls=self.lls))
