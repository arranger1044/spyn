from spn.linked.spn import Spn as SpnLinked

from spn.linked.layers import SumLayer as SumLayerLinked
from spn.linked.layers import ProductLayer as ProductLayerLinked
from spn.linked.layers import CategoricalInputLayer
from spn.linked.layers import CategoricalSmoothedLayer \
    as CategoricalSmoothedLayerLinked
from spn.linked.layers import CategoricalIndicatorLayer \
    as CategoricalIndicatorLayerLinked
from spn.linked.layers import CategoricalCLInputLayer \
    as CategoricalCLInputLayerLinked

from spn.linked.nodes import SumNode
from spn.linked.nodes import ProductNode
from spn.linked.nodes import CategoricalSmoothedNode
from spn.linked.nodes import CategoricalIndicatorNode
from spn.linked.nodes import CLTreeNode

from spn.utils import pairwise

from spn import INT_TYPE

import numpy

from math import ceil

import scipy.sparse

import sklearn.preprocessing

import random
import itertools
from collections import deque

import dataset

import logging


class SpnFactory(object):

    """
    WRITEME
    """

 
    #####################################################
    #
    #####################################################
    @classmethod
    def linked_kernel_density_estimation(cls,
                                         n_instances,
                                         features,
                                         node_dict=None,
                                         alpha=0.1
                                         # ,batch_size=1,
                                         # sparse=False
                                         ):
        """
        WRITEME
        """

        n_features = len(features)

        # the top one is a sum layer with a single node
        root_node = SumNode()
        root_layer = SumLayerLinked([root_node])

        # second one is a product layer with n_instances nodes
        product_nodes = [ProductNode() for i in range(n_instances)]
        product_layer = ProductLayerLinked(product_nodes)
        # linking them to the root node
        for prod_node in product_nodes:
            root_node.add_child(prod_node, 1. / n_instances)

        # last layer can be a categorical smoothed input
        # or sum_layer + categorical indicator input

        input_layer = None
        layers = None
        n_leaf_nodes = n_features * n_instances

        if node_dict is None:
            # creating a sum_layer with n_leaf_nodes
            sum_nodes = [SumNode() for i in range(n_leaf_nodes)]
            # store them into a layer
            sum_layer = SumLayerLinked(sum_nodes)
            # linking them to the products above
            for i, prod_node in enumerate(product_nodes):
                for j in range(n_features):
                    # getting the next n_features nodes
                    prod_node.add_child(sum_nodes[i * n_features + j])
            # now creating the indicator nodes
            input_layer = \
                CategoricalIndicatorLayerLinked(vars=features)
            # linking the sum nodes to the indicator vars
            for i, sum_node in enumerate(sum_nodes):
                # getting the feature id
                j = i % n_features
                # and thus its number of values
                n_values = features[j]
                # getting the indices of indicators
                start_index = sum(features[:j])
                end_index = start_index + n_values
                indicators = [node for node
                              in input_layer.nodes()][start_index:end_index]
                for ind_node in indicators:
                    sum_node.add_child(ind_node, 1. / n_values)

            # storing levels
            layers = [sum_layer, product_layer,
                      root_layer]
        else:
            # create a categorical smoothed layer
            input_layer = \
                CategoricalSmoothedLayerLinked(vars=features,
                                               node_dicts=node_dict,
                                               alpha=alpha)
            # it shall contain n_leaf_nodes nodes
            smooth_nodes = list(input_layer.nodes())
            assert len(smooth_nodes) == n_leaf_nodes

            # linking it
            for i, prod_node in enumerate(product_nodes):
                for j in range(n_features):
                    # getting the next n_features nodes
                    prod_node.add_child(smooth_nodes[i * n_features + j])
            # setting the used levels
            layers = [product_layer, root_layer]

        # create the spn from levels
        kern_spn = SpnLinked(input_layer, layers)
        return kern_spn


    @classmethod
    def linked_naive_factorization(cls,
                                   features,
                                   node_dict=None,
                                   alpha=0.1):
        """
        WRITEME
        """
        n_features = len(features)

        # create an input layer
        input_layer = None
        layers = None

        # first layer is a product layer with n_feature children
        root_node = ProductNode()
        root_layer = ProductLayerLinked([root_node])

        # second is a sum node on an indicator layer
        if node_dict is None:
            # creating sum nodes
            sum_nodes = [SumNode() for i in range(n_features)]
            # linking to the root
            for node in sum_nodes:
                root_node.add_child(node)
            # store into a level
            sum_layer = SumLayerLinked(sum_nodes)
            # now create an indicator layer
            input_layer = CategoricalIndicatorLayerLinked(vars=features)
            # and linking it
            # TODO make this a function
            for i, sum_node in enumerate(sum_nodes):
                # getting the feature id
                j = i % n_features
                # and thus its number of values
                n_values = features[j]
                # getting the indices of indicators
                start_index = sum(features[:j])
                end_index = start_index + n_values
                indicators = [node for node
                              in input_layer.nodes()][start_index:end_index]
                for ind_node in indicators:
                    sum_node.add_child(ind_node, 1. / n_values)

            # collecting layers
            layers = [sum_layer, root_layer]

        # or a categorical smoothed layer
        else:
            input_layer = CategoricalSmoothedLayerLinked(vars=features,
                                                         node_dicts=node_dict,
                                                         alpha=alpha)
            # it shall contain n_features nodes
            smooth_nodes = list(input_layer.nodes())
            assert len(smooth_nodes) == n_features
            for node in smooth_nodes:
                root_node.add_child(node)

            # set layers accordingly
            layers = [root_layer]

        # build the spn
        naive_fact_spn = SpnLinked(input_layer, layers)

        return naive_fact_spn


    @classmethod
    def linked_random_spn_top_down(cls,
                                   vars,
                                   n_layers,
                                   n_max_children,
                                   n_scope_children,
                                   max_scope_split,
                                   merge_prob=0.5,
                                   rand_gen=None):
        """
        WRITEME
        """

        def cluster_scopes(scope_list):
            cluster_dict = {}

            for i, var in enumerate(scope_list):
                cluster_dict[var] += {i}
            return cluster_dict

        def cluster_set_scope(scope_list):
            return {scope for scope in scope_list}

        def link_leaf_to_input_layer(sum_leaf,
                                     scope_var,
                                     input_layer,
                                     rand_gen):
            for indicator_node in input_layer.nodes():
                if indicator_node.var == scope_var:
                    rand_weight = rand_gen.random()
                    sum_leaf.add_child(indicator_node, rand_weight)
                    # print(sum_leaf, indicator_node, rand_weight)
            # normalizing
            sum_leaf.normalize()
        #
        # creating a product layer
        #

        def build_product_layer(parent_layer,
                                parent_scope_list,
                                n_max_children,
                                n_scope_children,
                                input_layer,
                                rand_gen):

            # grouping the scopes of the parents
            scope_clusters = cluster_set_scope(parent_scope_list)
            # for each scope add a fixed number of children
            children_lists = {scope: [ProductNode(var_scope=scope)
                                      for i in range(n_scope_children)]
                              for scope in scope_clusters}
            # counting which node is used
            children_counts = {scope: [0 for i in range(n_scope_children)]
                               for scope in scope_clusters}
            # now link those randomly to their parent
            for parent, scope in zip(parent_layer.nodes(), parent_scope_list):
                # only for nodes not becoming leaves
                if len(scope) > 1:
                    # sampling at most n_max_children from those in the same
                    # scope
                    children_scope_list = children_lists[scope]
                    sample_length = min(
                        len(children_scope_list), n_max_children)
                    sampled_ids = rand_gen.sample(range(n_scope_children),
                                                  sample_length)
                    sampled_children = [None for i in range(sample_length)]
                    for i, id in enumerate(sampled_ids):
                        # getting the sampled child
                        sampled_children[i] = children_scope_list[id]
                        # updating its counter
                        children_counts[scope][id] += 1

                    for child in sampled_children:
                        # parent is a sum layer, we must set a random weight
                        rand_weight = rand_gen.random()
                        parent.add_child(child, rand_weight)

                    # we can now normalize it
                    parent.normalize()
                else:
                    # binding the node to the input layer
                    (scope_var,) = scope
                    link_leaf_to_input_layer(parent,
                                             scope_var,
                                             input_layer,
                                             rand_gen)

            # pruning those children never used
            for scope in children_lists.keys():
                children_scope_list = children_lists[scope]
                scope_counts = children_counts[scope]
                used_children = [child
                                 for count, child in zip(scope_counts,
                                                         children_scope_list)
                                 if count > 0]
                children_lists[scope] = used_children

            # creating the layer and new scopelist
            # print('children list val', children_lists.values())
            children_list = [child
                             for child in
                             itertools.chain.from_iterable(
                                 children_lists.values())]
            scope_list = [key
                          for key, child_list in children_lists.items()
                          for elem in child_list]
            # print('children list', children_list)
            # print('scope list', scope_list)
            prod_layer = ProductLayerLinked(children_list)

            return prod_layer, scope_list

        def build_sum_layer(parent_layer,
                            parent_scope_list,
                            rand_gen,
                            max_scope_split=-1,
                            merge_prob=0.5):

            # keeping track of leaves
            # leaf_props = []
            scope_clusters = cluster_set_scope(parent_scope_list)

            # looping through all the parent nodes and their scopes
            # in order to decompose their scope
            dec_scope_list = []
            for scope in parent_scope_list:
                # decomposing their scope into k random pieces
                k = len(scope)
                if 1 < max_scope_split <= len(scope):
                    k = rand_gen.randint(2, max_scope_split)
                shuffled_scope = list(scope)
                rand_gen.shuffle(shuffled_scope)
                dec_scopes = [frozenset(shuffled_scope[i::k])
                              for i in range(k)]
                dec_scope_list.append(dec_scopes)
                # if a decomposed scope consists of only one var, generate a
                # leaf
                # leaves = [(parent, (dec_scope,))
                #           for dec_scope in dec_scopes if len(dec_scope) == 1]
                # leaf_props.extend(leaves)

            # generating a unique decomposition
            used_decs = {}
            children_list = []
            scope_list = []
            for parent, decs in zip(parent_layer.nodes(),
                                    dec_scope_list):
                merge_count = 0
                for scope in decs:
                    sum_node = None
                    try:
                        rand_perc = rand_gen.random()
                        if (merge_count < len(decs) - 1 and
                                rand_perc > merge_prob):
                            sum_node = used_decs[scope]
                            merge_count += 1

                        else:
                            raise Exception()
                    except:
                        # create a node for it
                        sum_node = SumNode(var_scope=scope)
                        children_list.append(sum_node)
                        scope_list.append(scope)
                        used_decs[scope] = sum_node

                    parent.add_child(sum_node)

            # unique_dec = {frozenset(dec) for dec in
            #               itertools.chain.from_iterable(dec_scope_list)}
            # print('unique dec', unique_dec)
            # building a dict scope->child
            # children_dict = {scope: SumNode() for scope in unique_dec}
            # now linking parents to their children
            # for parent, scope in zip(parent_layer.nodes(),
            #                          parent_scope_list):
            #     dec_scopes = dec_scope_list[scope]
            #     for dec in dec_scopes:
            # retrieving children
            # adding it
            #         parent.add_child(children_dict[dec])

            # we already have the nodes and their scopes
            # children_list = [child for child in children_dict.values()]
            # scope_list = [scope for scope in children_dict.keys()]

            sum_layer = SumLayerLinked(nodes=children_list)

            return sum_layer, scope_list

        # if no generator is provided, create a new one
        if rand_gen is None:
            rand_gen = random.Random()

        # create input layer
        # _vars = [2, 3, 2, 2, 4]
        input_layer = CategoricalIndicatorLayerLinked(vars=vars)

        # create root layer
        full_scope = frozenset({i for i in range(len(vars))})
        root = SumNode(var_scope=full_scope)
        root_layer = SumLayerLinked(nodes=[root])
        last_layer = root_layer

        # create top scope list
        last_scope_list = [full_scope]

        layers = [root_layer]
        layer_count = 0
        stop_building = False
        while not stop_building:
            # checking for early termination
            # this one leads to split product nodes into leaves
            if layer_count >= n_layers:
                print('Max level reached, trying to stop')
                max_scope_split = -1

            # build a new layer alternating types
            if isinstance(last_layer, SumLayerLinked):
                print('Building product layer')
                last_layer, last_scope_list = \
                    build_product_layer(last_layer,
                                        last_scope_list,
                                        n_max_children,
                                        n_scope_children,
                                        input_layer,
                                        rand_gen)
            elif isinstance(last_layer, ProductLayerLinked):
                print('Building sum layer')
                last_layer, last_scope_list = \
                    build_sum_layer(last_layer,
                                    last_scope_list,
                                    rand_gen,
                                    max_scope_split,
                                    merge_prob)

            # testing for more nodes to expand
            if last_layer.n_nodes() == 0:
                print('Stop building')
                stop_building = True
            else:
                layers.append(last_layer)
                layer_count += 1

        # checking for early termination
        # if not stop_building:
        #     if isinstance(last_layer, ProductLayerLinked):
        # building a sum layer splitting everything into one
        # length scopes
        #         last_sum_layer, last_scope_list = \
        #             build_sum_layer(last_layer,
        #                             last_scope_list,
        #                             rand_gen,
        #                             max_scope_split=-1)
        # then linking each node to the input layer
        #         for sum_leaf, scope in zip(last_sum_layer.nodes(),
        #                                    last_scope_list):
        #             (scope_var,) = scope
        #             link_leaf_to_input_layer(sum_leaf,
        #                                      scope_var,
        #                                      input_layer,
        #                                      rand_gen)
        #     elif isinstance(last_layer, SumLayerLinked):
        #         pass

        # print('LAYERS ', len(layers), '\n')
        # for i, layer in enumerate(layers):
        #     print('LAYER ', i)
        #     print(layer)
        # print('\n')
        spn = SpnLinked(input_layer=input_layer,
                        layers=layers[::-1])
        # testing
        # scope_list = [
        #     frozenset({1, 3, 4}), frozenset({2, 0}), frozenset({1, 3, 4})]
        # sum_layer = SumLayerLinked(nodes=[SumNode(), SumNode(), SumNode()])

        # prod_layer, scope_list = build_product_layer(
        #     sum_layer, scope_list, 2, 3, input_layer, rand_gen)

        # sum_layer1, scope_list_2 = build_sum_layer(prod_layer,
        #                                            scope_list,
        #                                            rand_gen,
        #                                            max_scope_split=2
        #                                            )
        # prod_layer_2, scope_list_3 = build_product_layer(sum_layer1,
        #                                                  scope_list_2,
        #                                                  2,
        #                                                  3,
        #                                                  input_layer,
        #                                                  rand_gen)
        # create spn from layers
        # spn = SpnLinked(input_layer=input_layer,
        #                 layers=[prod_layer_2, sum_layer1,
        #                         prod_layer, sum_layer, root_layer])
        return spn

    @classmethod
    def layered_linked_spn(cls, root_node):
        """
        Given a simple linked version (parent->children),
        returns a layered one (linked + layers)
        """
        layers = []
        root_layer = None
        input_nodes = []
        layer_nodes = []
        input_layer = None

        # layers.append(root_layer)
        previous_level = None

        # collecting nodes to visit
        open = deque()
        next_open = deque()
        closed = set()

        open.append(root_node)

        while open:
            # getting a node
            current_node = open.popleft()
            current_id = current_node.id

            # has this already been seen?
            if current_id not in closed:
                closed.add(current_id)
                layer_nodes.append(current_node)
                # print('CURRENT NODE')
                # print(current_node)

                # expand it
                for child in current_node.children:
                    # only for non leaf nodes
                    if (isinstance(child, SumNode) or
                            isinstance(child, ProductNode)):
                        next_open.append(child)
                    else:
                        # it must be an input node
                        if child.id not in closed:
                            input_nodes.append(child)
                            closed.add(child.id)

            # open is now empty, but new open not
            if (not open):
                # swap them
                open = next_open
                next_open = deque()

                # and create a new level alternating type
                if previous_level is None:
                    # it is the first level
                    if isinstance(root_node, SumNode):
                        previous_level = SumLayerLinked([root_node])
                    elif isinstance(root_node, ProductNode):
                        previous_level = ProductLayerLinked([root_node])
                elif isinstance(previous_level, SumLayerLinked):
                    previous_level = ProductLayerLinked(layer_nodes)
                elif isinstance(previous_level, ProductLayerLinked):
                    previous_level = SumLayerLinked(layer_nodes)

                layer_nodes = []

                layers.append(previous_level)

        #
        # finishing layers
        #

        #
        # checking for CLTreeNodes
        cltree_leaves = False
        for node in input_nodes:
            if isinstance(node, CLTreeNode):
                cltree_leaves = True
                break

        if cltree_leaves:
            input_layer = CategoricalCLInputLayerLinked(input_nodes)
        else:
            # otherwiise assuming all input nodes are homogeneous
            if isinstance(input_nodes[0], CategoricalSmoothedNode):
                # print('SMOOTH LAYER')
                input_layer = CategoricalSmoothedLayerLinked(input_nodes)
            elif isinstance(input_nodes[0], CategoricalIndicatorNode):
                input_layer = CategoricalIndicatorLayerLinked(input_nodes)

        spn = SpnLinked(input_layer=input_layer,
                        layers=layers[::-1])
        return spn

    @classmethod
    def pruned_spn_from_slices(cls, node_assoc, building_stack, logger=None):
        """
        WRITEME
        """
        if logger is None:
            logger = logging.getLogger(__name__)
        # traversing the building stack
        # to link and prune nodes
        for build_node in reversed(building_stack):

            # current node
            current_id = build_node.id
            # print('+ Current node: %d', current_id)
            current_children_slices = build_node.children
            # print('\tchildren: %r', current_children_slices)
            current_children_weights = build_node.weights
            # print('\tweights: %r', current_children_weights)

            # retrieving corresponding node
            node = node_assoc[current_id]
            # print('retrieved node', node)

            # discriminate by type
            if isinstance(node, SumNode):
                logging.debug('it is a sum node %d', current_id)
                # getting children
                for child_slice, child_weight in zip(current_children_slices,
                                                     current_children_weights):
                    # print(child_slice)
                    # print(child_slice.id)
                    # print(node_assoc)
                    child_id = child_slice.id
                    child_node = node_assoc[child_id]
                    # print(child_node)

                    # checking children types as well
                    if isinstance(child_node, SumNode):
                        logging.debug('++ pruning node: %d', child_node.id)
                        # this shall be pruned
                        for grand_child, grand_child_weight \
                                in zip(child_node.children,
                                       child_node.weights):
                            node.add_child(grand_child,
                                           grand_child_weight *
                                           child_weight)

                    else:
                        logging.debug('+++ Adding it as child: %d',
                                      child_node.id)
                        node.add_child(child_node, child_weight)
                        # print('children added')

            elif isinstance(node, ProductNode):
                logging.debug('it is a product node %d', current_id)
                # linking children
                for child_slice in current_children_slices:
                    child_id = child_slice.id
                    child_node = node_assoc[child_id]

                    # checking for alternating type
                    if isinstance(child_node, ProductNode):
                        logging.debug('++ pruning node: %d', child_node.id)
                        # this shall be pruned
                        for grand_child in child_node.children:
                            node.add_child(grand_child)
                    else:
                        node.add_child(child_node)
                        # print('+++ Linking child %d', child_node.id)

        # this is superfluous, returning a pointer to the root
        root_build_node = building_stack[0]
        return node_assoc[root_build_node.id]

    @classmethod
    def layered_pruned_linked_spn(cls, root_node):
        """
        WRITEME
        """
        #
        # first traverse the spn top down  to collect a bottom up traversal order
        # it could be done in a single pass I suppose, btw...
        building_queue = deque()
        traversal_stack = deque()

        building_queue.append(root_node)

        while building_queue:
            #
            # getting current node
            curr_node = building_queue.popleft()
            #
            # appending it to the stack
            traversal_stack.append(curr_node)
            #
            # considering children
            try:
                for child in curr_node.children:
                    building_queue.append(child)
            except:
                pass
        #
        # now using the inverse traversal order
        for node in reversed(traversal_stack):

            # print('retrieved node', node)

            # discriminate by type
            if isinstance(node, SumNode):

                logging.debug('it is a sum node %d', node.id)
                current_children = node.children[:]
                current_weights = node.weights[:]

                # getting children
                children_to_add = deque()
                children_weights_to_add = deque()
                for child_node, child_weight in zip(current_children,
                                                    current_weights):
                    # print(child_slice)
                    # print(child_slice.id)
                    # print(node_assoc)

                    print(child_node)

                    # checking children types as well
                    if isinstance(child_node, SumNode):
                        # this shall be prune
                        logging.debug('++ pruning node: %d', child_node.id)
                        # del node.children[i]
                        # del node.weights[i]

                        # adding subchildren
                        for grand_child, grand_child_weight \
                                in zip(child_node.children,
                                       child_node.weights):
                            children_to_add.append(grand_child)
                            children_weights_to_add.append(grand_child_weight *
                                                           child_weight)
                            # node.add_child(grand_child,
                            #                grand_child_weight *
                            #                child_weight)

                        # print(
                        #     'remaining  children', [c.id for c in node.children])
                    else:
                        children_to_add.append(child_node)
                        children_weights_to_add.append(child_weight)

                #
                # adding all the children (ex grand children)
                node.children.clear()
                node.weights.clear()
                for child_to_add, weight_to_add in zip(children_to_add, children_weights_to_add):
                    node.add_child(child_to_add, weight_to_add)

                    # else:
                    #     print('+++ Adding it as child: %d', child_node.id)
                    #     node.add_child(child_node, child_weight)
                    #     print('children added')

            elif isinstance(node, ProductNode):

                logging.debug('it is a product node %d', node.id)
                current_children = node.children[:]

                children_to_add = deque()
                # linking children
                for i, child_node in enumerate(current_children):

                    # checking for alternating type
                    if isinstance(child_node, ProductNode):

                        # this shall be pruned
                        logging.debug('++ pruning node: %d', child_node.id)
                        # this must now be useless
                        # del node.children[i]

                        # adding children
                        for grand_child in child_node.children:
                            children_to_add.append(grand_child)
                            # node.add_child(grand_child)
                    else:
                        children_to_add.append(child_node)
                    #     node.add_child(child_node)
                    #     print('+++ Linking child %d', child_node.id)
                #
                # adding grand children
                node.children.clear()
                for child_to_add in children_to_add:
                    node.add_child(child_to_add)
        """
        #
        # printing
        print(\"TRAVERSAL\")
        building_queue = deque()
        building_queue.append(root_node)

        while building_queue:
            #
            # getting current node
            curr_node = building_queue.popleft()
            #
            # appending it to the stack
            print(curr_node)
            #
            # considering children
            try:
                for child in curr_node.children:
                    building_queue.append(child)
            except:
                pass
        """

        #
        # now transforming it layer wise
        # spn = SpnFactory.layered_linked_spn(root_node)
        return root_node


