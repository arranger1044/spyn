import graphviz

from cltree.cltree import CLTree


def add_nodes(graph, nodes):
    """
    """
    for n in nodes:
        if isinstance(n, tuple):
            graph.node(n[0], **n[1])
        else:
            graph.node(n)
    return graph


def add_edges(graph, edges):
    for e in edges:
        if isinstance(e[0], tuple):
            graph.edge(*e[0], **e[1])
        else:
            graph.edge(*e)
    return graph

"""
    styles = {
        'graph': {
            'label': 'A Fancy Graph',
            'fontsize': '16',
            'fontcolor': 'white',
            'bgcolor': '#333333',
            'rankdir': 'BT',
        },
        'nodes': {
            'fontname': 'Helvetica',
            'shape': 'hexagon',
            'fontcolor': 'white',
            'color': 'white',
            'style': 'filled',
            'fillcolor': '#006699',
        },
        'edges': {
            'style': 'dashed',
            'color': 'white',
            'arrowhead': 'open',
            'fontname': 'Courier',
            'fontsize': '12',
            'fontcolor': 'white',
        }
    }
"""


def apply_styles(graph, styles):
    graph.graph_attr.update(
        ('graph' in styles and styles['graph']) or {}
    )
    graph.node_attr.update(
        ('nodes' in styles and styles['nodes']) or {}
    )
    graph.edge_attr.update(
        ('edges' in styles and styles['edges']) or {}
    )
    return graph


def cltree_2_dot(cltree,
                 format='svg',
                 styles=None,
                 output='cltree',
                 prefix='clt_',
                 cluster=None):
    """
    WRITEME
    """

    #
    # create the graph
    if cluster is not None:
        graph = graphviz.Digraph(format=format,
                                 name=cluster)
    else:
        graph = graphviz.Digraph(format=format)

    #
    # getting a prefix for the nodes
    node_prefix = prefix

    for node in map(str, cltree.features):
        graph.node(node_prefix + node, label=node)

    n_features = len(cltree.features)

    # print('CLTree', cltree._tree)

    #
    # adding parent->children edges
    for child_id, parent_id in zip(range(1, n_features),
                                   cltree._tree[1:]):
        graph.edge(node_prefix + str(cltree.features[parent_id]),
                   node_prefix + str(cltree.features[child_id]))

    #
    # applying styles
    if styles is not None:
        apply_styles(graph, styles)

    #
    # optionally saving in the specified format
    if output is not None:
        graph.render(output)

    return graph
