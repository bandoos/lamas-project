from streamlit_agraph import agraph, Node, Edge, Config
from lamas.gossip import core as gossip


def is_active(a, b, actives):
    return (a, b) in actives or (b, a) in actives


def is_inactive(a, b, actives):
    return not is_active(a, b, actives)


def mk_visjs(net,
             add_kbs=None,
             actives=None,
             ):
    nodes = []
    edges = []

    for node, data in net.nodes(data=True):
        nodes.append(Node(id=node,
                          label=f'Agent-{node + 1}',
                          shape="circular"
                          ))

        if add_kbs:
            kb = add_kbs[node]
            kb_node_id = f'kb__{node}'
            nodes.append(Node(kb_node_id, label=kb.display(), shape="box"))
            edges.append(Edge(kb_node_id, node, color="grey", dashes=True))

    for a, b in net.edges:
        inact = False if actives is None else is_inactive(a, b, actives)
        edges.append(Edge(source=a, target=b, color="red" if inact else "green", type="curvedCW"))
        edges.append(Edge(source=b, target=a, color="red" if inact else "green"))
        # hidden=False if actives is None else inact))
        # edges.append(Edge(source=b, target=a, hidden=False if actives is None else inact))

    config = Config(width=750,
                    height=950,
                    directed=True,
                    physics=True,
                    hierarchical=False,
                    # **kwargs
                    )

    return agraph(nodes=nodes,
                  edges=edges,
                  config=config)
