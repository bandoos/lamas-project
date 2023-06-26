import copy

import networkx as nx
from typing import Set, List, Tuple, Callable, Generator, Optional
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from pprint import pprint
from benedict import BeneDict
from functools import partial
from itertools import chain
from lamas.expr.core import Expr, And, Or, Not, K, E, Atom
from itertools import groupby
from collections import OrderedDict
import pyvis.network as visnet
from itertools import islice, cycle, tee
from collections import OrderedDict
from typing import Any
from contextlib import redirect_stdout
import io
from copy import deepcopy

import pandas as pd
import seaborn as sns

EdgeList = List[Tuple[int, int]]
EdgeListGen = Generator[EdgeList, None, None]
KRule = Callable[[Set[Expr]], Set[Expr]]


# Let's defien a structure to store the formulas making up the knowledge
# base of an individual agent

class KB(Set[Expr]):
    """A knowledge base is simply a set mutable of expressions"""

    def __init__(self, of_agent, init=()):
        super().__init__(init)
        self.of_agent = of_agent

    def __repr__(self):
        formulas_repr = ', '.join(sorted([repr(x) for x in self]))
        return f'KB[{self.of_agent}]({formulas_repr})'

    def display(self):
        return "\n".join(sorted([repr(x) for x in self])) or "{}"

    def apply_rule(self, k_rule: KRule) -> "KB":
        return KB(self.of_agent, k_rule(self.of_agent, self))

    def get_copy(self):
        return KB(self.of_agent, set(self))


# necessitation
def KRULE_atomic_necessitation(of_agent: int, exprs: Set[Expr]) -> Set[Expr]:
    out = set(exprs)
    for e in exprs:
        if isinstance(e, Atom) and not K(of_agent, e) in exprs:
            out.add(K(of_agent, e))
    return out


def greedy_group_by(it, key):
    ret = OrderedDict()

    for k, vs in groupby(it, key):
        if not ret.get(k):
            ret[k] = []
        ret[k].extend(vs)
    return ret


# E-introduction

# e intro is not pure kb rule, it also depends on the total number of agents

def KRULE_E_introduction(n_agents, of_agent: int, exprs: Set[Expr]) -> Set[Expr]:
    out = set(exprs)  # copy the input

    # for each K-formula, if K_i phi for all i then E phi
    k_formulas = list(filter(lambda e: isinstance(e, K), exprs))

    # group the formulas by the operand
    k_by_phi = greedy_group_by(k_formulas, lambda k: k.rand)

    # for each phi, and list of k-formulas involving that phi
    for phi, kfs in k_by_phi.items():
        # get the set agents that knows phi
        ag_ids = set([k.i for k in kfs])
        # if all agents are covered
        if set(range(n_agents)) ^ ag_ids == set():
            out.add(E(phi))
    return out


def create_cycle_graph(n_agents) -> nx.Graph:
    """
    Create an undirected cycle graph of `n_agents`.

    returns a nx.Graph structure representing the overall connectivity
    at gossip-layer of the agents.

    Sets the 'pos' attribute on the nodes accroding to the circular layout
    """
    # 1. create nx cycle graph for n_agents
    g = nx.cycle_graph(n_agents)
    # 2. compute layout once for consitent rendering
    layout = nx.layout.circular_layout(g)
    # 3. store necassry infomation in the node metadata
    for node, (x, y) in layout.items():
        # Set the node positions in the networkx graph
        g.nodes[node]['pos'] = (x, y)
        # Create an empty knowledge base for each agent
        g.nodes[node]['kb'] = KB(of_agent=node)
    return g


def print_kbs(g: nx.Graph):
    "Helper to print all the kbs of agents in graph"
    for id, data in g.nodes(data=True):
        print(data['kb'])


def show_kbs(g: nx.Graph):
    """Helper to print all the kbs of agents in graph"""
    return "\n".join([repr(data['kb']) for _, data in g.nodes(data=True)])


def get_kbs(g: nx.Graph):
    return [data['kb'].get_copy() for _, data in g.nodes(data=True)]


def simple_draw(g, with_pos=True, title=None, figsize=(8, 8)):
    """The simplest way to visualize the network.
    if with_pos is True (default), will lookup the 'pos' attribute of the nodes
    to determine position in the render.
    """
    fig, ax = plt.subplots(figsize=figsize)
    if title:
        ax.set_title(title)
    args = {'ax': ax}
    if with_pos:
        args['pos'] = {node: data['pos'] for node, data in g.nodes(data=True)}
    nx.draw(g, with_labels=True, **args)
    return fig


def set_one_based_labels(g: nx.Graph):
    for node in g.nodes:
        g.nodes[node]['label'] = str(1 + node)


def graph_for_viz(g: nx.Graph, notebook=False, add_kbs=True, one_based=False):
    g = g.copy()
    node_ids = list(g.nodes)
    for node_id in node_ids:
        node = g.nodes[node_id]
        kb = node['kb']
        del node['kb']
        node['title'] = repr(kb)
        node['label'] = str(node_id) if not one_based else str(1 + node_id)

        if add_kbs:
            kb_node_id = f'kb__{node_id}'
            g.add_node(kb_node_id, label=kb.display(), shape="box")
            g.add_edge(node_id, kb_node_id, color="grey", dashes=True)

    vn = visnet.Network(notebook=True, cdn_resources='remote')
    vn.from_nx(g)
    return vn


def dropn(it, n):
    _, it = tee(it, 2)
    for i in range(n):
        next(it)
    yield from it


def my_connections(n_agents) -> EdgeListGen:
    # at each cycle there will always be n_agents//2 paris selected
    # if n_agents is odd, one is excluded each time
    n_conns = n_agents // 2
    i_hat = 0
    while True:
        conns = []
        for i in range(n_conns):
            a = (i_hat + 2 * i) % n_agents
            b = (i_hat + 2 * i + 1) % n_agents
            conns.append((a, b) if a < b else (b, a))
        yield sorted(conns, key=lambda x: x[0])
        i_hat = (i_hat - 1) % n_agents


def symmetric_gossip(g: nx.Graph, gossip_edges: EdgeList, verbose=False):
    # for each edge involved in the gossip round
    info = {'diffs': {}, 'diffed_edges': set(), 'edges': gossip_edges}
    for a, b in gossip_edges:
        info['diffs'][(a, b)] = [None, None]

        # get the kbs of the 2 agents
        kb_a = g.nodes[a]['kb']
        kb_b = g.nodes[b]['kb']
        verbose and print(f"gossiping: {a} <-> {b}, overall_diff={kb_a ^ kb_b}")

        # compute the diffs
        diff_a_b = kb_a - kb_b  # what does a have that b does not?
        diff_b_a = kb_b - kb_a  # what does b have that a does not?

        if not diff_a_b and not diff_b_a:
            verbose and print(f"\tNo diff to gossip about")
        else:
            info['diffed_edges'].add((a, b))
            if diff_a_b:  # if a has smth that b does not
                kb_b |= diff_a_b
                info['diffs'][(a, b)][0] = diff_a_b
                verbose and print(f"\t {a} -[{diff_a_b}]-> {b}")

            if diff_b_a:  # if b has smth that a does not
                kb_a |= diff_b_a
                info['diffs'][(a, b)][1] = diff_b_a
                verbose and print(f"\t {b} -[{diff_b_a}]-> {a}")

            # now

    return info


# we at each timstep the edges along which the gossip flows change
def step_gossip(g: nx.Graph, conn_gen: EdgeListGen, verbose=False):
    return symmetric_gossip(g, next(conn_gen), verbose=verbose)


def fire_k_rule(ag_meta, k_rule: KRule):
    old_kb: KB = ag_meta['kb']
    new_kb = old_kb.apply_rule(k_rule)
    ag_meta['kb'] = new_kb
    return old_kb ^ new_kb


def fire_rule_all_agents(g: nx.Graph, k_rule: KRule):
    rule_diffs = {}
    for node in g.nodes:
        diff = fire_k_rule(g.nodes[node], k_rule)
        rule_diffs[node] = diff
    return rule_diffs


def create_rules(rules, g: nx.Graph):
    return OrderedDict([
        (k, make_rule(g))
        for k, make_rule in rules.items()
    ])


def fire_all_rules(rules, g: nx.Graph, after_application=None):
    rule_diffs = {}
    for rule_name, rule in rules.items():
        diffs = fire_rule_all_agents(g, rule)
        rule_diffs[rule_name] = diffs
        if diffs and after_application:
            after_application(g)
    return rule_diffs


rules_neccess_and_E = OrderedDict(
    [('necess', lambda _: KRULE_atomic_necessitation),
     # the i introduction rule must be parametrized on the number of agents
     ('E-intro', lambda g: partial(KRULE_E_introduction, len(g)))]
)


@dataclass
class Property:
    name: str
    desc: str
    check_fn: Callable[[nx.Graph], Tuple[bool, Any]]

    def check(self, g: nx.Graph):
        sat, details = self.check_fn(g)
        return {'sat': sat, 'details': details, 'desc': self.desc}


def property_all_have_phi(phi, name=None):
    """Create a property that checks if all agents have `phi` in their kb"""

    def _check(g: nx.Graph):
        who_not = [node for node, data in g.nodes(data=True) if phi not in data['kb']]
        return len(who_not) == 0, who_not

    description = f'all have ({repr(phi)})'
    return Property(
        name=name or description,
        check_fn=_check,
        desc=description
    )


def check_all_props(props, g: nx.Graph):
    return {p.name: p.check(g) for p in props}


def run_demo(n_agents,
             inject_0: Set[Expr],
             k_rule_factories=None,
             check_properties=None,
             do_yield=False,
             verbose=True,
             ):
    # create the net
    gossip_net = create_cycle_graph(n_agents)
    # inject initial fact
    gossip_net.nodes[0]['kb'] |= inject_0

    # intialize overall state of the system
    state = BeneDict(
        t=-1,
        last_change=-1,
    )
    info_0 = {}

    if check_properties is None:
        check_properties = []

    if k_rule_factories is None:
        rules = {}
    else:
        rules = create_rules(k_rule_factories, gossip_net)

    if rules:
        rule_diffs_0 = fire_all_rules(rules, gossip_net)
        verbose and print("Application rules after intial injection gave:", rule_diffs_0)
        info_0['rule_diffs'] = rule_diffs_0

    info_0['props'] = check_all_props(check_properties, gossip_net)
    info_0['kbs'] = []

    for node, data in gossip_net.nodes(data=True):
        # print(data['kb'])
        info_0['kbs'].append(data['kb'].get_copy())

    if do_yield:
        yield deepcopy(gossip_net), state.copy(), info_0

    # print initial setting
    verbose and print("Initial state (t=-1):")
    verbose and print_kbs(gossip_net)
    verbose and print('-' * 42)

    edge_gen = my_connections(n_agents)

    while True:
        state.t += 1  # t starts from -1 so the first time t=0
        # if the other direction did not change at the previous step we can stop.
        if state.last_change < state.t - 1:
            verbose and print(f"====Stopping in at t={state.t} last_change={state.last_change}")
            break

        # get the edges to use for this iteration
        edges = next(edge_gen)

        # do exchange
        info = symmetric_gossip(gossip_net, edges)
        did_change = len(info['diffed_edges']) > 0
        if did_change:  # if there was a change
            state['last_change'] = state.t  # then set that a fwd change happend a time t
            # if there are rules to fire
            if rules:
                # fire the rules sequenctially
                rule_diffs = fire_all_rules(
                    rules,
                    gossip_net,
                    # after each application propagate the new diffs
                    lambda net: symmetric_gossip(net, edges)
                )
                info['rule_diffs'] = rule_diffs
                verbose and print("Application of rules gave:", rule_diffs)

        info['props'] = check_all_props(check_properties, gossip_net)
        info['kbs'] = []
        for node, data in gossip_net.nodes(data=True):
            # print(data['kb'])
            info['kbs'].append(data['kb'].get_copy())

        # print("======", info['kbs'])

        verbose and print(f"Info [state={state}]:")
        verbose and pprint(info, indent=2)
        verbose and print_kbs(gossip_net)
        verbose and print('-' * 42)

        if do_yield:
            tup = deepcopy(gossip_net), state.copy(), info
            yield tup


demo_k_E_props = [
    property_all_have_phi(Atom('p'), name='all_p'),
    property_all_have_phi(K(1, Atom('p')), name='all_K_1_p'),
    property_all_have_phi(E(Atom('p')), name='all_E_p')
]


def run_basic_demo_k_E(n_agents):
    return run_demo(n_agents,
                    inject_0={Atom('p')},
                    k_rule_factories=rules_neccess_and_E,
                    check_properties=demo_k_E_props
                    )


def run_basic_demo_get_trace(n_agents):
    out = io.StringIO()
    with redirect_stdout(out):
        list(run_basic_demo_k_E(n_agents))
    trace = out.getvalue()
    return trace


def run_basic_trace_k_E(n_agents, print_outs=True):
    demo_gen = run_demo(n_agents,
                        inject_0={Atom('p')},
                        k_rule_factories=rules_neccess_and_E,
                        check_properties=demo_k_E_props,
                        verbose=False,
                        do_yield=True
                        )

    props_index = None
    history = []

    for update in demo_gen:
        net, state, info = update
        # print("------", info.get('kbs'))

        history.append((net, state, info))
        # print("===================hist:")
        # pprint(history)
        if state['t'] == -1:
            props_index = {}
            for pname, pval in info['props'].items():
                props_index[pname] = -1
        else:
            for pname, pval in info['props'].items():
                # if the prop was never sat before
                if props_index[pname] == -1 and pval['sat']:
                    props_index[pname] = state['t']

        if print_outs:
            print(state)
            pprint(info)
            print_kbs(net)
            print('-' * 42)

    return props_index, history.copy()


def info_jsonable(info):
    ret = deepcopy(info)
    if info.get('diffs'):
        new = {}
        for e in info['diffs']:
            new[str(e)] = info['diffs'][e]
        ret['diffs'] = new
    return ret


def run_exp_for(ns):
    records = []
    for n in ns:
        r, _ = run_basic_trace_k_E(n, print_outs=False)
        r['all_E_p/all_p'] = r['all_E_p'] / r['all_p']
        r['all_E_p/n'] = r['all_E_p'] / n
        r['all_p/n'] = r['all_p'] / n
        for k, v in r.items():
            records.append({"prop": k, "value": v, 'n_agents': n})
    return pd.DataFrame(records)


def make_plot_one(df_res):
    df_plot_1 = df_res[df_res.prop.isin(['all_p',
                                         'all_K_1_p',
                                         'all_E_p',
                                         'all_E_p/all_p'])]
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.lineplot(data=df_plot_1,
                 x="n_agents",
                 y="value",
                 hue="prop",
                 style="prop",
                 # size="prop"
                 )
    plt.title("Steps required to reach each global property as function of Number of agents")
    plt.xlabel("#agents")
    plt.ylabel("#steps")
    plt.xticks(df_plot_1['n_agents'].unique(), rotation=45)
    plt.yticks(df_plot_1['value'].unique(), rotation=45)
    plt.grid()
    return fig


def make_plot_two(df_res, log_scale=False):
    df_plot_2 = df_res[df_res.prop.isin(['all_E_p/n', 'all_p/n'])]
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.lineplot(data=df_plot_2,
                 x="n_agents",
                 y="value",
                 hue="prop",
                 style="prop",
                 # size="prop"
                 )
    plt.xlabel("#agents")
    plt.ylabel("value")
    plt.xticks(df_plot_2['n_agents'].unique(), rotation=45)
    # plt.yticks(df_plot_2['value'].unique())
    if log_scale:
        plt.xscale('log')
        plt.yscale('log')
    plt.grid()
    plt.title("Ratios of steps-to-property to number of agents")
    plt.axhline(y=1)
    plt.axhline(y=0.5, c="orange")
    return fig


def res_to_wide(df_res):
    return df_res.pivot(index='n_agents', columns='prop', values='value').reset_index()
