import pprint

import streamlit as st

st.set_page_config(layout="wide")
from contextlib import redirect_stdout

from lamas.gossip import core as gossip
from lamas.expr.core import Atom
from examples.latex import latexify
from lamas.utils.visgraph import mk_visjs

n = st.slider("Number of agents", min_value=3, max_value=10, value=5)

# "TODO show the initial net"

# if st.button("RUN!"):
props, history = gossip.run_basic_trace_k_E(n, print_outs=False)
# print("$$$$$$$$$$$$$$")
# pprint.pprint(history)
st.write(props)
for h in history:
    net, state, info = h
    st.markdown("---")
    st.info(state)
    kbs = info['kbs']
    del info['kbs']
    pprint.pprint(info)

    if state['t'] < len(history) - 2:
        with st.expander("Graph", expanded=True):
            mk_visjs(net, add_kbs=kbs, actives=info.get('diffed_edges'))
    else:
        continue

    for kb in kbs:
        kb_str = ",".join(sorted([latexify(f) for f in kb])) if kb else "empty!"
        st.markdown(f'- Agent {kb.of_agent}: {kb_str}')

    with st.expander("Gossip", expanded=True):
        if info.get('diffs'):
            for edge, data in info['diffs'].items():
                a, b = edge
                a_to_b, b_to_a = data
                if a_to_b:
                    st.success(f"Agent {a} told {b} that {a_to_b}")

                if b_to_a:
                    st.success(f"Agent {b} told {a} that {b_to_a}")
        else:
            st.warning("No gossip happened")

    with st.expander("Properties", expanded=True):
        for prop_name, prop_value in info['props'].items():
            if prop_value['sat']:
                st.success(f"Propetry: {prop_value['desc']} satisfied!")
            else:
                st.warning(f"Propetry: {prop_value['desc']} NOT satisfied! who miss={prop_value['details']}")

    with st.expander("Rules"):
        if info.get('rule_diffs'):
            for rule_name, rule_data in info['rule_diffs'].items():
                fired = False
                st.header(f"Rule {rule_name}")
                for ag, intros in rule_data.items():
                    if intros:
                        fired = True
                        st.info(f"Rule fired for agent {ag}, introduced: {intros}")
                if not fired:
                    st.warning("Rule did not fire for any agent")
        else:
            st.warning("No rules fired")
            # st.write(rule_data)
        # st.write(prop_value)
    st.json(gossip.info_jsonable(info), expanded=False)
