import streamlit as st
from contextlib import redirect_stdout

from lamas.gossip import core as gossip
from lamas.expr.core import Atom

n = st.slider("Number of agents", min_value=3, max_value=10)

"TODO show the initial net"

if st.button("RUN!"):
    trace = gossip.run_basic_demo_get_trace(n)
    st.text(trace)

#
# net = gossip.create_cycle_graph(n)
#
# net.nodes[0]['kb'] |= {Atom('p')}
#
# st.text(gossip.show_kbs(net))
# conn_gen = gossip.my_connections(n)
#
# if st.button("step"):
#     gossip.step_gossip(net, conn_gen)
#     ...
