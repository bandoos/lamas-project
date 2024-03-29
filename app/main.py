#!/usr/bin/env python3
import streamlit as st
import lamas.gossip.core as gossip
import streamlit.components.v1 as comps
from lamas.utils.visgraph import mk_visjs

st.title("Investigating Simple Gossip Networks and Epistemic Logic")
st.header("A Study of Information Propagation")
st.markdown("LAMAS Proejct (Yara Bikowski, Luca Bandelli)")

st.markdown("""
# Site Map
You can navigate preferably using the sidebar on the left. The available pages are:
- this `home-page`
- `text trace` see a detailed text trace of the system <a href='/text_trace'>goto</a>.
- `trace` see a detailed graphical trace of the system <a href='/trace'>goto</a>.
- `experiment` reproduce the results presented in the report <a href='/experiment'>goto</a>. 
- `report` consult the pdf of the Project report <a href='/report'>goto</a>. 

## Introduction
On this website, you can see information and implementation of our project on gossip networks and Epistemic logic for the Logical Aspects of Multi-agent Systems course of
the Rijksuniversiteit Groningen. 
 
This project was realized by Luca Bandelli and Yara Bikowski.

We investigated the spread of information in a cyclic gossip network using various concepts of Epistemic logic. For this, 
we made a report with the explanations of methods and simplifications that we use (see report page for more details). 
This report also contains examples of how information would spread throughout a network. 
This website includes Python program that displays the evolution of the gossip network based. 

""")

st.markdown(r"""
We make use of the following 2 priciples from the system $KEC(m)$:

- (R2): From $\varphi$ follows $K_i \varphi$
- (A6): $E\varphi \leftrightarrow (K_1\varphi \land ... \land K_m\varphi)$

From (A6) via a single Equivalence Elimination we get the rule we are looking for

- $E\varphi \leftrightarrow (K_1\varphi \land ... \land K_m\varphi)$ (A6)
- $(K_1\varphi \land ... \land K_m\varphi) \rightarrow E\varphi$  (EE:1)


""", unsafe_allow_html=True)

simple_g = gossip.create_cycle_graph(5)
simple_g.nodes[0]['kb'] |= {gossip.Atom('p'), gossip.K(0, gossip.Atom('p'))}

# gossip.set_one_based_labels(simple_g)

# fig = gossip.simple_draw(simple_g, figsize=(7, 7))
# st.pyplot(fig)

st.markdown("""
#### Example of cyclic gossip network 
At t=0 only the first agent knows the secret $p$.
""")

mk_visjs(simple_g, [dat['kb'] for _, dat in simple_g.nodes(data=True)])

# \begin{itemize}
# \item (R2):
# \item (A6):
# \end{itemize}
# From axiom A6, we can get $(K_1\varphi \land ... \land K_m\varphi) \rightarrow E\varphi$ using equivalence-elimination as can be seen from the following proof:
# \begin{enumerate}
# \item $E\varphi \leftrightarrow (K_1\varphi \land ... \land K_m\varphi)$ \hfill (A6)
# \item $(K_1\varphi \land ... \land K_m\varphi) \rightarrow E\varphi$ \hfill (EE:1)
# \end{enumerate}
