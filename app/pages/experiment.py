import streamlit as st
import lamas.gossip.core as gossip

st.set_page_config(layout="wide")

st.title("Reproducing the experiment results")

values = st.slider(
    'Angent number sweep',
    3, 50, (3, 40))
st.write('Values:', values)


@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')


@st.cache_data
def run(ns):
    return gossip.run_exp_for(ns)


if st.button("Run!"):
    _min, _max = values
    values = list(range(_min, _max + 1))
    df_res = run(values)

    df_wide = gossip.res_to_wide(df_res)
    st.write(df_wide)

    csv = convert_df(df_wide)

    fig1 = gossip.make_plot_one(df_res)
    fig2 = gossip.make_plot_two(df_res)

    cols = st.columns(2)
    with cols[0]:
        st.pyplot(fig1)
    with cols[1]:
        st.pyplot(fig2)

    st.download_button(
        "Press to Download",
        csv,
        f"results_gossip_{_min}_{_max}.csv",
        "text/csv",
        key='download-csv'
    )
