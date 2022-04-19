# import dependencies
import pandas as pd
import streamlit as st

# load data
@st.cache
def load_data():
    df_sake = pd.read_csv('data/sake_list_final.csv').drop(columns="Unnamed: 0")
    return df_sake

df_sake = load_data()

# create sidebar
add_sidebar = st.sidebar.selectbox("sidebar fun times", ("pee pee", "poo poo"))

if __name__ == "__main__":
    df_sake = load_data()
    print(df_sake.tail(3))
