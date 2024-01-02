from dotenv import load_dotenv
import os

from pandasai.smart_dataframe import SmartDataframe

from pandasai.llm.openai import OpenAI

import streamlit as st
import pandas as pd
import pygwalker as pyg
import streamlit.components.v1 as comp




load_dotenv()
api_key = os.environ["api_key"]

llm = OpenAI(api_token="api_key")

st.set_page_config(
    page_title = 'Data Visualizer',
    layout = 'wide',
)

st.title("Visualize your Data")



@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)


uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df = load_data(uploaded_file)
    pyg_html = pyg.walk(df,return_html=True)
    comp.html(pyg_html, height=900, scrolling=True)

    prompt = st.text_area("Enter your prompt:")
    pandas_ai = SmartDataframe(df, config={"llm": llm})


    if st.button("Generate"):
        if prompt:
            with st.spinner("Generating response..."):
                st.write(pandas_ai.chat(df, prompt))
        else:
            st.warning("Please enter a prompt.")
  
else:
    st.write("Please upload a valid file")
   








    