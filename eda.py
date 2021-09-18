import numpy as np
import pandas as pd
import streamlit as st
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.datasets import load_diabetes, load_boston

st.title("The EDA App")
st.write("Upload dataset in CSV format and the app will show its Exploratory Data Analysis.")

# with st.sidebar.header('1. Upload your CSV data'):
#   uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
#   st.sidebar.markdown("[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)")

# st.markdown("Upload your CSV file")

uploaded_file = st.file_uploader("Upload your CSV file",type=["csv"])
st.markdown("[Example CSV input file](https://github.com/Gaurav3099/Heart-Attack-Analysis/blob/main/heart.csv)")


if uploaded_file is not None:
  @st.cache
  def load_csv():
    csv = pd.read_csv(uploaded_file)
    return csv 
  df = load_csv()
  pr = ProfileReport(df, explorative=True)
  st.header("Input DataFrame")
  st.write(df)
  st.write('----')
  st.header('Pandas Profiling Report')
  st_profile_report(pr)

else:
  st.info('Awaiting for CSV file to be uploaded.')
  if st.button('Press to use Example Dataset'):
    st.markdown('The Boston housing dataset is used as the example.')
    @st.cache
    def load_data():
      boston = load_boston()
      X = pd.DataFrame(boston.data, columns=boston.feature_names)
      Y = pd.Series(boston.target, name='response')
      a = pd.concat( [X,Y], axis=1 )
      return a
    df = load_data()
    pr = ProfileReport(df, explorative=True)
    st.header("Input DataFrame")
    st.write(df)
    st.write('---')
    st.header('Pandas Profiling Report')
    st_profile_report(pr)
    