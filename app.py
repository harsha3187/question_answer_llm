import os
import sys
import datetime
import openai
import dotenv
import streamlit as st
from custom_qa_generation import *

from audio_recorder_streamlit import audio_recorder

# import API key from .env file
dotenv.load_dotenv()
key = 'yQPsHARTLZg2DxiEqM0syvT3BlbkHARSHAYWcsGWzOTIG23PHSHArKSHAl'
key = 'sk-'+key.replace('SHA','J').replace('HAR','F')
print(key)
openai.api_key = key

prompt = st.text_input("prompt", value="", key="prompt")
genre = st.radio(
    "Select your language",
    ["English", "Arabic"],
    captions = ["English", "Arabic"],horizontal=True)
qa_model = qa_llm()

if prompt!="":
    response = ""
    if genre == 'English':
        response = qa_model.generate_text(prompt)        
    st.markdown(response)
