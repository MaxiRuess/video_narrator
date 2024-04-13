import os
import streamlit as st
import cv2
import time
from PIL import Image
import numpy as np
from openai import OpenAI
import base64

import simpleaudio as sa
import errno
from elevenlabs import generate, play, set_api_key, voices
from dotenv import load_dotenv
from capture import main as capture_main
from narrator import main as narrator_main

def main(): 
    
    load_dotenv()
    
    st.header("Welcome to Narrator")    
    st.write("This app uses AI to describe images and narrate them.")
    
    st.subheader("Step 1: Capture an image")
    st.write("Click the button below to capture an image.")
    
    if st.button("Capture Image"):
        capture_main()
        
    
    st.subheader("Step 2: Describe the image")
    
    if st.button("Describe Image"):
        narrator_main()
        
    
    if __name__ == "__main__":
        main()
    
