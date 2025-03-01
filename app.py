import os
import traceback
import streamlit as st
import speech_recognition as sr
from transformers import pipeline

st.set_page_config(layout="wide")

st.title("🎧 Audio Analysis 📝")

st.sidebar.title("Audio Analysis")
st.sidebar.write("The Audio Analysis app is a powerful tool that allows you to analyze audio files and gain valuable insights from them. It combines speech recognition and sentiment analysis techniques to transcribe the audio and determine the sentiment expressed within it.")

st.sidebar.header("Upload Audio")
audio_file = st.sidebar.file_uploader("Browse", type=["wav"])
upload_button = st.sidebar.button("Upload")

def perform_sentiment_analysis(text):
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    sentiment_analysis = pipeline("sentiment-analysis", model=model_name)
    results = sentiment_analysis(text)
    sentiment_label = results[0]['label']
    sentiment_score = results[0]['score']
    return sentiment_label, sentiment_score

def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        transcribed_text = recognizer.recognize_google(audio_data)
    return transcribed_text

if audio_file and upload_button:
    try:
        # Transcribe the audio file
        transcribed_text = transcribe_audio(audio_file)
        
        # Display transcribed text
        st.header("Transcribed Text")
        st.text_area("Transcribed Text", transcribed_text, height=200)
        
        # Perform sentiment analysis
        sentiment_label, sentiment_score = perform_sentiment_analysis(transcribed_text)
        
        # Display sentiment analysis results
        st.header("Sentiment Analysis")
        negative_icon = "👎"
        neutral_icon = "😐"
        positive_icon = "👍"
        
        if sentiment_label == "NEGATIVE":
            st.write(f"{negative_icon} Negative (Score: {sentiment_score:.2f})", unsafe_allow_html=True)
        elif sentiment_label == "NEUTRAL":
            st.write(f"{neutral_icon} Neutral (Score: {sentiment_score:.2f})", unsafe_allow_html=True)
        elif sentiment_label == "POSITIVE":
            st.write(f"{positive_icon} Positive (Score: {sentiment_score:.2f})", unsafe_allow_html=True)

        st.info(
            "The sentiment score measures how strongly positive, negative, or neutral the feelings or opinions are. "
            "A higher score indicates a positive sentiment, while a lower score indicates a negative sentiment."
        )
    
    except Exception as ex:
        st.error("Error occurred during audio transcription and sentiment analysis.")
        st.error(str(ex))
 
