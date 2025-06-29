import os
import pandas as pd
import nltk
import spacy
import streamlit as st
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from textblob import TextBlob

# Downloads
nltk.download('punkt')
nltk.download('stopwords')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Cache training so it only runs once
@st.cache_resource
def train_bot():
    # Load MedQuAD dataset
    df = pd.read_csv("medquad.csv")

    # Flatten Q&A pairs
    qa_pairs = []
    for _, row in df.iterrows():
        question = str(row['question']).strip()
        answer = str(row['answer']).strip()
        if question and answer:
            qa_pairs.extend([question, answer])

    # Initialize chatbot
    chatbot = ChatBot(
        'HealthBot',
        logic_adapters=['chatterbot.logic.BestMatch'],
        storage_adapter='chatterbot.storage.SQLStorageAdapter',
        database_uri='sqlite:///database.sqlite3'
    )

    # Only train if DB doesn't exist
    if not os.path.exists("database.sqlite3"):
        trainer = ListTrainer(chatbot)
        trainer.train(qa_pairs)

    return chatbot

chatbot = train_bot()

# Streamlit UI
st.title("💬 Health & Wellness Chatbot")
st.markdown("Ask a health-related question based on MedQuAD data.")

user_input = st.text_input("🩺 Your question:")

if user_input:
    sentiment = TextBlob(user_input).sentiment
    response = chatbot.get_response(user_input)

    if sentiment.polarity < -0.5:
        st.write("😔 I'm sorry you're feeling down. Here's some help:")

    st.write(f"🤖 {response}")
