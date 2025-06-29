import streamlit as st
import pandas as pd
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
import os
import pickle

# Cache for chatbot
@st.cache_resource
def get_trained_chatbot():
    if os.path.exists("chatbot.pkl"):
        with open("chatbot.pkl", "rb") as f:
            return pickle.load(f)
    
    chatbot = ChatBot("HealthBot")
    trainer = ListTrainer(chatbot)

    # Load your dataset
    df = pd.read_csv("medquad.csv")  # Make sure this file is in your repo

    # Train the bot on the QA pairs
    qa_pairs = []
    for _, row in df.iterrows():
        qa_pairs.append(str(row['question']))
        qa_pairs.append(str(row['answer']))

    trainer.train(qa_pairs)

    with open("chatbot.pkl", "wb") as f:
        pickle.dump(chatbot, f)

    return chatbot

# Streamlit UI
st.title("🩺 NLP Health Chatbot")
st.markdown("Ask a health-related question:")

user_input = st.text_input("You:")

if user_input:
    bot = get_trained_chatbot()
    response = bot.get_response(user_input)
    st.write(f"**HealthBot:** {response}")
