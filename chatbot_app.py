import nltk
import streamlit as st
from textblob import TextBlob
from chatterbot import ChatBot

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load the pre-trained chatbot from the SQLite database
chatbot = ChatBot(
    "HealthBot",
    storage_adapter="chatterbot.storage.SQLStorageAdapter",
    database_uri="sqlite:///database.sqlite3",
    logic_adapters=["chatterbot.logic.BestMatch"]
)

# Streamlit UI
st.title("Health & Wellness Chatbot (MedQuAD Powered)")
user_input = st.text_input("Ask a health-related question:")

if user_input:
    sentiment = TextBlob(user_input).sentiment
    response = chatbot.get_response(user_input)

    if sentiment.polarity < -0.5:
        st.write("I'm sorry you're feeling down. Here's what I found:")

    st.write(str(response))
