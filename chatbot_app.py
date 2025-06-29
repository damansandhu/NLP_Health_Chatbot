import pandas as pd
import nltk
import spacy
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from textblob import TextBlob
import streamlit as st
from tqdm import tqdm

# Download NLTK & spaCy resources if missing
nltk.download('punkt')
nltk.download('stopwords')
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Load MedQuAD dataset
df = pd.read_csv(r"C:\Users\damanbirs\OneDrive - WESTMED AMBULANCE\Desktop\NLP\medquad.csv")

# Flatten Q&A into a list for training
qa_flat = []
for q, a in zip(df['question'].astype(str), df['answer'].astype(str)):
    if pd.notna(q) and pd.notna(a):
        qa_flat.extend([q.strip(), a.strip()])

# Prepare pairs for training (2-line conversations)
conversations = [qa_flat[i:i+2] for i in range(0, len(qa_flat)-1, 2)]

# Initialize chatbot
chatbot = ChatBot(
    'HealthBot',
    logic_adapters=['chatterbot.logic.BestMatch'],
    storage_adapter='chatterbot.storage.SQLStorageAdapter',
    database_uri='sqlite:///database.sqlite3'
)

# Train chatbot with tqdm progress bar
trainer = ListTrainer(chatbot)
st.info("Training the chatbot... please wait ⏳ (only runs the first time)")

for convo in tqdm(conversations, desc="Training chatbot", unit="pair"):
    try:
        trainer.train(convo)
    except Exception:
        continue  # Skip any broken pairs

# Streamlit interface
st.title("💬 Health & Wellness Chatbot (Powered by MedQuAD)")
user_input = st.text_input("Ask your health-related question:")

if user_input:
    # Clean input using NLTK
    tokens = nltk.word_tokenize(user_input.lower())
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    processed_input = ' '.join(tokens)

    # Sentiment analysis
    sentiment = TextBlob(user_input).sentiment
    response = chatbot.get_response(processed_input)

    # Show empathetic message if user is frustrated
    if sentiment.polarity < -0.5:
        st.write("😔 I'm sorry you're feeling this way. Here's what I found:")

    # Show the chatbot response
    st.write(str(response))
