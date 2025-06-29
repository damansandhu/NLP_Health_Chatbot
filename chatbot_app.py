import os
import pandas as pd
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
from nltk.data import load

# Define the path to your custom nltk_data folder
nltk_data_path = os.path.join(os.path.dirname(__file__), "nltk_data")
nltk.data.path.append(nltk_data_path)

# Explicitly load English stopwords and punkt tokenizer from your custom path
stopwords_path = os.path.join(nltk_data_path, "corpora", "stopwords")
stop_words = set(stopwords.words('english'))

punkt_path = os.path.join(nltk_data_path, "tokenizers", "punkt", "english.pickle")
tokenizer = load(punkt_path)

# Load dataset
df = pd.read_csv("medquad.csv")
questions = df['question'].fillna('').astype(str).tolist()
answers = df['answer'].fillna('').astype(str).tolist()

# Preprocess text
def preprocess(text):
    tokens = tokenizer.tokenize(text.lower())  # using the loaded tokenizer
    return ' '.join([word for word in tokens if word.isalpha() and word not in stop_words])

preprocessed_questions = [preprocess(q) for q in questions]

# Load sentence-transformers model
model = SentenceTransformer('all-MiniLM-L6-v2')
question_embeddings = model.encode(preprocessed_questions, convert_to_tensor=True)

# Streamlit UI
st.title("Health Q&A Chatbot (MedQuAD)")
user_input = st.text_input("Ask a health-related question:")

if user_input:
    processed_input = preprocess(user_input)
    input_embedding = model.encode(processed_input, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(input_embedding, question_embeddings)
    best_idx = cos_scores.argmax()

    best_question = questions[best_idx]
    best_answer = answers[best_idx]

    st.markdown(f"**Closest Match:** {best_question}")
    st.write(f"**Answer:** {best_answer}")
