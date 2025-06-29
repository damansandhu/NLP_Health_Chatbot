import os
import pandas as pd
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util

# Point NLTK to the custom nltk_data folder
nltk_data_path = os.path.join(os.path.dirname(__file__), "nltk_data")
nltk.data.path.append(nltk_data_path)

# Load dataset
df = pd.read_csv("medquad.csv")
questions = df['question'].fillna('').astype(str).tolist()
answers = df['answer'].fillna('').astype(str).tolist()

# Preprocess text
stop_words = set(stopwords.words('english'))
def preprocess(text):
    tokens = word_tokenize(text.lower())
    return ' '.join([word for word in tokens if word.isalpha() and word not in stop_words])

# Filter out short/empty questions
filtered_data = [(q, a) for q, a in zip(questions, answers) if len(str(q).strip()) > 10 and len(str(a).strip()) > 10]
filtered_questions, filtered_answers = zip(*filtered_data)

preprocessed_questions = [preprocess(q) for q in filtered_questions]

# Load sentence-transformers model (stronger model)
model = SentenceTransformer('all-mpnet-base-v2')
question_embeddings = model.encode(preprocessed_questions, convert_to_tensor=True)

# Streamlit UI
st.title("Health Q&A Chatbot (MedQuAD)")
user_input = st.text_input("Ask a health-related question:")

if user_input:
    processed_input = preprocess(user_input)
    input_embedding = model.encode(processed_input, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(input_embedding, question_embeddings)
    
    best_score = cos_scores.max().item()
    best_idx = cos_scores.argmax()

    # Optional: use a threshold to reject weak matches
    threshold = 0.45
    if best_score >= threshold:
        best_question = filtered_questions[best_idx]
        best_answer = filtered_answers[best_idx]
        st.markdown(f"**Closest Match:** {best_question}")
        st.write(f"**Answer:** {best_answer}")
    else:
        st.warning("Sorry, I couldn’t find a good answer to your question.")

