import pandas as pd
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load QA dataset
df = pd.read_csv("medquad.csv")
questions = df['question'].fillna('').astype(str).tolist()
answers = df['answer'].fillna('').astype(str).tolist()

# Preprocess questions
stop_words = set(stopwords.words('english'))
def preprocess(text):
    tokens = word_tokenize(text.lower())
    return ' '.join([word for word in tokens if word.isalpha() and word not in stop_words])

preprocessed_questions = [preprocess(q) for q in questions]

# Vectorize questions
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(preprocessed_questions)

# Streamlit UI
st.title("Health Q&A Chatbot (MedQuAD)")
user_input = st.text_input("Ask a health-related question:")

if user_input:
    processed = preprocess(user_input)
    input_vec = vectorizer.transform([processed])
    sims = cosine_similarity(input_vec, X)
    best_idx = sims.argmax()
    best_match = questions[best_idx]
    best_answer = answers[best_idx]

    st.markdown(f"**Closest Match:** {best_match}")
    st.write(f"**Answer:** {best_answer}")
