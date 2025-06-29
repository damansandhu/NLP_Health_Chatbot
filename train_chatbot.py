import pandas as pd
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
import os

# Path to your MedQuAD CSV file
csv_path = r"C:\Users\damanbirs\OneDrive - WESTMED AMBULANCE\Desktop\NLP\medquad.csv"

# Path where you want the database to be saved
db_path = r"C:\Users\damanbirs\OneDrive - WESTMED AMBULANCE\Desktop\NLP\database.sqlite3"

# Optional: Remove old database if it exists
if os.path.exists(db_path):
    os.remove(db_path)

# Load MedQuAD data
df = pd.read_csv(csv_path)
qa_pairs = []

for i, row in df.iterrows():
    question = str(row.get("question", "")).strip()
    answer = str(row.get("answer", "")).strip()
    if question and answer:
        qa_pairs.append(question)
        qa_pairs.append(answer)

# Initialize chatbot
chatbot = ChatBot(
    "HealthBot",
    storage_adapter="chatterbot.storage.SQLStorageAdapter",
    database_uri=f"sqlite:///{db_path}",
    logic_adapters=["chatterbot.logic.BestMatch"]
)

# Train chatbot
trainer = ListTrainer(chatbot)
trainer.train(qa_pairs)

print("✅ Training complete. Database saved at:", db_path)
