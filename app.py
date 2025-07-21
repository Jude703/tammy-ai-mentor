##
import streamlit as st
import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from openai import OpenAI

nltk.download("vader_lexicon")

client = OpenAI()

EMBEDDINGS_DIR = "cleaned_embeddings_new"
MEMORY_FILE = "tammy_memory.json"
PERSONA_FILE = "tammy_prompt.txt"

@st.cache_data
def load_all_chunks(directory):
    chunks = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        chunks.extend(data)
                except json.JSONDecodeError:
                    continue
    return chunks

def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def search_chunks(query, top_k=10):
    query_emb = get_embedding(query)
    chunk_embeddings = [chunk["embedding"] for chunk in all_chunks if "embedding" in chunk]
    similarities = cosine_similarity([query_emb], chunk_embeddings)[0]
    ranked = sorted(zip(similarities, all_chunks), key=lambda x: x[0], reverse=True)
    top_chunks = [entry[1] for entry in ranked[:top_k]]
    return top_chunks

def detect_tone(text):
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(text)["compound"]
    if score >= 0.5:
        return "positive"
    elif score <= -0.5:
        return "negative"
    else:
        return "neutral"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_memory(memory):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

def load_persona_prompt():
    with open(PERSONA_FILE, "r", encoding="utf-8") as f:
        return f.read().strip()

def generate_response(memory, persona, context_chunks, tone, new_question):
    context = "\n\n".join([chunk["chunk_content"] for chunk in context_chunks])
    history = "\n\n".join([f"User: {turn['question']}\nTammy: {turn['answer']}" for turn in memory])
    system_message = f"{persona}\n\nContext:\n{context}\n\nConversation History:\n{history}"

    tone_prefix = {
        "positive": "Great energy! Let's build on that.",
        "negative": "Thanks for trusting me. I'm here with you.",
        "neutral": "Let's tackle this together step by step."
    }

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"{tone_prefix[tone]}\n\nQuestion: {new_question}"}
    ]

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.4
    )

    return response.choices[0].message.content

all_chunks = load_all_chunks(EMBEDDINGS_DIR)
persona_prompt = load_persona_prompt()
memory_data = load_memory()

st.title("Tammy – Your AI Mentor")

user_question = st.text_input("What would you like to ask Tammy?")

if st.button("Ask Tammy"):
    if user_question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            tone = detect_tone(user_question)
            top_chunks = search_chunks(user_question)
            answer = generate_response(memory_data, persona_prompt, top_chunks, tone, user_question)
            memory_data.append({"question": user_question, "answer": answer})
            save_memory(memory_data)
        st.success("Tammy's Response:")
        st.write(answer)
