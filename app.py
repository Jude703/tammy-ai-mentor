import streamlit as st
import json
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download("vader_lexicon")

client = OpenAI()
sia = SentimentIntensityAnalyzer()

def load_embeddings(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def cosine_similarity_np(vec1, vec2):
    vec1 = np.array(vec1).reshape(1, -1)
    vec2 = np.array(vec2).reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]

def get_top_chunks(user_embedding, all_chunks, top_k=15):
    scored_chunks = []
    for chunk in all_chunks:
        similarity = cosine_similarity_np(user_embedding, chunk["embedding"])
        scored_chunks.append((similarity, chunk))
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored_chunks[:top_k]]

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def analyze_sentiment(text):
    score = sia.polarity_scores(text)
    if score["compound"] >= 0.05:
        return "positive"
    elif score["compound"] <= -0.05:
        return "negative"
    else:
        return "neutral"

def load_persona(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def generate_response(persona_prompt, retrieved_chunks, user_input, sentiment):
    context = "\n\n".join([chunk["text"] for chunk in retrieved_chunks])
    full_prompt = f"""{persona_prompt}

# Context:
{context}

# User message (sentiment: {sentiment}):
{user_input}

# Tammy's response:"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are Tammy, an empathetic and insightful AI mentor."},
            {"role": "user", "content": full_prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

st.set_page_config(page_title="Tammy - AI Mentor", layout="wide")
st.title("💡 Tammy — Your AI Mentor for Business Growth")

user_input = st.chat_input("Ask Tammy anything...")
if user_input:
    with st.spinner("Thinking..."):
        persona_prompt = load_persona("tammy-ai-mentor/persona/tammy_persona.txt")
        all_chunks = load_embeddings("tammy-ai-mentor/data/combined_embeddings.json")
        user_embedding = get_embedding(user_input)
        top_chunks = get_top_chunks(user_embedding, all_chunks, top_k=15)
        sentiment = analyze_sentiment(user_input)
        response = generate_response(persona_prompt, top_chunks, user_input, sentiment)

    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        st.markdown(response)

