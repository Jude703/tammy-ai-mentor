import streamlit as st
import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from openai import OpenAI

nltk.download("vader_lexicon")


client = OpenAI()


st.set_page_config(page_title="Tammy — AI Mentor", layout="wide")


sentiment_analyzer = SentimentIntensityAnalyzer()


@st.cache_data
def load_embeddings_from_folder(folder_path):
    embeddings = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                data = json.load(f)
                embeddings.extend(data)
    return embeddings

def get_top_chunks(user_question, all_chunks, top_n=5):

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=user_question
    )
    question_vector = np.array(response.data[0].embedding).reshape(1, -1)

 
    chunk_vectors = np.array([chunk["embedding"] for chunk in all_chunks])
    similarities = cosine_similarity(question_vector, chunk_vectors)[0]


    top_indices = np.argsort(similarities)[::-1][:top_n]
    top_chunks = [all_chunks[i] for i in top_indices]

    return top_chunks

def analyze_sentiment(text):
    scores = sentiment_analyzer.polarity_scores(text)
    if scores["compound"] >= 0.5:
        return "positive"
    elif scores["compound"] <= -0.5:
        return "negative"
    else:
        return "neutral"

def generate_tammy_response(user_question, retrieved_chunks, sentiment_label):
    context = "\n---\n".join([chunk["text"] for chunk in retrieved_chunks])

    system_prompt = f"""
You are Tammy, an AI Mentor for business founders. You are clear, compassionate, strategic, and insightful. You always speak with warmth and honesty. Use the following context to answer. Prioritize clarity, emotional intelligence, and tactical value. Keep your answer grounded in the files of Tammy (Instructions, Toolkits, Integration, Check-in Templates). Sentiment of the user input: {sentiment_label}.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question},
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.7
    )

    return response.choices[0].message.content.strip()


st.title(" Tammy — AI Mentor")


tab1, tab2 = st.tabs([" conversation", "list of questions"])

with tab1:
    user_input = st.text_input("what is on your mind?", placeholder="how to use egg method in my business؟")
    if user_input:
        sentiment = analyze_sentiment(user_input)
        st.write(f"sentiment analysis `{sentiment}`")

        with st.spinner("Tammy thinks deeply"):
            chunks = load_embeddings_from_folder("cleaned_embeddings_new")
            top_chunks = get_top_chunks(user_input, chunks, top_n=5)
            response = generate_tammy_response(user_input, top_chunks, sentiment)

            st.markdown("Tammy responds:")
            st.write(response)

          
            if "history" not in st.session_state:
                st.session_state.history = []
            st.session_state.history.append({"q": user_input, "a": response})

with tab2:
    st.markdown("### questions list ")
    if "history" in st.session_state:
        for entry in st.session_state.history[::-1]:
            st.markdown(f"**Q:** {entry['q']}")
            st.markdown(f"**A:** {entry['a']}")
            st.markdown("---")
    else:
        st.info("no questions yet")


st.sidebar.markdown("future settings")
st.sidebar.checkbox("activate long term memory (Memory)", value=False, help="it will be activated in the future.")

