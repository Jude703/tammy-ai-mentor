import streamlit as st
import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

client = OpenAI()

st.set_page_config(page_title="Tammy - AI Mentor", page_icon="🤖")
st.title("Tammy: Your AI Mentor")

if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_embeddings = st.file_uploader("Upload your embeddings JSON file", type=["json"])

if uploaded_embeddings:
    embeddings_data = json.load(uploaded_embeddings)
    texts = [item["text"] for item in embeddings_data]
    vectors = [item["embedding"] for item in embeddings_data]

    st.success("Embeddings loaded successfully.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask Tammy anything")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        sentiment = sia.polarity_scores(user_input)["compound"]
        mood = "curious" if sentiment > 0.05 else "neutral" if sentiment > -0.05 else "concerned"

        with st.spinner("Thinking..."):
            user_embedding = client.embeddings.create(
                input=user_input,
                model="text-embedding-3-small"
            ).data[0].embedding

            similarities = cosine_similarity([user_embedding], vectors)[0]
            top_indices = np.argsort(similarities)[-5:][::-1]
            top_chunks = [texts[i] for i in top_indices]

            persona = (
                "You are Tammy, an AI mentor for business. You are sharp, empathetic, and always help the user gain clarity. "
                "Your tone is confident, warm, and honest. If a question is vague or seems exploratory, respond with curiosity. "
                "If something is missing from your knowledge base, acknowledge it honestly and invite the user to clarify what they seek. "
                "You speak with insight and encouragement, helping people uncover what really matters and move forward with clarity. "
                f"The user's current mood appears to be {mood}. Respond accordingly."
            )

            system_msg = {"role": "system", "content": persona}
            history_msgs = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-5:]]
            context_msg = {"role": "user", "content": "Here are relevant excerpts from the knowledge base:\n\n" + "\n\n".join(top_chunks)}
            final_question = {"role": "user", "content": user_input}

            full_prompt = [system_msg] + history_msgs + [context_msg, final_question]

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=full_prompt,
                temperature=0.7
            )

            assistant_reply = response.choices[0].message.content.strip()

        with st.chat_message("assistant"):
            st.markdown(assistant_reply)
        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
