import streamlit as st
import os
import json
import numpy as np
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# تنزيل ملفات المشاعر
nltk.download("vader_lexicon")

# إعداد واجهة OpenAI
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# تحميل ملفات embeddings
st.title("🔍 AI Answer Generator with Smart Memory")

uploaded_file = st.file_uploader("Upload your embeddings JSON file", type="json")

if uploaded_file is not None:
    embeddings_data = json.load(uploaded_file)
    texts = [item["text"] for item in embeddings_data]
    vectors = np.array([item["embedding"] for item in embeddings_data])

    question = st.text_input("Ask your question:")

    if question:
        # توليد embedding للسؤال
        with st.spinner("Generating embedding and searching..."):
            question_embedding = client.embeddings.create(
                model="text-embedding-3-small",
                input=question
            ).data[0].embedding

            similarities = cosine_similarity(
                [question_embedding],
                vectors
            )[0]

            top_indices = similarities.argsort()[::-1][:5]
            top_chunks = [embeddings_data[i] for i in top_indices]

            # عرض أفضل 5 نتائج
            st.subheader("🔎 Top 5 relevant chunks:")
            for i, chunk in enumerate(top_chunks, 1):
                st.markdown(f"**{i}.** `{chunk.get('chunk_title', 'No title')}` — Score: {similarities[top_indices[i-1]]:.4f}")
                st.write(chunk["text"])
                st.markdown("---")

            # توليد الرد النهائي
            context = "\n\n".join([chunk["text"] for chunk in top_chunks])

            system_msg = "You are Tammy, a clarity-driven and kind AI mentor. Answer based only on the provided context below."

            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ]

            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages
            )

            final_answer = response.choices[0].message.content

            # تحليل المشاعر
            sia = SentimentIntensityAnalyzer()
            sentiment = sia.polarity_scores(final_answer)

            st.subheader("🧠 Tammy's Answer")
            st.write(final_answer)

            st.subheader("💬 Sentiment Analysis")
            st.json(sentiment)
