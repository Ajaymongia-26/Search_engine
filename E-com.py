# =========================== 1. IMPORT THE NECESSARY LIBRARIES ===========================
import pandas as pd
import numpy as np
import os
import faiss
import pickle
import base64
import random
import re
import json
import redis
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from datetime import datetime, date, timedelta
import streamlit as st

# =========================== 2. LOAD AND CLEAN THE DATA ===========================
# Load data
laptop_data = pd.read_csv('/Users/m1/Downloads/lp____________________________data.csv')

# Add random battery life
battery_life = ['10 hours', '8 hours', '12 hours', '6 hours', '5 hours', '7 hours', '9 hours', '11 hours', '4 hours']
laptop_data['Battery Life'] = [random.choice(battery_life) for _ in range(len(laptop_data))]

# Save cleaned data temporarily
laptop_data.to_csv('/Volumes/Disk_2/PRACTICE OF AIML COURSE/E_commmerce project/lp_data.csv', index=False)

# Reload with error handling
laptop_data = pd.read_csv('/Volumes/Disk_2/PRACTICE OF AIML COURSE/E_commmerce project/lp_data.csv', on_bad_lines='skip')
laptop_data = laptop_data.dropna().drop_duplicates()

# Clean text columns
laptop_data['Price'] = laptop_data['Price'].astype(str).str.replace('?', 'Rs. ', regex=False)
laptop_data['review'] = laptop_data['review'].astype(str).str.replace('ð  ', '', regex=False)
laptop_data['Warranty Summary'] = laptop_data['Warranty Summary'].astype(str).apply(lambda x: x.encode('ascii', errors='ignore').decode())
laptop_data['combined'] = laptop_data.apply(lambda x: ' '.join(x.astype(str)), axis=1)

# Remove HTML tags and stopwords
def remove_tags(text):
    return re.sub(re.compile('<.*?>'), '', text)

laptop_data['combined'] = laptop_data['combined'].apply(remove_tags).str.lower()
Stop_w_list = stopwords.words('english')
laptop_data['combined'] = laptop_data['combined'].apply(lambda x: ' '.join([item for item in x.split() if item not in Stop_w_list]))

# =========================== 3. EMBEDDING THE DATA ===========================
model = SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def build_index(documents):
    embeddings = model.encode(documents, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

index, embeddings = build_index(laptop_data["combined"].tolist())

# =========================== 4. USER INTERFACE ===========================
st.markdown("""
<div style='background-color:#f9f9f9; padding:15px; border-radius:10px'>
    <h2 style='color:#333333;'>LAPTOP RECOMMENDATION CHATBOT</h2>
    <p style='font-size:16px;'>Ask your query like a Customer :</p>
    <ul>
        <li> Best gaming laptops under 60k</li>
        <li> Light-weight laptop with long battery life</li>
        <li> HP, DELL, APPLE, ASUS, LENOVO, MICROSOFT, SAMSUNG, MSI laptops with good reviews</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    .block-container {
        text-align: left;
        max-width: 80%;
        margin: auto;
    }
</style>
""", unsafe_allow_html=True)

# =========================== 5. USER INPUT SECTION ===========================
user_query = st.text_input("**💬 TYPE YOUR QUERY BELOW :**", "")

st.markdown("""
<style>
    .stTextInput > div > div > input {
        border-radius: 25px;
        padding: 15px 20px;
        font-size: 18px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# =========================== 6. SEARCH THE QUERY BY USER INPUT ===========================
# Connect to Redis
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
CHAT_HISTORY_KEY = "chat_history"

# Session state initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if 'selected_query' not in st.session_state:
    st.session_state.selected_query = None
    st.session_state.selected_results = []

def save_to_redis(query, results):
    entry = {
        "query": query,
        "results": results,
        "timestamp": datetime.now().isoformat()
    }
    redis_client.rpush(CHAT_HISTORY_KEY, json.dumps(entry))

def load_from_redis():
    data = redis_client.lrange(CHAT_HISTORY_KEY, 0, -1)
    return [json.loads(entry) for entry in data]

if st.button("🔍 Search"):
    if user_query.strip():
        query_embedding = model.encode([user_query])
        D, I = index.search(query_embedding, k=10)

        result_texts = []
        for idx in I[0]:
            result = laptop_data.iloc[idx]
            result_markdown = f"""
**🖥 Product Name:** {result['product_name']}  
**⭐ Overall Rating:** {result['overall_rating']}  
**💰 Price:** {result.get('Price', 'Not Available')}  
**🔋 Battery Life:** {result.get('Battery Life', 'Not Available')}  
**💻 Screen Resolution:** {result.get('Screen Resolution', 'Not Available')}  
**🗞️ Warranty Summary:** {result.get('Warranty Summary', 'Not Available')}  
**💬 Review:** {result.get('review', 'No review')}  
**📝 Title:** {result.get('title', 'No title')}  

---
"""
            result_texts.append(result_markdown)

        # =========================== 7. SHOW THE BEST TOP 10 LAPTOP ===========================
        save_to_redis(user_query, result_texts)
        st.session_state.chat_history.append({
            "query": user_query,
            "results": result_texts,
            "timestamp": datetime.now().isoformat()
        })

        st.subheader("📋 Best top 10 laptop :")
        for text in result_texts:
            st.markdown(text)

# =========================== 8. MAKE THE HISTORY OF SEARCHES QUERY ===========================
with st.sidebar:
    st.markdown("## 🕓 Chat History")
    redis_history = load_from_redis()
    history = st.session_state.get("chat_history", [])
    grouped_history = {}

    for entry in redis_history:
        timestamp = datetime.fromisoformat(entry["timestamp"])
        entry_date = timestamp.date()

        if entry_date == date.today():
            label = "Today"
        elif entry_date == date.today() - timedelta(days=1):
            label = "Yesterday"
        else:
            label = timestamp.strftime("%d %b %Y")

        grouped_history.setdefault(label, []).append(entry)

    # =========================== 9. SAVE THE HISTORY ONE BY ONE IN SIDEBAR ===========================
    for group, entries in grouped_history.items():
        st.markdown(f"### {group}")
        for i, entry in enumerate(entries[::-1]):
            full_query = entry["query"]
            short_query = ' '.join(full_query.split()[:4])

            if st.button(f"{short_query}", key=f"history-{group}-{i}"):
                st.session_state.selected_query = full_query
                st.session_state.selected_results = entry["results"]

    # =========================== 10. CLEAR THE CHAT HISTORY ===========================
    if st.button("Clear Chat History"):
        redis_client.delete(CHAT_HISTORY_KEY)
        st.session_state.chat_history = []
        st.session_state.selected_query = None
        st.session_state.selected_results = []
        st.success("Chat history cleared!")

# Show previous query results if selected
if st.session_state.selected_query:
    st.subheader("📄 Previous Query Result")
    st.markdown(f"**Query:** {st.session_state.selected_query}")
    for res in st.session_state.selected_results:
        st.markdown(res)
