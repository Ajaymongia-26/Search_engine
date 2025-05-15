# Search_engine
# 🔍 Flipkart Laptop Semantic Search Engine

This is a **semantic search engine** built using **Streamlit**, **FAISS**, and **Sentence Transformers** to help users find the most relevant laptops based on natural language queries. It uses real Flipkart laptop data and provides top 10 most relevant laptops by leveraging powerful text embeddings and vector similarity search.

---

## 🧠 Features

- 🔎 **Natural Language Search**: Enter queries like "budget laptop with long battery life".
- ⚡ **Fast FAISS Search**: Uses FAISS for efficient similarity search over laptop reviews and specs.
- 🤖 **Pretrained SentenceTransformer**: Generates semantic embeddings using `all-MiniLM-L6-v2`.
- 🖼️ **Clean UI with Streamlit**: Easy-to-use interface with query suggestions and rich laptop info.
- 🔋 **Synthetic Battery Life**: Adds random battery life info for more complete laptop details.

---

## 🗃️ Dataset

A cleaned version of Flipkart's laptop dataset with fields like:

- `product_name`
- `overall_rating`
- `Price`
- `Screen Resolution`
- `Warranty Summary`
- `review`
- `title`
- `Battery Life` (synthetically added)

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit** (for UI)
- **Pandas / NumPy** (for data wrangling)
- **SentenceTransformers** (for embeddings)
- **FAISS** (for fast vector search)
- **NLTK** (for stopword removal)

---
