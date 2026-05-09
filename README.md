# 🩺 AI Medical Assistant using RAG

## 📌 Project Overview

AI Medical Assistant using Retrieval-Augmented Generation (RAG) is an AI-powered application that allows users to upload medical PDF documents and ask questions based on the uploaded content.

The system retrieves relevant information from documents using semantic search and generates intelligent answers using Natural Language Processing techniques.

This project demonstrates the implementation of modern Generative AI concepts such as:

* Retrieval-Augmented Generation (RAG)
* Semantic Search
* Vector Databases
* Embeddings
* Document Question Answering

---

# 🚀 Features

✅ Upload multiple medical PDF files
✅ Extract text from documents
✅ Semantic search using FAISS
✅ AI-generated answers
✅ Context-aware question answering
✅ Interactive Streamlit UI
✅ Retrieval-based response generation

---

# 🛠 Technologies Used

* Python
* Streamlit
* LangChain
* FAISS Vector Database
* HuggingFace Embeddings
* Transformers
* PyPDF2

---

# 🧠 How It Works

1. User uploads medical PDF documents
2. Text is extracted from PDFs
3. Large text is divided into smaller chunks
4. Embeddings are generated from text chunks
5. Embeddings are stored in FAISS vector database
6. User asks a medical question
7. Relevant chunks are retrieved using semantic similarity
8. AI model generates the final answer

---

# 📂 Project Structure

```plaintext
medical-rag-project/
│
├── app.py
├── requirements.txt
├── README.md
└── images/
```

---

# ▶️ Run the Project

## Step 1 — Install Required Libraries

```bash
pip install -r requirements.txt
```

## Step 2 — Run Application

```bash
python -m streamlit run app.py
```

---

# 📸 Project Screenshots

## Home Page

![Home](images/screenshot1.png)

## PDF Upload

![Upload](images/screenshot2.png)

## AI Generated Answer

![Answer](images/screenshot3.png)

---

# 📌 Sample Questions

* What are symptoms of diabetes?
* Explain type 2 diabetes.
* What causes hypertension?
* Summarize the uploaded document.
* What are complications of diabetes?

---

# 🎯 Advantages

* Reduces hallucination compared to normal chatbots
* Provides context-aware responses
* Supports semantic document retrieval
* Easy to use interactive interface
* Useful for medical knowledge assistance

---

# 🔮 Future Enhancements

* Voice input support
* Chat history
* Multi-language support
* DOCX and TXT file support
* Cloud deployment
* Authentication system

--
