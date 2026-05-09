import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI Medical Assistant using RAG",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown(
    """
    <style>

    .stApp {
        background-color: #f5f7fa;
    }

    h1 {
        color: #0f172a;
        text-align: center;
        font-size: 42px;
    }

    .main-title {
        text-align: center;
        color: #0e7490;
        font-size: 20px;
        margin-bottom: 20px;
    }

    .stButton button {
        background-color: #0e7490;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
    }

    .answer-box {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
        margin-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.header("🩺 About Project")

    st.write("""
    This project is an AI-powered Medical Assistant
    using Retrieval-Augmented Generation (RAG).

    Features:
    - Upload medical PDFs
    - Ask medical questions
    - Semantic search
    - AI-generated answers
    - FAISS vector database
    """)

    st.subheader("⚙ Technologies Used")

    st.write("""
    - Python
    - Streamlit
    - LangChain
    - FAISS
    - HuggingFace
    - Transformers
    """)

# ---------------- TITLE ----------------

st.title("🩺 AI Medical Assistant")

st.markdown(
    '<p class="main-title">Retrieval-Augmented Generation (RAG) Based Medical Chatbot</p>',
    unsafe_allow_html=True
)

# ---------------- FILE UPLOAD ----------------

uploaded_files = st.file_uploader(
    "📄 Upload Medical PDF Files",
    type="pdf",
    accept_multiple_files=True
)

# ---------------- PDF PROCESSING ----------------

if uploaded_files:

    all_text = ""

    with st.spinner("📚 Reading PDF files..."):

        for pdf in uploaded_files:

            pdf_reader = PdfReader(pdf)

            for page in pdf_reader.pages:

                text = page.extract_text()

                if text:
                    all_text += text

    # ---------------- TEXT SPLITTING ----------------

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_text(all_text)

    # ---------------- EMBEDDINGS ----------------

    with st.spinner("🔍 Creating embeddings..."):

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # ---------------- VECTOR STORE ----------------

        vector_store = FAISS.from_texts(
            chunks,
            embedding=embeddings
        )

    st.success("✅ PDF Processing Completed!")

    # ---------------- QUESTION INPUT ----------------

    user_question = st.text_input(
        "💬 Ask a Medical Question"
    )

    # ---------------- QUESTION ANSWERING ----------------

    if user_question:

        with st.spinner("🤖 Generating Answer..."):

            # Retrieve similar chunks

            docs = vector_store.similarity_search(
                user_question
            )

            # Create context

            context = "\n".join(
                [doc.page_content for doc in docs]
            )

            # Load local model

            generator = pipeline(
                "text-generation",
                model="google/flan-t5-base"
            )

            # Prompt

            prompt = f"""
            Answer the medical question using the context below.

            Context:
            {context}

            Question:
            {user_question}
            """

            # Generate response

            response = generator(
                prompt,
                max_length=200,
                do_sample=False
            )

            answer = response[0]["generated_text"]

        # ---------------- DISPLAY ANSWER ----------------

        st.markdown(
            f"""
            <div class="answer-box">
            <h3>🧠 AI Generated Answer</h3>
            <p>{answer}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# ---------------- FOOTER ----------------

st.markdown("---")

st.markdown(
    """
    <center>
    AI Medical Assistant using RAG • Final Year Project
    </center>
    """,
    unsafe_allow_html=True
)