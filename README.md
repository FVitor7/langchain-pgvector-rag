# 📄 RAG - Powered PDF Assistant

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-Framework-black)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Vector%20Database-blue)
![pgVector](https://img.shields.io/badge/pgVector-Enabled-green)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED)
![License](https://img.shields.io/badge/License-MIT-yellow)

Semantic search system powered by **LangChain, PostgreSQL (pgVector) and Gemini**.

This project implements a complete **Retrieval-Augmented Generation (RAG)** pipeline capable of ingesting a PDF document, storing embeddings in PostgreSQL with pgVector, and answering questions strictly based on the document content via CLI.

🎓 **This repository contains an academic project developed during my MBA in Software Engineering with AI at Full Cycle.**


## 🎯 Objective

Build a system capable of:

* **Ingestion:** Read a PDF file and store its content as vector embeddings in PostgreSQL using pgVector.
* **Search:** Allow users to ask questions via CLI and receive answers based only on the document content.

If the answer is not explicitly present in the document, the system responds:

> `"I do not have enough information to answer your question."`

---

# 🏗 System Architecture

```mermaid
flowchart TD

    A[PDF Document] --> B[PyPDFLoader]
    B --> C[RecursiveCharacterTextSplitter]
    C --> D[Generate Embeddings]
    D --> E[(PostgreSQL + pgVector)]

    F[User Question CLI] --> G[Question Embedding]
    G --> H[Similarity Search k=10]
    H --> E
    H --> I[Retrieve Top Chunks]
    I --> J[Prompt Template]
    J --> K[LLM Gemini]
    K --> L[Answer Returned to CLI]
```

---

# 🧠 RAG Flow Explained

1. PDF is loaded
2. Text is split into chunks (1000 chars, 150 overlap)
3. Each chunk is converted into embeddings
4. Vectors are stored in PostgreSQL (pgVector)
5. User question is embedded
6. Top 10 similar chunks are retrieved
7. Prompt enforces strict context-based response
8. LLM generates final answer

---

# 🧰 Tech Stack

* **Language:** Python
* **Framework:** LangChain
* **Database:** PostgreSQL + pgVector
* **Containerization:** Docker & Docker Compose
* **Embeddings Models:**

  * Gemini → `gemini-embedding-001`
* **LLM Models:**

  * Gemini → `gemini-2.5-flash-lite`

---

# 📂 Project Structure

```
├── data/pdf
│   ├── document.pdf
├── src/
│   ├── chat.py
│   ├── ingest.py
│   ├── llm_manager.py
│   ├── search.py
│   └── utils.py
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md

```

---

# ⚙️ Setup Instructions

## 1️⃣ Clone Repository

```bash
git clone <your-repo-url>
cd <repo-name>
```

---

## 2️⃣ Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Configure Environment Variables

Create `.env` based on `.env.example`

Example:

```
GOOGLE_API_KEY=your_google_key
GOOGLE_EMBEDDING_MODEL=models/embedding-001
GOOGLE_CHAT_MODEL='gemini-2.5-flash-lite'
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag
PG_VECTOR_COLLECTION_NAME=company_revenue_rag
PDF_PATH=./data/pdf/document.pdf
```

---

# 🐳 Run PostgreSQL + pgVector

```bash
docker compose up -d
```

---

# 🚀 Run PDF Ingestion

```bash
python src/ingest.py
```

✔ Loads PDF
✔ Splits into chunks (1000 / 150 overlap)
✔ Generates embeddings
✔ Stores vectors in PostgreSQL

---

# 💬 Start CLI Chat

```bash
python src/chat.py
```

Example:

```
QUESTION: What is the revenue of the company Beta IA LTDA??

ANSWER: R$ 40.733.987,34
```

Out-of-context example:

```
QUESTION: What is the capital of France?

ANSWER: I do not have enough information to answer your question.
```

---

# 🔐 Prompt Safety Enforcement

The system strictly:

* Uses only retrieved context
* Rejects external knowledge
* Prevents hallucinations
* Avoids opinion-based answers
* Returns fixed fallback response when necessary

---

# 🧪 Requirements Fulfilled

✔ Chunk size = 1000
✔ Overlap = 150
✔ Similarity search (k=10)
✔ PostgreSQL + pgVector storage
✔ CLI interaction
✔ Strict prompt enforcement
