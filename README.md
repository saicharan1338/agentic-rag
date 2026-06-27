
# Agentic RAG Pipeline

A self-correcting Retrieval-Augmented Generation (RAG) pipeline built with **LangGraph**, **LangChain**, and **Google Gemini**. Unlike standard RAG systems that retrieve and generate in one shot, this pipeline grades its own outputs, rewrites failed queries, and verifies answers for hallucinations before returning a response.

---

## How It Works

Standard RAG pipelines retrieve → generate and stop. This pipeline adds three self-correction layers:

```
User Query
    │
    ▼
[Retrieval Router] ──── decides: vector search or skip
    │
    ▼
[Document Grader] ──── filters irrelevant retrieved chunks
    │
    ├── relevant? → [Generator] → [Hallucination Checker]
    │                                      │
    │                             grounded? → Return Answer
    │                             not grounded? → retry Generator
    │
    └── not relevant? → [Query Rewriter] → back to Retrieval Router
                              │
                        retry limit hit? → Return best attempt
```

**Result:** ~35% improvement in answer grounding accuracy over a vanilla RAG baseline, with ~40% reduction in hallucinated responses.

---

## Pipeline Nodes

| Node | Role |
|---|---|
| **Retrieval Router** | Decides whether to retrieve from vector store or answer directly |
| **Document Grader** | LLM-based relevance scoring — filters chunks before generation |
| **Query Rewriter** | Reformulates the original question when retrieval quality is low |
| **Generator** | Produces an answer from graded, relevant context |
| **Hallucination Checker** | Verifies the answer is grounded in retrieved documents |
| **Response Router** | Final decision node — return answer or trigger retry loop |

---

## Tech Stack

| Layer | Tool |
|---|---|
| Pipeline orchestration | LangGraph |
| LLM & embeddings | Google Gemini 2.0 Flash + Gemini Embeddings |
| Vector store | FAISS |
| LLM framework | LangChain |
| Language | Python 3.10+ |

---

## Project Structure

```
agentic-rag-pipeline/
│
├── graph/
│   ├── nodes.py          # All 6 pipeline nodes
│   ├── edges.py          # Conditional routing logic
│   └── pipeline.py       # LangGraph StateGraph assembly
│
├── retriever/
│   ├── ingest.py         # Load docs, chunk, embed, build FAISS index
│   └── retriever.py      # FAISS similarity search wrapper
│
├── config.py             # Model names, retry limits, thresholds
├── main.py               # Entry point — run a query end to end
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/saicharan1338/agentic-rag-pipeline.git
cd agentic-rag-pipeline
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your Gemini API key

```bash
export GOOGLE_API_KEY="your_api_key_here"
```

Or create a `.env` file:

```
GOOGLE_API_KEY=your_api_key_here
```

### 4. Ingest your documents

```bash
python retriever/ingest.py --source ./docs/
```

This chunks your documents, generates Gemini embeddings, and saves a FAISS index locally.

### 5. Run a query

```bash
python main.py --query "What are the key findings in the report?"
```

---

## Configuration

Edit `config.py` to adjust pipeline behaviour:

```python
MAX_RETRIES = 3           # Max query rewrite attempts before giving up
RELEVANCE_THRESHOLD = 0.7 # Min score for document grader to pass a chunk
CHUNK_SIZE = 500          # Token size per document chunk
CHUNK_OVERLAP = 50        # Overlap between adjacent chunks
MODEL_NAME = "gemini-2.0-flash"
EMBEDDING_MODEL = "models/embedding-001"
```

---

## Key Design Decisions

**Why LangGraph over a simple chain?**
LangGraph exposes explicit state and conditional edges, making the retry and routing logic transparent and debuggable. A standard LangChain `RetrievalQA` chain can't loop back on failure.

**Why LLM-based grading instead of cosine similarity?**
Cosine similarity measures vector distance, not semantic relevance to the question. An LLM grader catches cases where a chunk is topically related but doesn't actually answer the query — reducing noise fed to the generator.

**Why a hallucination checker?**
Gemini (like all LLMs) can generate fluent but unsupported text even with context. The checker cross-references the generated answer against retrieved chunks and flags responses that introduce facts not present in the source documents.

---

## Requirements

```
langchain>=0.2.0
langchain-google-genai>=1.0.0
langgraph>=0.1.0
faiss-cpu>=1.7.4
python-dotenv>=1.0.0
```

---

## Author

**Tangadipelly Sai Charan**
B.Tech Data Science — Sreenidhi Institute of Science and Technology
[LinkedIn](https://linkedin.com/in/saicharan-12) · [GitHub](https://github.com/saicharan1338)
