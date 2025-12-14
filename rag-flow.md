# Deep RAG Pipeline - Complete Workflow

---

## 🎯 Overview

**Multi-Agent Deep RAG for Financial Documents**

A complete pipeline for extracting, embedding, and retrieving information from financial SEC filings (10-K, 10-Q, 8-K reports) using multimodal content and advanced retrieval strategies.

---

## 📊 Pipeline Architecture

```
PDFs → Extract → Describe → Embed → Retrieve → Answer
  ↓       ↓         ↓         ↓        ↓
 Step 1  Step 2   Step 3   Step 4   Step 5
```

---

## Step 1️⃣: PDF Extraction with Docling

### Input
- Financial PDFs (10-K, 10-Q, 8-K reports)
- Organized in: `data/rag-data/pdfs/`

### Process
1. **Parse PDFs** using Docling converter
2. **Extract three content types:**
   - 📄 **Markdown**: Full document text with page breaks
   - 📊 **Tables**: With 2 paragraphs of context + page numbers
   - 🖼️ **Images**: Large charts/diagrams (>500x500 pixels)

### Output Structure
```
data/rag-data/
├── markdown/{company}/{document}.md
├── tables/{company}/{document}/table_X_page_Y.md
└── images/{company}/{document}/page_Y.png
```

### Key Features
- Page-level tracking for all content
- Contextual table extraction
- Smart image filtering (size-based)
- Metadata extraction from filenames

---

## Step 2️⃣: Image Description Generation

### Input
- Extracted images from Step 1
- Located in: `data/rag-data/images/`

### Process
1. **Load images** using PIL
2. **Encode to base64** for API transmission
3. **Generate descriptions** using Gemini 2.5 Flash Multimodal 
4. **Save as markdown** files

### AI Prompt Focus
- Chart/graph data trends and axis labels
- Table structures and key data points
- Text content summaries
- Visual layout descriptions

### Output
```
data/rag-data/images_desc/
└── {company}/{document}/page_Y.md
```

### Why This Step?
**Unified Embedding Space:** Convert visual content to text so everything can use the same embedding model (Approach 2: Text-only embeddings)

---

## Step 3️⃣: Vector Database Ingestion

### Input
- Markdown files (Step 1)
- Table files (Step 1)
- Image descriptions (Step 2)

### Process

#### 3.1 Initialize Components
```python
# Gemini Embeddings (Full dimensionality)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

# BM25 Sparse Embeddings
sparse_embeddings = FastEmbedSparse(
    model_name="Qdrant/bm25"
)

# Vector Store (Hybrid Mode)
vector_store = QdrantVectorStore(
    retrieval_mode=RetrievalMode.HYBRID
)
```

#### 3.2 Content Processing

**For Each Content Type:**

| Content Type | Source | Chunking Strategy |
|--------------|--------|-------------------|
| **Text** | Markdown files | Split by `<!-- page break -->` markers |
| **Tables** | Table MD files | Individual table + context (no splitting) |
| **Images** | Description MD files | Full description (no splitting) |

#### 3.3 Metadata Enrichment

**Extracted Metadata:**
- `company_name` (e.g., "amazon", "apple")
- `doc_type` (e.g., "10-k", "10-q", "8-k")
- `fiscal_year` (e.g., 2024)
- `fiscal_quarter` (e.g., "q3")
- `content_type` ("text", "table", "image")
- `page` (page number)
- `file_hash` (for deduplication)

### Output
**Single Qdrant Collection**
- **Hybrid search enabled** (dense + sparse)
- **Rich metadata** for filtering
- **All content types unified** in one collection

---

## Step 4️⃣: Advanced Retrieval

### Retrieval Pipeline

#### 4.1 Filter Extraction with LLM

**Natural Language → Structured Filters**

```python
User Query: "Amazon Q3 2024 revenue"
    ↓
LLM Extraction
    ↓
Filters: {
    "company_name": "amazon",
    "doc_type": "10-q",
    "fiscal_year": 2024,
    "fiscal_quarter": "q3"
}
```

**Gemini 2.5 Flash** extracts structured metadata from conversational queries

#### 4.2 Hybrid Search

**Dense + Sparse Retrieval**

```
Query: "What is Apple's revenue?"
    ↓
┌─────────────────┬─────────────────┐
│  Dense Search   │  Sparse Search  │
│  (Semantic)     │  (Keyword)      │
│  Gemini-001     │  BM25           │
└────────┬────────┴────────┬────────┘
         │                 │
         └─────┬───────────┘
               ↓
       Reciprocal Rank
           Fusion
               ↓
         Top K Results
```

**Why Hybrid?**
- **Dense**: Understands semantic meaning
- **Sparse**: Matches exact keywords
- **Combined**: Best of both worlds

#### 4.3 Reranking

**Cross-Encoder Reranking**

```
Initial Results (k=10)
    ↓
BAAI/bge-reranker-base
(Cross-Encoder)
    ↓
Reranked Results (top_k=5)
(Sorted by relevance score)
```

**Purpose:** Deep interaction between query and documents for precise ranking

---

## Step 5️⃣: Complete Retrieval Flow

### Function: `retrieve_with_reranking()`

```
User Query
    ↓
1. Extract Filters (LLM)
    ↓
2. Hybrid Search (Dense + Sparse)
   • Apply metadata filters
   • Fetch top K candidates
    ↓
3. Rerank (Cross-Encoder)
   • Score query-document pairs
   • Return top N results
    ↓
Final Results
```

---

## 🔑 Key Design Decisions

### 1. Unified Text Embeddings (Approach 2)
**Decision:** Use text embeddings for ALL content types
- ✅ Single embedding model (Gemini-001)
- ✅ Unified search across all types
- ✅ Simple architecture
- ✅ Cost-effective

**Alternative Rejected:**
- ❌ Multimodal embeddings (API issues, complexity)

### 2. Hybrid Search
**Dense + Sparse = Better Results**
- Dense: "revenue growth" → finds "increased sales"
- Sparse: "Q3 2024" → exact match
- Together: Comprehensive retrieval

### 3. LangChain Abstractions
**Simplified Code:**
```python
# Before (Raw Qdrant)
embedding = embeddings.embed_query(text)
sparse = sparse_embeddings.embed_query(text)
qdrant_client.upsert(points=[...])

# After (LangChain)
doc = Document(page_content=text, metadata=metadata)
vector_store.add_documents([doc])
```

### 4. Metadata-Driven Filtering
**Structured Filters > Text Search**
- Company, year, quarter → precise filtering
- LLM extracts filters from natural language
- Reduces search space before semantic retrieval

---

## 📈 Performance Characteristics

### Scalability
- Handles documents of any size
- Automatic chunking by pages
- Deduplication prevents redundancy
- Incremental ingestion supported

### Retrieval Speed
- **Hybrid Search:** Fast vector + keyword matching
- **Reranking:** Batch processing for efficiency
- **End-to-end:** Sub-second response times

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **PDF Extraction** | Docling | Convert PDFs to structured content |
| **Vision** | Gemini 2.5 Flash | Generate image descriptions |
| **Embeddings** | Gemini Embedding 001 | Dense semantic vectors (full dimensionality) |
| **Sparse** | FastEmbed BM25 | Keyword matching |
| **Vector DB** | Qdrant | Hybrid search storage |
| **Framework** | LangChain | Abstraction layer |
| **Reranker** | BAAI/bge-reranker-base | Cross-encoder reranking |
| **LLM** | Gemini 2.5 Flash | Filter extraction, Q&A |

---

## 🎓 Learning Outcomes

### For Students
1. **End-to-end RAG pipeline** from PDFs to answers
2. **Multimodal content handling** (text, tables, images)
3. **Advanced retrieval strategies** (hybrid + reranking)
4. **Production-ready patterns** (deduplication, metadata)
5. **LangChain best practices** for clean code

---

## 🚀 Future Enhancements

### Potential Improvements
1. **Multi-query retrieval** - Generate multiple query variations
2. **Contextual compression** - Filter irrelevant context
3. **Parent-child retrieval** - Link chunks to full documents
4. **Graph-based RAG** - Entity relationships
5. **Streaming responses** - Real-time answer generation

---

## 📝 Summary

### Progressive Flow
```
PDFs (unstructured)
    ↓
Extraction (structured: text + tables + images)
    ↓
Vision AI (images → text descriptions)
    ↓
Embeddings (all text → vectors)
    ↓
Vector DB (hybrid storage)
    ↓
Retrieval (filters + hybrid search + reranking)
    ↓
Answers (precise, relevant, sourced)
```

### Core Principle
**"Everything is Text, Everything is Searchable"**

By converting all modalities to text and using unified embeddings with hybrid search, we create a simple yet powerful RAG system that works reliably at scale.

---

## 📚 Pipeline Sequence

| Order | Stage | Required? | Output |
|-------|-------|-----------|--------|
| 1️⃣ | Data Extraction | ✅ Yes | Markdown, Tables, Images |
| 2️⃣ | Image Descriptions | ✅ Yes | Text descriptions of images |
| 3️⃣ | Data Ingestion | ✅ Yes | Vector database populated |
| 4️⃣ | Retrieval | ✅ Yes | Search & retrieval functions |

---

**End of Pipeline Documentation**
