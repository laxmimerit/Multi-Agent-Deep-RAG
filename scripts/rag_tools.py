# ## Advanced RAG - Retrieval Strategies
# ### Dense, Sparse, Hybrid, and Reranking

from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse

# re-ranking for better result
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# metadata filtering
from qdrant_client.models import Filter, FieldCondition, MatchValue

# metadata extraction from LLM
from scripts.schema import ChunkMetadata

from langchain_core.tools import tool
import subprocess
import sys

# Configuration
COLLECTION_NAME = "financial_docs"
EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "gemini-2.5-flash"

RERANKER_MODEL = "BAAI/bge-reranker-base"

# ### Initialize LLM and Vector Store

# Initialize LLM
llm = ChatGoogleGenerativeAI(model=LLM_MODEL)

# Gemini embeddings
embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

# Sparse embeddings
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

# Connect to existing collection
vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    sparse_embedding=sparse_embeddings,
    collection_name=COLLECTION_NAME,
    url="http://localhost:6333",
    retrieval_mode=RetrievalMode.HYBRID,
)


# ### Filter Extraction with LLM


def extract_filters(user_query: str):

    prompt = f"""
            Extract metadata filters from the query. Return None for fields not mentioned.

                <USER QUERY STARTS>
                {user_query}
                </USER QUERY ENDS>

                #### EXAMPLES
                COMPANY MAPPINGS:
                - Amazon/AMZN -> amazon
                - Google/Alphabet/GOOGL/GOOG -> google
                - Apple/AAPL -> apple
                - Microsoft/MSFT -> microsoft
                - Tesla/TSLA -> tesla
                - Nvidia/NVDA -> nvidia
                - Meta/Facebook/FB -> meta

                DOC TYPE:
                - Annual report -> 10-k
                - Quarterly report -> 10-q
                - Current report -> 8-k

                EXAMPLES:
                "Amazon Q3 2024 revenue" -> {{"company_name": "amazon", "doc_type": "10-q", "fiscal_year": 2024, "fiscal_quarter": "q3"}}
                "Apple 2023 annual report" -> {{"company_name": "apple", "doc_type": "10-k", "fiscal_year": 2023}}
                "Tesla profitability" -> {{"company_name": "tesla"}}

                Extract metadata based on the user query only:
            """

    structurerd_llm = llm.with_structured_output(ChunkMetadata)

    metadata = structurerd_llm.invoke(prompt)

    if metadata:
        filters = metadata.model_dump(exclude_none=True)
    else:
        filters = {}

    return filters


@tool
def hybrid_search(query: str, k: int = 5):
    """
    Search historical financial documents (SEC filings: 10-K, 10-Q, 8-K) using hybrid search.

    **IMPORTANT: This is the PRIMARY tool for financial research.**
    **ALWAYS call this tool FIRST for ANY financial question unless:**
    - User explicitly asks for "current", "live", "real-time", or "latest" market data
    - User asks about current stock prices or today's market information

    This tool searches through:
    - Historical SEC filings (10-K annual reports, 10-Q quarterly reports)
    - Financial statements, revenue, expenses, cash flow data
    - Company performance metrics from past quarters and years
    - Automatically extracts filters (company, year, quarter, doc type) from your query

    Use this for queries about:
    - Historical revenue, profit, expenses ("What was Amazon's revenue in Q1 2024?")
    - Year-over-year or quarter-over-quarter comparisons
    - Financial metrics from SEC filings
    - Any historical financial data

    Args:
        query: Natural language search query (e.g., "Amazon Q1 2024 revenue")
        k: Number of results to return (default: 5)

    Returns:
        List of Document objects with page content and metadata (source_file, page_number, etc.)
    """

    filters = extract_filters(query)

    qdrant_filter = None

    if filters:
        condition = [
            FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value))
            for key, value in filters.items()
        ]

        qdrant_filter = Filter(must=condition)

    results = vector_store.similarity_search(query=query, k=k, filter=qdrant_filter)

    return results



import subprocess
import sys

@tool
def live_finance_researcher(query: str):
    """
    Research live stock data using Yahoo Finance MCP.
    
    Use this tool to get:
    - Current stock prices and real-time market data
    - Latest financial news
    - Stock recommendations and analyst ratings
    - Option chains and expiration dates
    - Recent stock actions (splits, dividends)
    
    Args:
        query: The financial research question about current market data
        
    Returns:
        Research results from Yahoo Finance
    """

    code = f"""
import asyncio
from scripts.yahoo_mcp import finance_research
asyncio.run(finance_research("{query}"))
"""
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)

    return result.stdout