from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain_core.tools import tool
from qdrant_client.models import Filter, FieldCondition, MatchValue

from scripts.schema import ChunkMetadata


# Configuration
COLLECTION_NAME = "financial_docs"
EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "gemini-2.5-flash"


# Initialize components
llm = ChatGoogleGenerativeAI(model=LLM_MODEL)
embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

# Connect to vector store
vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    sparse_embedding=sparse_embeddings,
    collection_name=COLLECTION_NAME,
    url="http://localhost:6333",
    retrieval_mode=RetrievalMode.HYBRID
)


def extract_filters(user_query: str) -> dict:
    """
    Extract metadata filters from user query using LLM.

    Args:
        user_query: Natural language query

    Returns:
        Dictionary of filters (company_name, doc_type, fiscal_year, fiscal_quarter)
    """

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

    structured_llm = llm.with_structured_output(ChunkMetadata)
    metadata = structured_llm.invoke(prompt)
    filters = metadata.model_dump(exclude_none=True)

    return filters


@tool
def hybrid_search(query: str, k: int = 5) -> list:
    """
    Perform hybrid search (dense + sparse vectors) with automatic filter extraction.

    Args:
        query: Search query
        k: Number of results to return

    Returns:
        List of Document objects
    """

    filters = extract_filters(query)
    qdrant_filter = None

    if filters:
        conditions = [
            FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value))
            for key, value in filters.items()
        ]
        qdrant_filter = Filter(must=conditions)

    results = vector_store.similarity_search(
        query=query,
        k=k,
        filter=qdrant_filter
    )

    return results
