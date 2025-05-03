# utils/chroma_db.py
import chromadb
from chromadb.utils import embedding_functions
from config import config  

class ChromaManager:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)

        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.EMBEDDING_MODEL_NAME
        )
        self.collection = None 

    def get_or_create_collection(self, collection_name: str = config.CHROMA_COLLECTION_NAME):
        """Gets or creates a ChromaDB collection."""
        if self.collection is None:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
        return self.collection

    def add_data(self, documents: list[str], ids: list[str]):
        """Adds data to the ChromaDB collection."""
        if self.collection is None:
             raise ValueError("Collection must be initialized before adding data. Call get_or_create_collection first.")
        self.collection.add(
            documents=documents,
            ids=ids
        )

    def query_data(self, query_texts: list[str], n_results: int = 1):
        """Queries the ChromaDB collection."""
        if self.collection is None:
            raise ValueError("Collection must be initialized before querying data. Call get_or_create_collection first.")
        results = self.collection.query(
            query_texts=query_texts,
            n_results=n_results
        )
        return results


# Get the embedding model name from the configuration - Moved
embedding_model_name = config.EMBEDDING_MODEL_NAME


chroma_manager = ChromaManager()