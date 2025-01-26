from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from pathlib import Path

class VectorStore:
    def __init__(self, persist_directory: str = "./data/vectorstore"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.vectorstore = Chroma(
            persist_directory=str(self.persist_directory),
            embedding_function=OpenAIEmbeddings()
        )

    def add_texts(self, texts: list[str], metadatas: list[dict] = None):
        return self.vectorstore.add_texts(texts=texts, metadatas=metadatas)

    def similarity_search(self, query: str, k: int = 4):
        return self.vectorstore.similarity_search(query, k=k)
