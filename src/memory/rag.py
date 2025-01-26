from langchain.memory import ConversationBufferMemory
from .vectorstore import VectorStore

class RAGMemory:
    def __init__(self):
        self.vectorstore = VectorStore()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

    def add_to_memory(self, text: str, metadata: dict = None):
        self.vectorstore.add_texts([text], [metadata] if metadata else None)

    def get_relevant_context(self, query: str, k: int = 4):
        return self.vectorstore.similarity_search(query, k)

    def get_chat_history(self):
        return self.memory.chat_memory.messages
