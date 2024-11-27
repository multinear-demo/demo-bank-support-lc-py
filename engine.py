"""
Core RAG (Retrieval-Augmented Generation) engine implementation using LangChain.
This module provides the main RAG functionality for the Bank Customer Support app:
- Document ingestion from FAQ text files
- Vector indexing of documents
- Query processing using GPT-4 with retrieval augmentation
"""

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import os
from typing import Tuple, List
from pathlib import Path


class RAGEngine:
    """
    Core RAG engine using LangChain.
    This class handles document ingestion, indexing, and query processing.
    """

    def __init__(self):
        """
        Initialize the RAG engine by setting up the document index.
        """
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.model = os.getenv("OPENAI_MODEL", "gpt-4o")
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", 0.2))

        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(
            model_name=self.model,
            temperature=self.temperature
        )
        self.vector_store = None

    def refresh_index(self):
        """
        (Re)build the document index by processing all documents in the data directory.
        """
        # Load the document
        loader = TextLoader(str(Path(__file__).parent / "data" / "acme_bank_faq.txt"))
        documents = loader.load()

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)

        # Create vector store
        self.vector_store = FAISS.from_documents(splits, self.embeddings)

    async def process_query(
        self, msg_list: List[Tuple[str, bool]]
    ) -> Tuple[str, List[str]]:
        """
        Process a user query using RAG with the provided chat history.

        Args:
            msg_list (List[Tuple[str, bool]]): A list of messages from session history.
                Each tuple contains:
                - str: The message text.
                - bool: Indicator if the message is from the user (True) or AI (False).

        Returns:
            Tuple[str, List[str]]: A tuple containing:
                - str: The AI's response to the user's query
                - List[str]: A list of source documents used (empty for now)
        """
        try:
            # Build index if not already loaded
            if not self.vector_store:
                self.refresh_index()

            # Format chat history for LangChain
            chat_history = []
            for i in range(0, len(msg_list) - 1, 2):
                if i + 1 < len(msg_list):
                    chat_history.append((msg_list[i][0], msg_list[i + 1][0]))

            # Create QA chain
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vector_store.as_retriever(
                    search_kwargs={"k": 3}
                ),
                return_source_documents=True,
                verbose=False
            )

            # Get response
            result = await qa_chain.ainvoke({
                "question": msg_list[-1][0],
                "chat_history": chat_history
            })

            return result["answer"], []
        except Exception as e:
            print(e)
            return "Error processing request. Try again.", []
