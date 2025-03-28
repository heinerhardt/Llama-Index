import os
import asyncio
import logging
from typing import List, Optional
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb import PersistentClient
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import (
    Document, SimpleDirectoryReader, VectorStoreIndex,
    PromptTemplate, Settings
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine, BaseQueryEngine
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.evaluation import FaithfulnessEvaluator
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.schema import QueryBundle, NodeWithScore, TextNode
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent

@dataclass
class ProcessingConfig:
    chunk_size: int = 300
    chunk_overlap: int = 30
    embedding_model: str = "text-embedding-ada-002"  # OpenAI Embedding Model
    llm_model: str = "gpt-4"  # OpenAI LLM Model
    openai_model: str = "gpt-4"
    temperature: float = 0.1
    db_path: str = "./zenith_vector_db"
    collection_name: str = "zenith_vector_collection"
    embedding_dim = 1536

class DocumentProcessor:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.setup_logging()
        self.initialize_credentials()
        self.setup_database()
        self.setup_models()
        self.agent = None

    def setup_logging(self):
        logging.basicConfig(
            filename='./data_proc.log',
            filemode='a',
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )

    def initialize_credentials(self):
        self.openai_key = os.environ.get("OPENAI_API_KEY")

        if not self.openai_key:
            raise ValueError("Missing required OpenAI API key. Please set the OPENAI_API_KEY environment variable.")

        logging.info("OpenAI API key initialized.")

    def setup_database(self):
        try:
            self.db = PersistentClient(path=self.config.db_path)
            self.collection = self.db.get_or_create_collection(
                self.config.collection_name,
                metadata={"hnsw:space": "cosine"}                
            )
            self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
            logging.info("Vector store initialized successfully")
        except Exception as e:
            logging.error(f"Database initialization failed: {e}")
            raise

    def setup_models(self):
        self.llm = OpenAI(
            model=self.config.llm_model,
            temperature=self.config.temperature
        )
        self.embed_model = OpenAIEmbedding(
            model=self.config.embedding_model
        )

    def get_document_hash(self, document: Document) -> str:
        """Generate a unique hash for document content."""
        import hashlib
        return hashlib.md5(document.text.encode()).hexdigest()

    def is_document_processed(self, doc_hash: str) -> bool:
        """Check if document has already been processed."""
        result = self.collection.get(
            where={"document_hash": doc_hash},
            limit=1
        )
        return len(result['ids']) > 0

    async def process_documents(self, documents: List[Document]):
        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap
                ),
                self.embed_model,
            ],
            vector_store=self.vector_store,
        )

        processed_nodes = []
        for doc in documents:
            doc_hash = self.get_document_hash(doc)
            if self.is_document_processed(doc_hash):
                logging.info(f"Skipping already processed document: {doc_hash}")
                continue

            try:
                doc.metadata["document_hash"] = doc_hash
                nodes = await pipeline.arun(documents=[doc])
                processed_nodes.extend(nodes)
                logging.info(f"Successfully processed document: {doc_hash}")
            except Exception as e:
                logging.error(f"Error processing document {doc_hash}: {e}")

        return VectorStoreIndex(processed_nodes, embed_model=self.embed_model)

    def create_query_engine(self, index: VectorStoreIndex) -> RetrieverQueryEngine:
        Settings.llm = OpenAI(
            model=self.config.openai_model,
            temperature=self.config.temperature
        )

        prompt = PromptTemplate(
            template="Write a concise summary of the following:\n{text}\nCONCISE SUMMARY:"
        )

        # Create a retriever from the index
        retriever = VectorIndexRetriever(index=index)

        return RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=TreeSummarize(summary_template=prompt),
        )

    def create_chat_engine(self, index: VectorStoreIndex):
        Settings.llm = OpenAI(
            model=self.config.openai_model,
            temperature=self.config.temperature
        )
        retriever = VectorIndexRetriever(index=index)
        return SimpleChatEngine.from_defaults(
            retriever=retriever,
            llm=Settings.llm,
        )

    def create_retriever_agent(self, index: VectorStoreIndex):
        Settings.llm = OpenAI(
            model=self.config.openai_model,
            temperature=self.config.temperature
        )

        # Create a retriever
        retriever = VectorIndexRetriever(index=index)

        # Create a tool for retrieving documents
        doc_reader_tool = QueryEngineTool.from_defaults(
            query_engine=RetrieverQueryEngine(retriever=retriever),
            name="doc_reader",
            description="Useful for answering questions about the documents.",
        )

        # Create the agent
        self.agent = OpenAIAgent.from_tools(
            tools=[doc_reader_tool],
            llm=OpenAI(model=self.config.openai_model, temperature=self.config.temperature),
            verbose=True
        )
        return self.agent

    async def run_agent_chat(self, message: str):
        if self.agent:
            try:
                response = self.agent.chat(message)  # OpenAIAgent.chat is synchronous
                logging.info(f"Agent Response: {response}")
                return response
            except Exception as e:
                logging.error(f"Error during agent chat: {e}")
                return None
        else:
            logging.error("Agent not initialized. Call create_retriever_agent first.")
            return None

    async def evaluate_response(self, query_engine: BaseQueryEngine, query: str):
        try:
            response = await query_engine.aquery(query)
            evaluator = FaithfulnessEvaluator()
            if response and response.source_nodes:
                source_texts = [node.node.get_content() for node in response.source_nodes]
                logging.info(f"Source Texts for Query '{query}':")
                for i, text in enumerate(source_texts):
                    logging.info(f"Source {i+1}:\n{text}\n---")
                score = await evaluator.aevaluate(
                    query=query, contexts=source_texts, response=response.response
                )
                logging.info(f"Response Faithfulness Score: {score}")
                return score
            else:
                logging.warning("No source nodes found in the response, skipping evaluation.")
                return None
        except Exception as e:
            logging.error(f"Evaluation failed: {e}")
            logging.error(f"Error: {e}")
            return None

async def main():
    config = ProcessingConfig()
    processor = DocumentProcessor(config)

    try:
        documents = SimpleDirectoryReader(
            input_dir="C://Temp//Llama Index//Quickspecks"
        ).load_data()

        if not documents:
            logging.warning("No documents found to process")
            return

        index = await processor.process_documents(documents)

        # Create the Retriever-Enabled OpenAI Agent
        processor.create_retriever_agent(index)

        # Example usage of the agent
        query = "Summarize the key features of the HPE Synergy 480 Gen10" 
        response = await processor.run_agent_chat(query)
        if response:
            print(f"Agent Response for '{query}':\n{response.response}")

        # You can also have a conversation with the agent
        # response = await processor.run_agent_chat("Can you summarize the information?")
        # if response:
        #     print(f"Agent Response:\n{response.response}")

    except Exception as e:
        logging.error(f"Processing pipeline failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())