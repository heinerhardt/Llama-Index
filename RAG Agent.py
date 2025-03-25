import os
import asyncio
import logging

# ChromaDB imports
import chromadb
from chromadb import PersistentClient
from llama_index.vector_stores.chroma import ChromaVectorStore

# Llama Index imports
from llama_index.core import Document, SimpleDirectoryReader, VectorStoreIndex, PromptTemplate, Settings
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.evaluation import FaithfulnessEvaluator
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

# OpenAI imports
from llama_index.llms.openai import OpenAI


# Transformers imports
from huggingface_hub import login, whoami
from transformers import AutoTokenizer


# Configure logging to file
logging.basicConfig(
    filename='./data_proc.log',  # Log file name
    filemode='a',                # Append mode
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Get the Hugging Face and OpenAI API keys
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Login to Hugging Face with the token
try:
    login(token=HF_TOKEN)
    user = whoami()
    print(f"Successfully logged in as: {user}")
except Exception as e:
    print(f"Login failed: {e}")


# Load Llama3 model and tokenizer
model_name = "meta-llama/Llama-3.1-8B-Instruct"  # Replace with your desired model and variant
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize HuggingFaceLLM
llm = HuggingFaceInferenceAPI( 
    model_name=model_name,
    tokenizer=tokenizer, 
    token=True
)

# Create ChromaDB
db_path = "./zenith_vector_db"

try:
    db = PersistentClient(path=db_path)
    chroma_collection = db.get_or_create_collection("zenith_vector_collection")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    logging.info("ChromaDB initialized and collection created.")
    
except Exception as e:
    logging.error(f"An error occurred while initializing ChromaDB: {e}")

# Load the documents
reader = SimpleDirectoryReader(input_dir="C://Temp//Llama Index//Quickspecks")
documents = reader.load_data()

# Get a document by its ID
def get_document_by_id(document_id: str) -> Document:
    """Get a document in the collection by its ID."""
    result = vector_store.collection_kwargs.get("chroma_collection")
    result = chroma_collection.get(include=["documents", "metadatas"], where={"metadatas.doc_id": document_id})
    if result and "documents" in result and result["documents"]:
        metadata = result["metadatas"][0]
        return Document(page_content=result["documents"][0], metadata=metadata)
    else:
        return None

# Define an async function to run the pipeline
async def run_pipeline(documents):
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=300, chunk_overlap=30),
            HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", use_auth_token=HF_TOKEN),
        ],
        vector_store=vector_store, 
    )
    
    nodes = []
    for document in documents:
        existing_doc = get_document_by_id(document.doc_id)
        if existing_doc is not None:
            logging.info(f"Document '{document.doc_id}' has already been processed.")
            continue
        
        node = await pipeline.arun(documents=[document])
        nodes.extend(node)  # Collect nodes
        
        logging.info(f"Document '{document.doc_id}' processed and added to the vector database.")
        
        count = chroma_collection.count()  # Hypothetical function
        logging.info(f"Total documents in vector store after update: {count}")

    index = VectorStoreIndex(nodes, embed_model=HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", use_auth_token=HF_TOKEN))
    return index

# Evaluate response quality
def evaluate_response(response):
    evaluator = FaithfulnessEvaluator()
    faithfulness_score = evaluator.evaluate(response)

    logging.info(f"Response Evaluation - Faithfulness: {faithfulness_score}")

# Run the async function
if __name__ == "__main__":
    if documents:
        nodes = asyncio.run(run_pipeline(documents))
        logging.info("Pipeline run completed.")
        
        Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
        
        # Create a custom prompt template
        prompt_template = """Write a concise summary of the following:
        "{text}"
        CONCISE SUMMARY:"""
        prompt = PromptTemplate(template=prompt_template)
        
        # Use TreeSummarize and custom prompt with RetrieverQueryEngine
        summarizer = TreeSummarize(summary_template=prompt)
        query_engine = RetrieverQueryEngine(
            index=nodes,
            llm=llm,
            synthesizer=summarizer,
            response_mode="tree_summarize",
        )
        
        response = query_engine.query("What is the meaning of life?")
        logging.info(f"Query response: {response}")

        # Evaluate the response quality
        evaluate_response(response)
    else:
        logging.warning("No documents loaded.")