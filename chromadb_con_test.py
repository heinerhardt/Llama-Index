import os
import logging
from openai import OpenAI
from chromadb import PersistentClient

# Set up logging
logging.basicConfig(level=logging.INFO)

# Configuration
db_path = "./zenith_vector_db"
collection_name = "zenith_vector_collection"

# Set your OpenAI API key
openai_api_key = "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"  # Replace with your actual API key

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

def generate_embeddings(texts):
    try:
        response = client.embeddings.create(
            input=texts,
            model="text-embedding-ada-002"  # Specify the embedding model
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        logging.error(f"Error generating embeddings: {e}")
        return []

try:
    # Create or connect to ChromaDB client
    db = PersistentClient(path=db_path)
    logging.info(f"Connected to ChromaDB at: {db_path}")

    # Get the specified collection
    collection = db.get_collection(collection_name)
    logging.info(f"Collection '{collection_name}' retrieved successfully.")

    # Query texts to generate embeddings
    query_texts = ["HPE Alletra 4110", "iLO Amplifier Pack"]  # Add more texts as needed

    # Generate embeddings for the query texts
    query_embeddings = generate_embeddings(query_texts)

    if not query_embeddings:
        logging.warning("Could not generate embeddings. Skipping query.")
    else:
        # Fetch documents based on embeddings
        query_results = collection.query(
            query_embeddings=query_embeddings,  # Use the generated embeddings
            n_results=10  # Adjust limit as necessary
        )

        # Process the results
        documents = query_results.get('documents', [])
        if documents:
            for record in documents:
                print("--- Document ---")
                # Print all elements in the record list
                for element in record:
                    print(element)
                print("----------------")
        else:
            logging.info("No documents found matching the query.")

except Exception as e:
    logging.error(f"An error occurred while querying the collection: {e}")