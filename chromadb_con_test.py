import os
import logging
from chromadb import PersistentClient

# Set up logging
logging.basicConfig(level=logging.INFO)

# Configuration
db_path = "./zenith_vector_db"
collection_name = "zenith_vector_collection"

try:
    # Create or connect to ChromaDB client
    db = PersistentClient(path=db_path)
    logging.info(f"Connected to ChromaDB at: {db_path}")

    # Get the specified collection
    collection = db.get_collection(collection_name)
    logging.info(f"Collection '{collection_name}' retrieved successfully.")

    # Fetch all elements from the collection using query
    all_elements = collection.query()  # You can modify this query as needed
    logging.info(f"Retrieved {len(all_elements.get('ids', []))} items from the collection.")

    # Iterate through the retrieved elements and print their content
    if all_elements and "documents" in all_elements:
        for i in range(len(all_elements["documents"])):
            document = all_elements["documents"][i]
            doc_id = all_elements.get("ids", [])[i] if all_elements.get("ids") else "N/A"

            print("--- Document ---")
            print(f"ID: {doc_id}")
            print(f"Content:\n{document[:200]}...")  # Print first 200 characters
            print("----------------")
    else:
        logging.info("No documents found in the retrieved elements.")

except Exception as e:
    logging.error(f"An error occurred while retrieving elements from the collection: {e}")