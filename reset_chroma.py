# reset_chroma.py

import os
import sys

# Add the parent directory to the path to import db (if necessary for your environment)
# If index1.py runs fine, you might not need this, but it's good for utility scripts.
# sys.path.append(os.path.dirname(os.path.abspath(__file__))) 

# Import the necessary function from your existing db.py file
try:
    from db import force_delete_all_queries
    # Import CHROMA_CLIENT and COLLECTION_NAME for confirmation (optional, but helpful)
    from db import CHROMA_CLIENT, COLLECTION_NAME
except ImportError:
    print("Error: Could not import 'force_delete_all_queries' from db.py.")
    print("Please ensure db.py is in the same directory.")
    sys.exit(1)


def main_reset_chroma():
    """
    Utility script to connect to the persistent ChromaDB client and delete all
    documents in the SQL query store collection.
    """
    print("--- ChromaDB Reset Utility ---")
    print(f"Targeting Collection: '{COLLECTION_NAME}' in Path: './chroma_db_data'")

    # Call the function that performs the deletion logic
    force_delete_all_queries()
    
    # Check if the folder is still present (it should be, just empty)
    if os.path.exists("./chroma_db_data"):
        print("\nNote: The 'chroma_db_data' folder remains, but is now empty of documents.")
    else:
        print("\nNote: The 'chroma_db_data' folder was not found.")


if __name__ == "__main__":
    main_reset_chroma()