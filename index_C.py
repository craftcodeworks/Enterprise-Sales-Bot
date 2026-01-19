import os
import pyodbc
import json

from db import initialize_vector_db, force_delete_all_queries
from agent import run_sql_rag_agent
from schemas.state import ChatState

# --- MS Fabric DB Connection Details ---
USER_ID = os.getenv("FABRIC_DB_USER")
PASSWORD = os.getenv("FABRIC_DB_PASSWORD")
SERVER = os.getenv("FABRIC_DB_SERVER")
DATABASE = os.getenv("FABRIC_DB_NAME")



CONNECTION_STRING = (
    f'DRIVER={{ODBC Driver 18 for SQL Server}};'
    f'SERVER=tcp:{SERVER},1433;'
    f'DATABASE={DATABASE};'
    f'UID={USER_ID};'
    f'PWD={PASSWORD};'
    'Authentication=ActiveDirectoryPassword;'
    'Encrypt=yes;'
)


def main_chat_interface():
    print("="*60)
    print("  SALES DATA RAG AGENT - Terminal Interface")
    print("="*60)
    print("\n--- Running Startup Checks ---")
    
    # 1. Test DB connection
    try:
        cnxn = pyodbc.connect(CONNECTION_STRING)
        cnxn.close()
        print("✓ MS Fabric DB connection confirmed")
    except pyodbc.Error as ex:
        print(f"✗ DB connection FAILED: {ex}")
        return
    
    # 2. Initialize ChromaDB with optimized queries
    print("\n--- Initializing ChromaDB Vector Store ---")
    initialize_vector_db(query_file_path="queries_invoice.json")
    
    print("\n--- Startup Complete ---\n")
    print("="*60)
    print("  COMMANDS:")
    print("  • Type your sales question")
    print("  • 'SEED NOW' - Force re-seed ChromaDB")
    print("  • 'start over' or 'reset' - Clear conversation")
    print("  • 'exit' or 'quit' - End session")
    print("="*60)
    
    chat_state = ChatState()
    
    while True:
        user_question = input("\n👤 You: ").strip()
        
        # Manual seeding
        if user_question.upper() == 'SEED NOW':
            print("\n🔄 [MANUAL SEEDING TRIGGERED]")
            force_delete_all_queries()
            initialize_vector_db(query_file_path="queries_invoice.json")
            print("✓ [SEEDING COMPLETE]\n")
            continue
        
        # Exit
        if user_question.lower() in ['exit', 'quit']:
            print("\n👋 Goodbye! Session state cleared.")
            break
        
        if not user_question:
            continue
        
        print("🤖 Agent: Processing...", end="\r")
        
        # Run agent
        response_json_str = run_sql_rag_agent(user_question, CONNECTION_STRING, chat_state)
        
        try:
            response_dict = json.loads(response_json_str)
            print(f"🤖 Agent: {response_dict.get('bot_answer', 'Error parsing response')}")
        except json.JSONDecodeError:
            print(f"🤖 Agent: {response_json_str}")


if __name__ == "__main__":
    main_chat_interface()