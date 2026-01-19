import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import uvicorn
import requests
import asyncio
from fastapi import FastAPI, Request
import json
import pytz
from datetime import datetime, timedelta
from typing import Dict, Any
from schemas.state import ChatState
from agent import run_sql_rag_agent

# --- MS Fabric Connection ---
USER_ID = os.getenv("FABRIC_DB_USER")
PASSWORD = os.getenv("FABRIC_DB_PASSWORD")
SERVER = os.getenv("FABRIC_DB_SERVER")
DATABASE = os.getenv("FABRIC_DB_NAME")

FABRIC_DB_CONNECTION_STRING = (
    f'DRIVER={{ODBC Driver 18 for SQL Server}};'
    f'SERVER=tcp:{SERVER},1433;'
    f'DATABASE={DATABASE};'
    f'UID={USER_ID};'
    f'PWD={PASSWORD};'
    'Authentication=ActiveDirectoryPassword;'
    'Encrypt=yes;'
)

# --- Teams Bot Config ---
MICROSOFT_APP_ID = os.getenv("TEAMS_APP_ID")
MICROSOFT_APP_PASSWORD = os.getenv("TEAMS_APP_PASSWORD")
Tenant_ID = os.getenv("TEAMS_TENANT_ID")

# --- Global State ---
token_expiry_ist: datetime | None = None
TOKEN_LIFETIME = 1  # Hours
bot_token: str | None = None
conversation_store: Dict[str, Any] = {}
last_message_id_store: Dict[str, str] = {}
agent_states: Dict[str, ChatState] = {}
MAX_MESSAGES = 10

# --- Initialize ChromaDB at startup ---
from db import initialize_vector_db
print("--- Initializing ChromaDB Vector Store ---")
initialize_vector_db(query_file_path="queries_invoice.json")
print("--- ChromaDB Ready ---")

app = FastAPI()


# --- Get Bot Token ---
async def get_bot_token():
    """Fetches OAuth token from Microsoft Bot Framework."""
    url = f"https://login.microsoftonline.com/{Tenant_ID}/oauth2/v2.0/token"
    data = {
        "grant_type": "client_credentials",
        "client_id": MICROSOFT_APP_ID,
        "client_secret": MICROSOFT_APP_PASSWORD,
        "scope": "https://api.botframework.com/.default"
    }
    response = await asyncio.to_thread(requests.post, url, data=data)
    
    resp_json = response.json()
    if "access_token" not in resp_json:
        raise Exception(f"Failed to get Bot Token: {resp_json}")
    return resp_json["access_token"]


# --- Send Typing Indicator ---
async def send_typing_indicator(service_url: str, conv_id: str, token: str, body: dict):
    """Sends typing indicator to Teams."""
    api_url = f"{service_url.rstrip('/')}/v3/conversations/{conv_id}/activities"
    
    payload = {
        "type": "typing",
        "from": body["recipient"],
        "recipient": body["from"],
        "conversation": body["conversation"]
    }
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    try:
        await asyncio.to_thread(requests.post, api_url, headers=headers, json=payload, timeout=2)
    except Exception as e:
        print(f"[WARNING] Typing indicator failed: {e}")


# --- Format Response for Teams ---
def format_teams_message(text: str) -> str:
    """
    Formats response text for better display in Teams.
    Converts markdown-style formatting to Teams-compatible format.
    """
    # Teams supports basic markdown
    # No special conversion needed, but we can add adaptive cards later
    return text
@app.get("/health")
async def health(request: Request):
    """Handles incoming messages from Teams."""
    return "OK"

# --- Main Message Handler ---
@app.post("/api/messages")
async def messages(request: Request):
    """Handles incoming messages from Teams."""
    global token_expiry_ist, bot_token, agent_states, last_message_id_store, conversation_store
    
    body = await request.json()
    print("\n" + "="*60)
    print("INCOMING TEAMS MESSAGE")
    print("="*60)
    print(json.dumps(body, indent=2)[:500])

    conv_id = body["conversation"]["id"]
    message_id = body.get("id")
    sender_id = body["from"]["id"]
    bot_id = body["recipient"]["id"]
    activity_type = body.get("type")

    # Ignore non-message activities or bot's own messages
    if activity_type != "message" or sender_id == bot_id:
        return {"status": "ignored"}

    # Prevent duplicate processing
    if last_message_id_store.get(conv_id) == message_id:
        return {"status": "ignored"}

    last_message_id_store[conv_id] = message_id
    user_text = body.get("text", "").strip()
    service_url = body["serviceUrl"]
    
    # Initialize ChatState per conversation
    if conv_id not in agent_states:
        agent_states[conv_id] = ChatState()
        print(f"[DEBUG] New ChatState initialized for conversation: {conv_id}")
    
    current_chat_state = agent_states[conv_id]
    
    # Store conversation history
    if conv_id not in conversation_store:
        conversation_store[conv_id] = {
            "serviceUrl": service_url,
            "conversation": body["conversation"],
            "from": body["from"],
            "recipient": body["recipient"],
            "messages": []
        }
    
    conversation_store[conv_id]["messages"].append({
        "from": body["from"]["name"],
        "text": user_text,
        "timestamp": body.get("timestamp")
    })
    conversation_store[conv_id]["messages"] = conversation_store[conv_id]["messages"][-MAX_MESSAGES:]

    # --- Token Management ---
    ist = pytz.timezone("Asia/Kolkata")
    now_ist = datetime.now(ist)
    
    if token_expiry_ist is None or now_ist >= token_expiry_ist:
        bot_token = await get_bot_token()
        token_expiry_ist = now_ist + timedelta(hours=TOKEN_LIFETIME)
        print("[TOKEN] New bot token generated")
    else:
        print("[TOKEN] Using existing token")

    # --- Send Typing Indicator ---
    # Skip typing indicator for greetings (instant response expected)
    if user_text.lower() not in ["hi", "hello", "hii", "hey", "start over", "reset"]:
        await send_typing_indicator(service_url, conv_id, bot_token, body)

    # --- Agent Processing ---
    try:
        print(f"[AGENT] Processing: '{user_text}'")
        print(f"[STATE] Pending query: {current_chat_state.pending_query_id}")
        print(f"[STATE] Missing params: {current_chat_state.missing_params}")
        
        reply_text = await asyncio.to_thread(
            run_sql_rag_agent,
            user_question=user_text,
            connection_string=FABRIC_DB_CONNECTION_STRING,
            chat_state=current_chat_state
        )
        
        # Parse agent response
        try:
            agent_json = json.loads(reply_text)
            reply_text = agent_json.get("bot_answer", "Error: Unparseable response")
        except json.JSONDecodeError:
            # Agent returned plain text (shouldn't happen, but handle it)
            pass
        
        # Format for Teams display
        reply_text = format_teams_message(reply_text)
        
        print(f"[AGENT] Response length: {len(reply_text)} chars")
        print(f"[STATE] After processing - Pending: {current_chat_state.pending_query_id}")
        
    except Exception as e:
        print(f"[ERROR] Agent failure: {e}")
        import traceback
        print(f"[ERROR] Traceback:\n{traceback.format_exc()}")
        
        # User-friendly error message
        reply_text = (
            "I encountered an error while processing your request. "
            "Please try rephrasing your question or type 'start over' to begin again."
        )

    # --- Send Response to Teams ---
    api_url = f"{service_url.rstrip('/')}/v3/conversations/{conv_id}/activities"

    payload = {
        "type": "message",
        "text": reply_text,
        "from": body["recipient"],
        "recipient": body["from"],
        "conversation": body["conversation"],
        "replyToId": message_id
    }

    headers = {
        "Authorization": f"Bearer {bot_token}",
        "Content-Type": "application/json"
    }

    try:
        resp = await asyncio.to_thread(requests.post, api_url, headers=headers, json=payload, timeout=10)
        print(f"[TEAMS] Response status: {resp.status_code}")
        
        if resp.status_code != 200 and resp.status_code != 201:
            print(f"[TEAMS] Error response: {resp.text}")
        
        return {"status": resp.status_code}
        
    except Exception as e:
        print(f"[ERROR] Failed to send Teams response: {e}")
        return {"status": "error", "message": str(e)}


# --- Health Check ---
@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "sql_rag_chatbot",
        "active_conversations": len(agent_states),
        "total_messages_stored": sum(len(conv["messages"]) for conv in conversation_store.values())
    }


# --- Utility: Get Last Messages ---
@app.get("/last-messages/{conv_id}")
async def get_last_messages(conv_id: str):
    """Get conversation history for debugging."""
    if conv_id in conversation_store:
        return {
            "conversation_id": conv_id,
            "messages": conversation_store[conv_id]["messages"],
            "state": {
                "pending_query": agent_states[conv_id].pending_query_id if conv_id in agent_states else None,
                "missing_params": agent_states[conv_id].missing_params if conv_id in agent_states else [],
                "attempts": agent_states[conv_id].param_collection_attempts if conv_id in agent_states else 0
            }
        }
    return {"error": "Conversation not found"}


# --- Utility: Clear Conversation State ---
@app.post("/clear-conversation/{conv_id}")
async def clear_conversation(conv_id: str):
    """Manually clear conversation state (for debugging/admin)."""
    if conv_id in agent_states:
        agent_states[conv_id].clear_all()
        return {"status": "cleared", "conversation_id": conv_id}
    return {"error": "Conversation not found"}


# --- Utility: Get All Active Conversations ---
@app.get("/conversations")
async def get_conversations():
    """Get list of all active conversation IDs."""
    return {
        "total": len(agent_states),
        "conversations": [
            {
                "conv_id": conv_id,
                "message_count": len(conversation_store.get(conv_id, {}).get("messages", [])),
                "pending_query": state.pending_query_id,
                "missing_params": state.missing_params
            }
            for conv_id, state in agent_states.items()
        ]
    }


# --- Run Server ---
if __name__ == "__main__":
    print("="*60)
    print("  SALES DATA RAG AGENT - Teams Bot Server")
    print("="*60)
    print("\n🚀 Starting FastAPI server on port 5002...")
    print("📝 IMPORTANT: Seed queries using index.py FIRST!")
    print("   Run: python index.py")
    print("   Then type: SEED NOW")
    print("\n📡 Available endpoints:")
    print("   POST /api/messages - Main bot endpoint")
    print("   GET  /health - Health check")
    print("   GET  /last-messages/{conv_id} - View conversation history")
    print("   POST /clear-conversation/{conv_id} - Clear conversation state")
    print("   GET  /conversations - List all active conversations")
    print("\n" + "="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=5002)