import pyodbc
from typing import List, Tuple, Dict, Any
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# =========================
# 🔹 LLM ROUTER
# =========================

ROUTER_LLM = ChatGroq(
    temperature=0,
    model_name="moonshotai/kimi-k2-instruct-0905",
    groq_api_key=os.getenv("GROQ_API_KEY")
)


ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
You are a query routing classifier for a sales analytics system.

Return exactly ONE label:
CSO
PRODUCT
EXPORT
DOMESTIC
STATE
CATEGORY
GENERAL

Routing Rules (apply in this priority order):

1) If the question contains a CSO code (e.g. DCBH01, CSO123) OR the word 'CSO' → CSO
2) If the question is about products, product types, SKUs, items, or product performance → PRODUCT
3) If the question contains the word export or exports → EXPORT
4) If the question contains the word domestic → DOMESTIC
5) If the question mentions any Indian state name or standard state code → STATE
6) CRITICAL: If the question mentions 'salesperson', 'who generated', 'rank', 'performer', (about people, not products) → GENERAL
7) If the question mentions a business category (FMEG, wires, cables, switchgear, wiring devices, etc) → CATEGORY
8) Otherwise → GENERAL

Return only the label.
"""),
    ("human", "{question}")
])

ROUTER_CHAIN = ROUTER_PROMPT | ROUTER_LLM | StrOutputParser()


# =========================
# 🔹 CHROMA SETUP
# =========================

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-albert-small-v2"

EMBEDDINGS = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL_NAME
)

CHROMA_CLIENT = PersistentClient(path="./chroma_db_data")
COLLECTION_NAME = "sql_query_store"


# =========================
# 🔹 SQL EXECUTION
# =========================

def execute_sql_query_from_string(connection_string: str, sql_query: str) -> Tuple[List[str], List[List]]:
    guardrail = ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE"]
    q = sql_query.strip().upper()

    if any(word in q for word in guardrail) or not q.startswith("SELECT"):
        raise ValueError("SQL Guardrail Violated")

    cnxn = pyodbc.connect(connection_string)
    try:
        cursor = cnxn.cursor()
        cursor.execute(sql_query)
        columns = [c[0] for c in cursor.description]
        rows = [list(r) for r in cursor.fetchall()]
        return columns, rows
    finally:
        cnxn.close()


# =========================
# 🔹 VECTOR DB MANAGEMENT
# =========================

def initialize_vector_db(query_file_path: str = "queries_optimized.json"):
    with open(query_file_path) as f:
        queries = json.load(f)

    collection = CHROMA_CLIENT.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=EMBEDDINGS
    )

    ids = [q["id"] for q in queries]
    docs = [q["question"] for q in queries]

    metas = [{
        "sql_query": q["sql"],
        "parameters": json.dumps(q.get("parameters", [])),
        "optional_parameters": json.dumps(q.get("optional_parameters", [])),
        "defaults": json.dumps(q.get("defaults", {}))
    } for q in queries]

    collection.upsert(documents=docs, metadatas=metas, ids=ids)
    print(f"SUCCESS: Synced {len(ids)} SQL queries")


def get_query_by_id(query_id: str) -> Tuple[str, str, List[str], List[str], Dict[str, Any]]:
    """
    Fetch a specific query by its ID directly from ChromaDB.
    Used for query switching without re-running semantic search.
    """
    collection = CHROMA_CLIENT.get_collection(
        name=COLLECTION_NAME,
        embedding_function=EMBEDDINGS
    )
    
    result = collection.get(ids=[query_id], include=["metadatas"])
    
    if not result["ids"]:
        raise ValueError(f"Query ID '{query_id}' not found in ChromaDB")
    
    meta = result["metadatas"][0]
    return (
        query_id,
        meta["sql_query"],
        json.loads(meta.get("parameters", "[]")),
        json.loads(meta.get("optional_parameters", "[]")),
        json.loads(meta.get("defaults", "{}"))
    )


# =========================
# 🧠 HYBRID ROUTING + SEARCH
# =========================

def semantic_search_sql(user_query: str, k: int = 1) -> Tuple[str, str, List[str], List[str], Dict[str, Any]]:

    collection = CHROMA_CLIENT.get_collection(
        name=COLLECTION_NAME,
        embedding_function=EMBEDDINGS
    )

    # 1️⃣ LLM Routing
    family = ROUTER_CHAIN.invoke({"question": user_query}).strip().upper()
    print(f"[DEBUG] LLM Query Family: {family}")

    q_lower = " " + user_query.lower() + " "

    INDIAN_STATES = [
        "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh",
        "goa", "gujarat", "haryana", "himachal pradesh", "jharkhand", "karnataka",
        "kerala", "madhya pradesh", "maharashtra", "manipur", "meghalaya", "mizoram",
        "nagaland", "odisha", "punjab", "rajasthan", "sikkim", "tamil nadu",
        "telangana", "tripura", "uttar pradesh", "uttarakhand", "west bengal",
        "delhi", "jammu", "kashmir"
    ]

    STATE_CODES = ["ap","ar","as","br","cg","ga","gj","hr","hp","jh","ka","kl","mp","mh",
                "mn","ml","mz","nl","or","pb","rj","sk","tn","ts","tr","up","uk","wb","dl"]

    mentions_state = any(f" {s} " in q_lower for s in INDIAN_STATES) or \
                    any(f" {c} " in q_lower for c in STATE_CODES)


    all_data = collection.get(include=["documents", "metadatas"])
    all_ids = collection.get()["ids"]

    allowed_docs, allowed_meta, allowed_ids = [], [], []

    for doc, meta, qid in zip(all_data["documents"], all_data["metadatas"], all_ids):
        qid_l = qid.lower()
        if family == "CSO" and "cso" not in qid_l:
           continue
        if family == "PRODUCT" and "product" not in qid_l:
            continue
        if family == "EXPORT" and "export" not in qid_l:
            continue
        if family == "DOMESTIC" and "domestic" not in qid_l:
            continue
        if family == "STATE" and not mentions_state:
            continue
        if family == "CATEGORY" and "category" not in qid_l:
            continue
        
        # GENERAL family: exclude specialized queries that require specific filters
        if family == "GENERAL":
            # Exclude product_segment queries (those are for PRODUCT family)
            if "product_segment" in qid_l:
                continue
            # Exclude state-specific queries unless user mentioned a state
            if "by_state" in qid_l or "state_category" in qid_l:
                if not mentions_state:
                    continue
            # Exclude CSO-specific queries unless user mentioned CSO
            if "by_cso" in qid_l or "cso_category" in qid_l:
                mentions_cso = "cso" in user_query.lower() or any(
                    code in user_query.upper() for code in ["DCBH", "DCMH", "DCRJ"]  # Common CSO prefixes
                )
                if not mentions_cso:
                    continue
        
        # Prefer domestic over export unless explicitly mentioned
        mentions_export = "export" in user_query.lower()
        if not mentions_export and "export" in qid_l and family == "PRODUCT":
            continue

        allowed_docs.append(doc)
        allowed_meta.append(meta)
        allowed_ids.append(qid)

    if not allowed_docs:
        allowed_docs, allowed_meta, allowed_ids = all_data["documents"], all_data["metadatas"], all_ids

    # 2️⃣ Safe Temporary Collection
    temp_name = "router_tmp"

    try:
        CHROMA_CLIENT.delete_collection(temp_name)
    except:
        pass

    temp = CHROMA_CLIENT.create_collection(name=temp_name, embedding_function=EMBEDDINGS)

    try:
        temp.upsert(documents=allowed_docs, metadatas=allowed_meta, ids=allowed_ids)

        results = temp.query(
            query_texts=[user_query],
            n_results=k,
            include=["metadatas"]
        )

    finally:
        try:
            CHROMA_CLIENT.delete_collection(temp_name)
        except:
            pass

    if not results or not results["metadatas"][0]:
        raise ValueError("Routing failed: No SQL query selected")

    match = results["metadatas"][0][0]
    query_id = results["ids"][0][0]

    sql = match["sql_query"]
    params = json.loads(match.get("parameters", "[]"))
    optional = json.loads(match.get("optional_parameters", "[]"))
    defaults = json.loads(match.get("defaults", "{}"))

    print(f"[DEBUG] Semantic Match (ID: {query_id})")
    print(f"[DEBUG] Required params: {params}")
    print(f"[DEBUG] Optional params: {optional}")

    return query_id, sql, params, optional, defaults

def force_delete_all_queries():
    """
    Deletes ALL documents in the sql_query_store collection.
    Used for forced manual re-seeding.
    """
    try:
        collection = CHROMA_CLIENT.get_collection(
            name=COLLECTION_NAME,
            embedding_function=EMBEDDINGS
        )

        count_before = collection.count()
        collection.delete(where={"sql_query": {"$ne": "NEVER_MATCH"}})
        count_after = collection.count()

        print(f"SUCCESS: Deleted {count_before - count_after} entities from ChromaDB collection '{COLLECTION_NAME}'.")

    except Exception as e:
        print(f"ERROR: Failed to delete entities from ChromaDB. Error: {e}")