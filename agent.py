import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, Any, Tuple, List
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
 
from db import semantic_search_sql, execute_sql_query_from_string, get_query_by_id
from schemas.state import ChatState


# --- Currency Formatting Utilities (CRITICAL for accurate financial reporting) ---
def format_indian_currency(amount) -> str:
    """
    Formats a numeric amount into Indian currency format.
    LLMs cannot reliably do this arithmetic, so we MUST do it in Python.
    
    Conversion rules:
    - >= 1,00,00,000 (10 million / 1 crore): Show in Crores (Cr)
    - >= 1,00,000 (100 thousand / 1 lakh): Show in Lakhs (L)
    - >= 1,000: Show in Thousands (K)
    - < 1,000: Show as is
    
    Examples:
        2582531935 → "₹258.25 Cr"  (NOT ₹25.82 Cr!)
        264407334 → "₹26.44 Cr"
        5500000 → "₹55 L"
        85000 → "₹85 K"
        850 → "₹850"
    """
    if amount is None:
        return "N/A"
    
    try:
        amount = float(amount)
    except (ValueError, TypeError):
        return str(amount)
    
    # Handle negative amounts
    is_negative = amount < 0
    abs_amount = abs(amount)
    prefix = "-" if is_negative else ""
    
    # Convert based on magnitude
    if abs_amount >= 10_000_000:  # 1 Crore = 10 million
        formatted = abs_amount / 10_000_000
        if formatted == int(formatted):
            return f"{prefix}₹{int(formatted)} Cr"
        else:
            return f"{prefix}₹{formatted:.2f}".rstrip('0').rstrip('.') + " Cr"
    
    elif abs_amount >= 100_000:  # 1 Lakh = 100 thousand
        formatted = abs_amount / 100_000
        if formatted == int(formatted):
            return f"{prefix}₹{int(formatted)} L"
        else:
            return f"{prefix}₹{formatted:.2f}".rstrip('0').rstrip('.') + " L"
    
    elif abs_amount >= 1_000:  # Thousands
        formatted = abs_amount / 1_000
        if formatted == int(formatted):
            return f"{prefix}₹{int(formatted)} K"
        else:
            return f"{prefix}₹{formatted:.2f}".rstrip('0').rstrip('.') + " K"
    
    else:
        if abs_amount == int(abs_amount):
            return f"{prefix}₹{int(abs_amount)}"
        else:
            return f"{prefix}₹{abs_amount:.2f}".rstrip('0').rstrip('.')


def preformat_currency_in_rows(columns: List[str], rows: List[List]) -> List[List]:
    """
    Pre-formats currency values in the rows data before sending to LLM.
    This is CRITICAL for accurate financial reporting.
    """
    # Keywords that indicate a column contains currency/financial values
    currency_keywords = {
        'sales', 'value', 'amount', 'total', 'revenue', 'invoice', 
        'price', 'cost', 'sum', 'quantity_value', 'lineamount'
    }
    
    # Find indices of columns that likely contain currency values
    currency_column_indices = []
    for idx, col_name in enumerate(columns):
        col_lower = col_name.lower()
        if any(keyword in col_lower for keyword in currency_keywords):
            currency_column_indices.append(idx)
    
    if not currency_column_indices:
        return rows  # No currency columns found
    
    # Format currency values
    formatted_rows = []
    for row in rows:
        new_row = list(row)
        for idx in currency_column_indices:
            if idx < len(new_row):
                value = new_row[idx]
                if isinstance(value, (int, float)) and abs(value) >= 1000:
                    new_row[idx] = format_indian_currency(value)
        formatted_rows.append(new_row)
    
    return formatted_rows


# --- Query Variant Mapping (for dynamic switching) ---
QUERY_VARIANTS = {
    # filtered → base (when user says "all")
    "product_segment_domestic_by_category": "product_segment_domestic",
    "product_segment_export_by_category": "product_segment_export",
    "cso_category_performance": "sales_performance_by_cso",
    "cso_category_performance_export": "sales_performance_by_cso",
    "state_category_performance": "sales_performance_by_state",
    "state_category_performance_export": "sales_performance_by_state",
    # domestic → export (when user says "export")
    "product_segment_domestic|export": "product_segment_export",
    "product_segment_domestic_by_category|export": "product_segment_export_by_category",
    "cso_category_performance|export": "cso_category_performance_export",
    "state_category_performance|export": "state_category_performance_export",
    # base → category+export (when user adds category + export to base query)
    "sales_performance_by_cso|export": "cso_category_performance_export",
    "sales_performance_by_state|export": "state_category_performance_export",
    # export → domestic (when user says "domestic")
    "product_segment_export|domestic": "product_segment_domestic",
    "product_segment_export_by_category|domestic": "product_segment_domestic_by_category",
    "cso_category_performance_export|domestic": "cso_category_performance",
    "state_category_performance_export|domestic": "state_category_performance",
}

# --- Query Upgrades (when filter is added to base query) ---
# Maps (base_query, filter_type) to upgraded query
# This ensures product queries stay as product queries when filters are added
QUERY_UPGRADES = {
    # Product segment + category → Product segment by category
    "product_segment_domestic|business_category": "product_segment_domestic_by_category",
    "product_segment_export|business_category": "product_segment_export_by_category",
    
    # Base salesperson + category → Category performance
    "top_salesperson_flexible_period|business_category": "general_category_performance",
    "executive_sales_performance_period|business_category": "general_category_performance",
    
    # Base salesperson + state → State performance
    "top_salesperson_flexible_period|state_id": "sales_performance_by_state",
    "executive_sales_performance_period|state_id": "sales_performance_by_state",
    
    # State performance + category → State category performance
    "sales_performance_by_state|business_category": "state_category_performance",
    
    # Base salesperson + cso → CSO performance
    "top_salesperson_flexible_period|cso_id": "sales_performance_by_cso",
    "executive_sales_performance_period|cso_id": "sales_performance_by_cso",
    
    # CSO performance + category → CSO category performance
    "sales_performance_by_cso|business_category": "cso_category_performance",
    
    # Category performance + state → State category performance
    "general_category_performance|state_id": "state_category_performance",
    
    # Category performance + cso → CSO category performance  
    "general_category_performance|cso_id": "cso_category_performance",
}

# --- Query Parameter Support Config ---
# Maps each query ID to the set of filter params it supports
# This enables automatic query selection based on collected parameters
QUERY_SUPPORTED_PARAMS = {
    # Base queries (no filters)
    "top_salesperson_flexible_period": {"start_date", "end_date", "sort", "n"},
    "executive_sales_performance_period": {"start_date", "end_date", "sort", "n"},
    # "monthly_sales_breakdown": {"business_category", "start_date", "end_date"},
    
    # Single filter queries
    "sales_performance_by_state": {"state_id", "start_date", "end_date", "sort", "n"},
    "sales_performance_by_cso": {"cso_id", "start_date", "end_date", "sort", "n"},
    "sales_performance_by_cluster": {"cluster_id", "sort", "n"},
    "general_category_performance": {"business_category", "start_date", "end_date", "sort", "n"},
    "domestic_category_specific": {"business_category", "start_date", "end_date", "sort", "n"},
    "export_category_specific": {"business_category", "start_date", "end_date", "sort", "n"},
    
    # Combined filter queries (state + category)
    "state_category_performance": {"state_id", "business_category", "start_date", "end_date", "sort", "n"},
    "state_category_performance_export": {"state_id", "business_category", "start_date", "end_date", "sort", "n"},
    
    # Combined filter queries (cso + category)
    "cso_category_performance": {"cso_id", "business_category", "start_date", "end_date", "sort", "n"},
    "cso_category_performance_export": {"cso_id", "business_category", "start_date", "end_date", "sort", "n"},
    
    # Product segment queries
    "product_segment_domestic": {"start_date", "end_date"},
    "product_segment_domestic_by_category": {"business_category", "start_date", "end_date"},
    "product_segment_export": {"start_date", "end_date"},
    "product_segment_export_by_category": {"business_category", "start_date", "end_date"},
}

# Filter params that require query switching (location/category filters)
FILTER_PARAMS = {"state_id", "cso_id", "cluster_id", "business_category"}


def find_best_query_for_params(collected_params: dict, current_query_id: str = None) -> str:
    """
    Find the best query that supports ALL filter params in collected_params.
    
    First checks QUERY_UPGRADES for explicit mappings (preserves query type).
    Then falls back to finding any compatible query.
    
    Args:
        collected_params: Dict of all collected parameters
        current_query_id: Current query ID (used as fallback if no better match)
    
    Returns:
        Query ID that supports all filters, or current_query_id if none found
    """
    # Get filter params that user has provided
    user_filters = set(collected_params.keys()) & FILTER_PARAMS
    
    if not user_filters:
        return current_query_id  # No filter params, keep current query
    
    # PRIORITY 1: Check QUERY_UPGRADES for explicit type-preserving mappings
    # This ensures product queries stay as product queries, etc.
    if current_query_id:
        for filter_param in user_filters:
            upgrade_key = f"{current_query_id}|{filter_param}"
            if upgrade_key in QUERY_UPGRADES:
                upgraded_query = QUERY_UPGRADES[upgrade_key]
                print(f"[DEBUG] UPGRADE: '{current_query_id}' + {filter_param} → '{upgraded_query}'")
                return upgraded_query
    
    # PRIORITY 2: Check if current query already supports all filters
    if current_query_id and current_query_id in QUERY_SUPPORTED_PARAMS:
        current_supported = QUERY_SUPPORTED_PARAMS[current_query_id]
        if user_filters.issubset(current_supported):
            # Current query already works
            return current_query_id
    
    # PRIORITY 3: Find any query that supports all user's filters
    candidates = []
    for qid, supported in QUERY_SUPPORTED_PARAMS.items():
        if user_filters.issubset(supported):
            # Score by how many extra params it supports (prefer simpler queries)
            extra_filters = len(supported & FILTER_PARAMS) - len(user_filters)
            candidates.append((qid, extra_filters))
    
    if candidates:
        # Check if export is requested
        is_export = (current_query_id and 'export' in current_query_id.lower()) or \
                    collected_params.get('sales_type') == 'export'
        
        if is_export:
            # Prefer queries with 'export' in the name
            export_candidates = [(qid, score) for qid, score in candidates if 'export' in qid.lower()]
            if export_candidates:
                best = min(export_candidates, key=lambda x: x[1])
                return best[0]
        
        # Return query with fewest extra filter params (most specific match)
        best = min(candidates, key=lambda x: x[1])
        return best[0]
    
    return current_query_id  # No match found, keep current

 
# --- Configuration ---
LLM = ChatGroq(
    temperature=0.0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="moonshotai/kimi-k2-instruct-0905"
)
DOMAIN_DECLINE_MESSAGE = "I am a sales data assistant. How can I assist you with a sales data query?"
 
 
# --- 1. Intent Router Logic ---
INTENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """
     You are an intent router. Analyze the user's question and output ONE keyword only.
     
     **OUTPUT KEYWORDS (choose exactly one):**
     1. GREETING: ONLY explicit greetings like 'Hello', 'Hi', 'Hey there', 'Good morning'
     2. SALES: Sales data questions OR acknowledgments like 'nice', 'ok', 'good', 'thanks', 'great', 'cool'
     3. TABLE: Request to see data in table format (e.g., 'show table', 'in markdown', 'display as table')
     4. RESET: User wants to start over (e.g., 'Start over', 'Reset', 'New question', 'clear')
     5. REJECT: Outside sales domain or dangerous operations
     
     IMPORTANT:
     - 'nice', 'ok', 'good', 'thanks', 'great' → SALES (not GREETING!)
     - Numbers, dates, short answers → SALES
     - Business categories like 'wiring', 'switches', 'cables', 'FMEG', 'export', 'domestic' → SALES
     - Any follow-up that mentions dates, categories, or filters → SALES
     """
    ),
    ("human", "{question}")
])
INTENT_CHAIN = INTENT_PROMPT | LLM | StrOutputParser()
 


# --- 2. Context Analyzer Chain (For Follow-up Detection & Parameter Inheritance) ---
CONTEXT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """
     You are a conversation context analyzer for a sales data chatbot.
     
     **CURRENT DATE: {current_date}**
     
     **CONVERSATION HISTORY:**
     {conversation_history}
     
     **LAST SUCCESSFUL QUERY:**
     - Question: "{last_question}"
     - Query ID: {last_query_id}
     - Parameters used: {last_params}
     
     **CURRENT USER MESSAGE:** "{current_message}"
     
     **TASK:** Determine query type and which parameters to inherit from history.
     
     **CLASSIFICATION RULES:**
      
      ⚠️⚠️⚠️ CRITICAL RULE - CHECK THIS FIRST! ⚠️⚠️⚠️
      IF last_query_id contains "product_segment" AND current message has "salesperson"/"who"/"rank" → NEW_QUERY!
      IF last_query_id contains "salesperson"/"category_performance" AND current message has "product type"/"product segment" → NEW_QUERY!
      These are ALWAYS NEW_QUERY, never FOLLOW_UP!
      ⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️
      
      0. **ACKNOWLEDGMENT**: User is simply acknowledging or reacting (NOT requesting data)
        - Simple reactions: "nice", "ok", "good", "great", "thanks", "cool", "awesome", "got it"
        - Confirmations: "understood", "I see", "perfect", "alright"
        → OUTPUT: {{"query_type": "ACKNOWLEDGMENT", "inherit_params": [], "override_params": {{}}}}
     
     1. **CLARIFICATION_QUESTION**: User is ASKING ABOUT what was just shown (NOT requesting new data)
        - "highest or lowest?", "was that the top?", "its highest or lowest?"
        - "which one was that?", "so that's the best?"
        - Questions ABOUT the previous result without requesting new data
        → OUTPUT: {{"query_type": "CLARIFICATION_QUESTION", "inherit_params": [], "override_params": {{}}}}
     
     2. **FOLLOW_UP**: User is modifying or continuing the SAME TYPE of query
        - Changing just one parameter: "now top 5", "for FMEG", "in Rajasthan"
        - Asking for next result: "second highest", "who's next", "another one"
        - REQUESTING NEW DATA: "what was the lowest?", "show me the highest", "and the top one?"
        - Same query, different scope: "same for last month", "also in export"
        - Short implicit questions: "bottom?", "in Bihar?", "top 10?", "last quarter"
     
     3. **NEW_QUERY**: User is asking a DIFFERENT TYPE of question
        - Query subject changed: products → people, CSO → state, etc.
        - "Who" or "which salesperson" when previous was about products = NEW_QUERY
        - "Which product" when previous was about salespeople = NEW_QUERY
        - Different query type: was asking about sales, now asking about CSOs
        - IMPORTANT: If user doesn't mention a TIME PERIOD, still inherit dates!
     
     **QUERY SUBJECT DETECTION:**
     - "Who", "salesperson", "executive", "sales rep" → PERSON query
     - "Which product", "product type", "item" → PRODUCT query
     - "CSO", "cluster" → LOCATION/CSO query
      - CRITICAL: If previous was PERSON/CSO query and current asks about "product" = NEW_QUERY (NOT FOLLOW_UP!)
      - ⚠️ IMPORTANT: "product segment", "product type" after CSO/salesperson query = ALWAYS NEW_QUERY!
      - ⚠️ NEVER classify product questions as FOLLOW_UP to CSO/salesperson queries!
      - ⚠️ CRITICAL: "salesperson", "who", "rank" after product_segment query = ALWAYS NEW_QUERY!
     
     4. **CLARIFICATION**: User is providing missing info for a pending query
        - Single value response: "5", "RJC01", "last month"
        - Only relevant if there was a pending question
     
     **CRITICAL - DATE CALCULATION (use current_date: {current_date}):**
     
     When user mentions relative dates, calculate based on CURRENT DATE:
     - "last quarter" → Previous complete quarter according to the current date (if Jan-Mar, use Oct-Dec of prev year)
     - "last month" → Previous complete calendar month according to the current date
     - "this year" → Jan 1 to Dec 31 of current year
     - "last year" → Jan 1 to Dec 31 of previous year
     
     **DATE FORMAT:** Always use YYYY-MM-DD format.
     
     **PARAMETER EXTRACTION RULES:**
     
     **SORT DIRECTION SYNONYMS:**
     - HIGH/DESC: "top", "highest", "best", "greatest", "most", "maximum", "leading", "top-performing"
     - LOW/ASC: "bottom", "lowest", "worst", "least", "minimum", "fewest", "poorest", "bottom-performing"
     
     **"TOP N" / "BOTTOM N" PATTERNS (CRITICAL):**
     - "top 3", "best 5", "highest 10" → override: {{n: <number>, sort: "DESC"}}
     - "bottom 2", "worst 5", "lowest 3" → override: {{n: <number>, sort: "ASC"}}
     - ALWAYS include BOTH n AND sort when user specifies count with direction!
     
     **SINGLE-WORD MODIFIERS:**
     - "lowest", "worst", "bottom" → override: {{sort: "ASC"}}
     - "highest", "best", "top" → override: {{sort: "DESC"}}
     
     **DATES (start_date, end_date):**
     - Inherit dates UNLESS user mentions a new time period
     - "last quarter", "this month", "Dec 2024" = NEW dates, calculate in override_params
     - No date mentioned = INHERIT from history
     
     **OTHER PARAMETERS:**
      - Location (state_id, cluster_id, cso_id): Only inherit for FOLLOW_UP
      - Category (business_category): ALWAYS inherit for FOLLOW_UP if previous had it
      - Sales type (sales_type): ALWAYS inherit for FOLLOW_UP if previous had it (export/domestic)
      - CRITICAL: When changing only CSO/state, KEEP the business_category and sales_type from previous!
     
     **BUSINESS CATEGORY EXTRACTION (ALWAYS extract when mentioned):**
     - "wiring", "switches", "switchgear" → business_category: "'Wiring Devices & Switchgear'"
     - "FMEG", "fast moving" → business_category: "'FMEG'"
     - "cables", "wires", "W&C" → business_category: "'Wires & Cables'"
     - "all", "all categories", "everything" → business_category: "'FMEG', 'Wiring Devices & Switchgear', 'Wires & Cables'"
     - ALWAYS wrap the value in SINGLE QUOTES for SQL
      - CRITICAL: "for Wires", "Wires export", "FMEG domestic" = EXTRACT business_category!
     
     **STATE_ID CONVERSION (CRITICAL):**
     Always convert Indian state names to 2-letter state codes:
     - Andhra Pradesh → AP, Arunachal Pradesh → AR, Assam → AS, Bihar → BR
     - Chhattisgarh → CG, Goa → GA, Gujarat → GJ, Haryana → HR
     - Himachal Pradesh → HP, Jharkhand → JH, Karnataka → KA, Kerala → KL
     - Madhya Pradesh → MP, Maharashtra → MH, Manipur → MN, Meghalaya → ML
     - Mizoram → MZ, Nagaland → NL, Odisha → OD, Punjab → PB
     - Rajasthan → RJ, Sikkim → SK, Tamil Nadu → TN, Telangana → TS
     - Tripura → TR, Uttar Pradesh → UP, Uttarakhand → UK, West Bengal → WB
     - Delhi → DL, Jammu and Kashmir → JK, Ladakh → LA, Chandigarh → CH
     
     **EXAMPLES (current date is {current_date}):**
     
     0. Previous: "Lowest salesperson Nov 2025"
        Current: "nice" or "ok" or "thanks"
        → {{"query_type": "ACKNOWLEDGMENT"}} (just acknowledging, no data request)
     
     1. Previous: "Lowest salesperson Nov 2025"
        Current: "highest or lowest?"
        → {{"query_type": "CLARIFICATION_QUESTION"}} (asking ABOUT what was shown)
     
     2. Previous: "Highest sales January 2025"
        Current: "bottom 2" or "what was the lowest?"
        → FOLLOW_UP with override: {{n: 2, sort: "ASC"}}, inherit: [start_date, end_date]
     
     3. Previous: "bottom 2 performers"
        Current: "top 3"
        → FOLLOW_UP with override: {{n: 3, sort: "DESC"}}, inherit: [start_date, end_date, state_id]
     
     4. Previous: "Highest product type"
        Current: "for last quarter"
        → FOLLOW_UP with override: {{start_date: "2025-10-01", end_date: "2025-12-31"}}, inherit: [sort, n]
      
      5. CRITICAL - Previous: "top 5 salespersons under CSO DCBH01"
         Current: "Which product type generated highest invoice value"
         → NEW_QUERY (NOT FOLLOW_UP! Subject changed from PERSON to PRODUCT)
      
     **OUTPUT FORMAT (JSON only, no extra text):**
     {{
       "query_type": "ACKNOWLEDGMENT" | "CLARIFICATION_QUESTION" | "FOLLOW_UP" | "NEW_QUERY" | "CLARIFICATION",
       "confidence": "HIGH" | "MEDIUM" | "LOW",
       "reasoning": "One line explanation",
       "inherit_params": ["list", "of", "param", "names", "to", "carry", "forward"],
       "override_params": {{"param_name": "new_value_if_mentioned"}}
     }}
     """
    ),
    ("human", "{current_message}")
])
CONTEXT_CHAIN = CONTEXT_PROMPT | LLM | StrOutputParser()


# --- 3. Enhanced Parameter Extraction (Context-Aware) ---
PARAMETER_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """
     You are a parameter extractor for a Sales ERP system.
     
     **CONTEXT FROM CONVERSATION:**
     - Inherited parameters (from previous query): {inherited_params}
     - Override hints (from context analysis): {override_hints}
     
     **CURRENT STATE:**
     - Missing parameters: {missing_params}
     - Already collected: {collected_params}
     - Optional parameters: {optional_params}
     - Original question: {original_question}
     - Current date: {current_date}
     
     **EXTRACTION RULES:**
     1. 'n': Extract number. "best/top/who is" → 1; "top 5" → 5; "second" → 2; "third" → 3; "bottom 10" → 10
     2. 'sort': "Top/Best/Highest/Most/greatest" → "DESC" ; "Bottom/Worst/Lowest/Least" → "ASC"
     3. 'start_date'/'end_date': Convert to 'YYYY-MM-DD' ONLY for SPECIFIC dates:
         - VAGUE phrases like "last period", "recently", "previous period" → return null
         - ONLY extract dates for CLEAR references like:
      3a. 'start_date'/'end_date': Convert to 'YYYY-MM-DD'
        - "last month" → Previous full calendar month
        - "last {{m}} months" → 1st day of the month {{m}} months ago to today
        - "last quarter" → Previous 3-month period  
        - "this year" → Jan 1 to Dec 31 of current year
        - If a month is mentioned without year, use most recent past occurrence
        - For dates: Always interpret months/years relative to the current date ({current_date})
     4. 'cso_id': Look for codes (e.g., 'DCBH01') OR names (e.g., 'Mahesh Kumar')
     5. 'state_id': ALWAYS convert state names to 2-letter codes:
        AP=Andhra Pradesh, AR=Arunachal, AS=Assam, BR=Bihar, CG=Chhattisgarh,
        GA=Goa, GJ=Gujarat, HR=Haryana, HP=Himachal Pradesh, JH=Jharkhand,
        KA=Karnataka, KL=Kerala, MP=Madhya Pradesh, MH=Maharashtra, MN=Manipur,
        ML=Meghalaya, MZ=Mizoram, NL=Nagaland, OD=Odisha, PB=Punjab, RJ=Rajasthan,
        SK=Sikkim, TN=Tamil Nadu, TS=Telangana, TR=Tripura, UP=Uttar Pradesh,
        UK=Uttarakhand, WB=West Bengal, DL=Delhi, JK=J&K, CH=Chandigarh
     6. 'cluster_id': Look for codes (e.g., 'RJC01', 'MHC05')
     7. 'business_category': Can be a SINGLE value or a LIST of values for SQL IN() clause
        - "FMEG" or "fast moving" → "'FMEG'"
        - "W&C" or "wires" or "cables" → "'Wires & Cables'"
        - "Wiring Devices" or "switchgear" → "'Wiring Devices & Switchgear'"
        - MULTIPLE categories: "FMEG and Wires" → "'FMEG', 'Wires & Cables'"
        - ALWAYS wrap each category in SINGLE QUOTES for SQL
        - If user says "all categories" or doesn't specify, return null
     
     **PRIORITY ORDER:**
     1. Values explicitly in CURRENT user message (highest priority)
     2. Values from override_hints (user intent from context)
     3. Values from inherited_params (carry forward from history)
     4. Values already in collected_params
     5. null if cannot determine
     
     **CRITICAL:**
     - DO NOT overwrite explicitly provided values with inherited ones
     - If user says "second highest", set n=2 but inherit other params
     - If user says "same for FMEG", only change business_category, inherit rest
     - Be lenient and try to infer from context
     
     **Output:** Valid JSON only, no text before or after.
     Example (single): {{"n": 5, "sort": "DESC", "business_category": "'FMEG'"}}
      Example (multiple): {{"business_category": "'FMEG', 'Wires & Cables'"}}
     """
    ),
    ("human", "Current user message: {question}")
])
PARAMETER_EXTRACTION_CHAIN = PARAMETER_EXTRACTION_PROMPT | LLM | StrOutputParser()
 
 
# --- 3. Main RAG Prompt ---
# SYSTEM_PROMPT = """
# You are an intelligent sales data assistant for Microsoft Fabric database.
# Interpret SQL query results and provide clear, concise answers.
 
# **Rules:**
# 1. DO NOT mention SQL, columns, or raw data to the user
# 2. Start with a clear summary of findings
# 3. If rows are empty, say no data was found
# 4. DEFAULT: Provide conversational summary in plain English
# 5. Name key entities from the data (list actual names, values)
# 6. DO NOT ask if user wants table format - just summarize
# 7. **CRITICAL - Currency Formatting:**
#    - Always format amounts in Indian numbering system
#    - DO NOT add "Cr" or "L" at the end of a full amount.
#    - NEVER show decimal points. Round all values to the nearest whole number
#    - Use: Crores (Cr) for ≥1,00,00,000 | Lakhs (L) for ≥1,00,000 | Thousands (K) for ≥1,000 only when to shorten the bigger amount with some whole no. in Cr value. The context must not be hindered taht has been selected.
#    - Examples: ₹2.5 Cr, ₹45 L, ₹8.2 K, ₹500 and not ₹500,90,291 Cr
#    - For amounts in summaries, always use this format
#    - When presenting multiple results, format them clearly with names and their corresponding values
 
# Raw Data:
# Columns: {columns}
# Rows: {rows}
# """

# Replace your SYSTEM_PROMPT with this updated version:

SYSTEM_PROMPT = """
You are an intelligent sales data assistant for Microsoft Fabric database.
Interpret SQL query results and provide clear, concise answers.

**Rules:**
Give full amount as retreived after processing query.
1. DO NOT mention SQL, columns, or raw data to the user
2. Start with a clear summary of findings, donot hinder the order in which sql answer was retreived
Full amount captured on querying is to be shown in the summary in the Indian currency format mentioned in point 7
3. DEFAULT: Provide conversational summary in plain English
4. Name key entities from the data (list actual names, values)
5. DO NOT ask if user wants table format - just summarize
6. for answering the product based questions also mention the business category retrieved

7. **CRITICAL - Currency Formatting (MANDATORY):**
   You MUST convert ALL large amounts into format mentioned below. Follow these rules STRICTLY:
   
   **Step 1: Handle negative amounts**
   - If amount is negative, take the absolute value for calculation
   - Add "-" or "negative" before the formatted amount
   - Example: -264407334 → "-₹26.44 Cr" or "negative ₹26.44 Cr"
   
   **Step 2: Convert based on absolute value**
   - Amounts ≥ 1,00,00,000 (10 million): Convert to Crores (Cr)
     Examples: 
     • 264407334 → "₹26.44 Cr"
     • -264407334 → "-₹26.44 Cr"
     • 12500000 → "₹1.25 Cr"
   
   - Amounts ≥ 1,00,000 (100 thousand) but < 1 crore: Convert to Lakhs (L)
     Examples:
     • 5500000 → "₹55 L"
     • -450000 → "-₹4.5 L"
   
   - Amounts ≥ 1,000 but < 1 lakh: Convert to Thousands (K)
     Examples:
     • 85000 → "₹85 K"
     • -25000 → "-₹25 K"
   
   - Amounts < 1,000: Show as is with ₹ symbol
     Examples:
     • 850 → "₹850"
     • -500 → "-₹500"
   
   **Decimal Precision:**
   - Use maximum 2 decimal places
   - Drop unnecessary zeros: ₹25.00 Cr → ₹25 Cr
   - Keep significant decimals: ₹26.44 Cr (keep as is)
   
   **COMPLETE EXAMPLE:**
   If you see data: [['Jatin Patel', 264407334], ['Amit Kumar', -5500000], ['Raj Shah', 850]]
   
   You should say:
   "Jatin Patel generated the highest value at ₹26.44 Cr, while Amit Kumar had a negative value of -₹55 L, and Raj Shah contributed ₹850."
   
   NOT:
   "Jatin Patel generated ₹264407334..." ← WRONG!
   
   **EXAMPLES OF CORRECT FORMATTING:**
   ✓ 2644073344 → "₹264.40 Cr" or "264.40 crores"
   ✓ 264407334 → "₹26.44 Cr" or "26.44 crores"
   ✓ -264407334 → "-₹26.44 Cr" or "negative 26.44 crores"
   ✓ 12500000 → "₹1.25 Cr"
   ✓ -5500000 → "-₹55 L" 
   ✓ 450000 → "₹4.5 L"
   ✓ -25000 → "-₹25 K"
   ✓ 850 → "₹850"
   ✓ -500 → "-₹500"
   
   **WRONG - NEVER DO THIS:**
   ✗ ₹264407334 (raw large number)
   ✗ ₹2,64,40,73,341 (comma separated but not converted)
   ✗ 26.4407334 Cr (too many decimals)
   ✗ -₹-26.44 Cr (double negative)

8. **When presenting multiple results:**
   - List each person/entity with their formatted value
   - Maintain the sort order from the query results
   - Keep negative values visible throughout

9. **CRITICAL - Understand the question context:**
   - If user asked for "lowest", "bottom", "worst", "least" → say "lowest" not "highest"
   - If user asked for "highest", "top", "best" → say "highest" or "top"
   - If user asked for a specific ordinal like "5th lowest" or "2nd highest":
     * ONLY report that specific position (the last row in results)
     * Example: "5th lowest" with 5 rows → only mention the 5th person
     * Example: "2nd highest" with 2 rows → only mention the 2nd person
   
   **ORDINAL EXAMPLES:**
   - Question: "5th lowest" → Answer: "The 5th lowest performer is [Name] with [Value]"
   - Question: "2nd highest" → Answer: "The 2nd highest performer is [Name] with [Value]"
   - Question: "top 5" → Answer: List all 5, first is highest
   - Question: "bottom 3" → Answer: List all 3, first is lowest

Raw Data:
Columns: {columns}
Rows: {rows}

Query Context (IMPORTANT: Mention these filters naturally in your response - state, category, period, sales type):
{query_context}
"""

# Also update the TABLE formatting prompt:
 
HUMAN_PROMPT = "Original question: {question}"
 
AGENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", HUMAN_PROMPT)
])
 
def input_format(input_dict: dict) -> dict:
    return {
        "columns": input_dict.get("columns", "[]"),
        "rows": input_dict.get("rows", "[]"),
        "question": input_dict["question"],
        "query_context": input_dict.get("query_context", "{}")
    }
 
 
# --- 4. Helper Functions ---
 
def get_parameter_guidance() -> Dict[str, str]:
    """Returns user-friendly examples for each parameter type."""
    return {
        "n": "a number (e.g., 'top 5', 'best 10', or just '5')",
        "sort": "sorting direction (e.g., 'highest', 'lowest', 'top', 'bottom')",
        "start_date": "start date (e.g., 'last month', 'January 1 2024', '2024-01-01')",
        "end_date": "end date (e.g., 'today', 'December 31 2024', '2024-12-31')",
        "cluster_id": "cluster code (e.g., 'RJC01', 'MHC05')",
        "cso_id": "CSO ID code (e.g., 'CSO001')",
        "state_id": "state code (e.g., 'BH', 'RJ', 'MH')",
        "business_category": "business unit (e.g., 'FMEG', 'Wires & Cables', 'Wiring Devices & Switchgear')"
    }
 
 
def calculate_date_from_placeholder(placeholder: str) -> str:
    """
    Converts date placeholder tokens to actual YYYY-MM-DD dates.
   
    Placeholders:
    - __LAST_MONTH_START__: First day of last month
    - __LAST_MONTH_END__: Last day of last month
    - __THIS_MONTH_START__: First day of current month
    - __THIS_MONTH_END__: Last day of current month
    - __LAST_QUARTER_START__: First day of last quarter
    - __LAST_QUARTER_END__: Last day of last quarter
    """
    today = datetime.now()
   
    if placeholder == "__LAST_MONTH_START__":
        first_of_last_month = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
        return first_of_last_month.strftime("%Y-%m-%d")
   
    elif placeholder == "__LAST_MONTH_END__":
        first_of_this_month = today.replace(day=1)
        last_of_last_month = first_of_this_month - timedelta(days=1)
        return last_of_last_month.strftime("%Y-%m-%d")
   
    elif placeholder == "__THIS_MONTH_START__":
        return today.replace(day=1).strftime("%Y-%m-%d")
   
    elif placeholder == "__THIS_MONTH_END__":
        next_month = today.replace(day=28) + timedelta(days=4)
        last_of_this_month = next_month - timedelta(days=next_month.day)
        return last_of_this_month.strftime("%Y-%m-%d")
   
    elif placeholder == "__LAST_QUARTER_START__":
        current_quarter = (today.month - 1) // 3
        if current_quarter == 0:
            # Last quarter is Q4 of previous year
            last_q_start = datetime(today.year - 1, 10, 1)
        else:
            last_q_start = datetime(today.year, (current_quarter - 1) * 3 + 1, 1)
        return last_q_start.strftime("%Y-%m-%d")
   
    elif placeholder == "__LAST_QUARTER_END__":
        current_quarter = (today.month - 1) // 3
        if current_quarter == 0:
            # Last quarter is Q4 of previous year
            last_q_end = datetime(today.year - 1, 12, 31)
        else:
            last_q_end_month = current_quarter * 3
            last_q_end = (datetime(today.year, last_q_end_month, 28) + timedelta(days=4))
            last_q_end = last_q_end - timedelta(days=last_q_end.day)
        return last_q_end.strftime("%Y-%m-%d")
   
    # If not a placeholder, return as-is
    return placeholder
 
 
# def format_missing_params_message(missing: List[str], attempt: int, max_attempts: int) -> str:
#     """Creates user-friendly message for missing parameters."""
#     guidance = get_parameter_guidance()
   
#     missing_with_examples = [
#         f"• **{param}**: {guidance.get(param, 'please specify')}"
#         for param in missing
#     ]
   
#     return (
#         f"I need the following information (Attempt {attempt}/{max_attempts}):\n\n"
#         f"{chr(10).join(missing_with_examples)}\n\n"
#         f"💡 You can say 'skip' for optional fields or 'start over' to begin again."
#     )

def format_missing_params_message(missing: List[str], attempt: int, max_attempts: int) -> str:
    """Creates user-friendly conversational message for missing parameters."""
    guidance = get_parameter_guidance()
    
    # Build conversational prompt based on what's missing
    if len(missing) == 1:
        param = missing[0]
        example = guidance.get(param, 'please specify')
        
        # Single parameter - very natural phrasing
        prompts = {
            "start_date": f"For which time period? You can say something like 'last month', 'this quarter', or give me specific dates.",
            "end_date": f"Until when? You can say 'today', 'end of last month', or a specific date.",
            "n": f"How many results would you like to see? For example, 'top 5' or just '10'.",
            "state_id": f"Which state are you interested in? Please provide the state code (e.g., 'RJ', 'MH', 'BH').",
            "cluster_id": f"Which cluster? Please provide the cluster code (e.g., 'RJC01', 'MHC05').",
            "cso_id": f"Which CSO are you looking for? Please provide the CSO ID.",
            "business_category": f"Which business unit? You can say 'FMEG', 'Wires & Cables', 'Switchgear' or their export variant."
        }
        
        return prompts.get(param, f"Could you specify the {param.replace('_', ' ')}? ({example})")
    
    elif len(missing) == 2:
        # Two parameters - natural conjunction
        param1, param2 = missing[0], missing[1]
        
        # Special case for date ranges
        if set(missing) == {"start_date", "end_date"}:
            return "Which time period should I look at?"
        
        # General case for two params
        example1 = guidance.get(param1, 'please specify')
        example2 = guidance.get(param2, 'please specify')
        return f"I need two more details: the {param1.replace('_', ' ')} ({example1}) and the {param2.replace('_', ' ')} ({example2})."
    
    else:
        # Three or more parameters - still conversational but clearer
        param_list = []
        for param in missing:
            example = guidance.get(param, 'please specify')
            param_list.append(f"the {param.replace('_', ' ')} ({example})")
        
        if len(missing) == 3:
            return f"I need a few more details: {param_list[0]}, {param_list[1]}, and {param_list[2]}."
        else:
            items = ", ".join(param_list[:-1])
            return f"I need some more information: {items}, and {param_list[-1]}."
 
 
def extract_from_original_question(original_question: str, param: str) -> Any:
    """
    Tries to extract parameter directly from the original question.
    This is a fallback when LLM extraction fails.
    """
    upper_q = original_question.upper()
   
    # Extract 'n' from phrases like "top 3", "bottom 5", "best 10", "top 5 performers"
    if param == "n":
        import re
        patterns = [
            # Pattern 1: "top 5", "bottom 3", "best 10" (with optional words after)
            r'\b(?:TOP|BOTTOM|BEST|WORST|FIRST|LAST|MINIMUM|MAXIMUM)\s+(\d+)',
            # Pattern 2: "5 top performers", "10 salespersons"
            r'\b(\d+)\s+(?:TOP|BOTTOM|BEST|WORST|SALESPERSON|SALES|EXECUTIVE|PERFORMER|PERFORMING)'
        ]
        for pattern in patterns:
            match = re.search(pattern, upper_q)
            if match:
                return int(match.group(1))
   
    # Extract 'sort' from top/bottom keywords (separate from 'n' extraction)
    if param == "sort":
        import re
        # Bottom/Worst keywords → ASC (lowest first)
        # Use word boundaries to avoid matching "MIN" in "PERFORMING"
        asc_pattern = r'\b(BOTTOM|WORST|LOWEST|LEAST|MINIMUM|POOREST|WEAKEST)\b'
        if re.search(asc_pattern, upper_q):
            return "ASC"
        # Top/Best keywords → DESC (highest first)
        desc_pattern = r'\b(TOP|BEST|HIGHEST|MOST|MAXIMUM|GREATEST|STRONGEST|LARGEST)\b'
        if re.search(desc_pattern, upper_q):
            return "DESC"
   
    # Extract business_category (can be multiple - format for SQL IN clause)
    if param == "business_category":
        categories = []
        if "FMEG" in upper_q or "FAST MOVING" in upper_q:
            categories.append("'FMEG'")
        if any(word in upper_q for word in ["W&C", "WIRES", "CABLES", "WIRE AND CABLE"]):
            categories.append("'Wires & Cables'")
        if any(word in upper_q for word in ["WIRING DEVICES", "SWITCHGEAR", "SWITCHES", "SWITCH"]):
            categories.append("'Wiring Devices & Switchgear'")
        # Note: "Export" here is a category filter, not the export sales type
        # Don't extract Export as business_category from question - it's handled by query routing
        if categories:
            return ", ".join(categories)
   
    # Extract cluster_id, cso_id, state_id (alphanumeric codes)
    if param in ["cluster_id", "cso_id", "state_id"]:
        import re
        
        # STATE NAME TO CODE MAPPING (for state_id extraction)
        if param == "state_id":
            state_name_map = {
                'rajasthan': 'RJ', 'gujarat': 'GJ', 'maharashtra': 'MH',
                'delhi': 'DL', 'karnataka': 'KA', 'tamil nadu': 'TN',
                'kerala': 'KL', 'andhra pradesh': 'AP', 'telangana': 'TS',
                'uttar pradesh': 'UP', 'madhya pradesh': 'MP', 'punjab': 'PB',
                'haryana': 'HR', 'west bengal': 'WB', 'bihar': 'BR',
                'odisha': 'OR', 'jharkhand': 'JH', 'chhattisgarh': 'CG',
                'assam': 'AS', 'goa': 'GA', 'himachal': 'HP',
                'uttarakhand': 'UK', 'jammu': 'JK', 'gujrat': 'GJ',  # Common misspelling
            }
            lower_q = original_question.lower()
            for state_name, state_code in state_name_map.items():
                if state_name in lower_q:
                    return state_code
        
        # PRIORITY 1: Look for codes with numbers (more specific): RJC01, MHC05, CSO001
        codes_with_numbers = re.findall(r'\b[A-Z]{2,4}\d+\b', upper_q)
        if codes_with_numbers:
            return codes_with_numbers[0]
       
        # PRIORITY 2: Look for 2-letter state codes NOT followed by common words
        # Exclude common words like: IN, ON, BY, TO, AT, OR, SO, AS, IS
        exclude_words = {'IN', 'ON', 'BY', 'TO', 'AT', 'OR', 'SO', 'AS', 'IS', 'OF', 'AN', 'IF', 'IT'}
        codes_two_letter = re.findall(r'\b([A-Z]{2})\b', upper_q)
        for code in codes_two_letter:
            if code not in exclude_words:
                return code

    # Extract start_date / end_date from relative time expressions
    if param in ["start_date", "end_date"]:
        lower_q = original_question.lower()
        today = datetime.now()
        
        # Last quarter
        if "last quarter" in lower_q or "previous quarter" in lower_q:
            current_quarter = (today.month - 1) // 3
            if current_quarter == 0:
                # Last quarter is Q4 of previous year
                if param == "start_date":
                    return datetime(today.year - 1, 10, 1).strftime("%Y-%m-%d")
                else:  # end_date
                    return datetime(today.year - 1, 12, 31).strftime("%Y-%m-%d")
            else:
                last_q_start_month = (current_quarter - 1) * 3 + 1
                last_q_end_month = current_quarter * 3
                if param == "start_date":
                    return datetime(today.year, last_q_start_month, 1).strftime("%Y-%m-%d")
                else:  # end_date
                    last_q_end = datetime(today.year, last_q_end_month, 28) + timedelta(days=4)
                    last_q_end = last_q_end - timedelta(days=last_q_end.day)
                    return last_q_end.strftime("%Y-%m-%d")
        
        # Last month
        if "last month" in lower_q or "previous month" in lower_q:
            first_of_this_month = today.replace(day=1)
            last_of_last_month = first_of_this_month - timedelta(days=1)
            first_of_last_month = last_of_last_month.replace(day=1)
            if param == "start_date":
                return first_of_last_month.strftime("%Y-%m-%d")
            else:  # end_date
                return last_of_last_month.strftime("%Y-%m-%d")
        
        # This month
        if "this month" in lower_q or "current month" in lower_q:
            if param == "start_date":
                return today.replace(day=1).strftime("%Y-%m-%d")
            else:  # end_date
                next_month = today.replace(day=28) + timedelta(days=4)
                last_of_month = next_month - timedelta(days=next_month.day)
                return last_of_month.strftime("%Y-%m-%d")
        
        # This year
        if "this year" in lower_q or "current year" in lower_q:
            if param == "start_date":
                return datetime(today.year, 1, 1).strftime("%Y-%m-%d")
            else:  # end_date
                return datetime(today.year, 12, 31).strftime("%Y-%m-%d")
        
        # Last year
        if "last year" in lower_q or "previous year" in lower_q:
            if param == "start_date":
                return datetime(today.year - 1, 1, 1).strftime("%Y-%m-%d")
            else:  # end_date
                return datetime(today.year - 1, 12, 31).strftime("%Y-%m-%d")

    return None
 
 
# --- 5. Core Agent Function ---

def lookup_cso_by_name(connection_string: str, name: str) -> List[Tuple[str, str, str]]:
    """
    Looks up CSO IDs by name.
    Returns: List of (csoid, csoname, businesscategory)
    """
    sql = f"SELECT csoid, name, businesscategory FROM pwccso_pocdetails WHERE LOWER(name) LIKE LOWER('%{name}%')"
    try:
        columns, rows = execute_sql_query_from_string(connection_string, sql)
        return [(row[0], row[1], row[2]) for row in rows]
    except Exception as e:
        print(f"[ERROR] CSO lookup failed: {e}")
        return []


# def lookup_cluster_by_name(connection_string: str, name: str) -> List[Tuple[str, str]]:
#     """
#     Looks up Cluster IDs by name.
#     Returns: List of (cluster_id, cluster_name)
#     """
#     sql = f"SELECT DISTINCT pwcclusterid, pwcclustername FROM custtable WHERE LOWER(pwcclustername) LIKE LOWER('%{name}%') AND pwcclusterid IS NOT NULL"
#     try:
#         columns, rows = execute_sql_query_from_string(connection_string, sql)
#         return [(row[0], row[1]) for row in rows]
#     except Exception as e:
#         print(f"[ERROR] Cluster lookup failed: {e}")
#         return []


# def lookup_state_by_name(connection_string: str, name: str) -> List[Tuple[str, str]]:
#     """
#     Looks up State IDs by name.
#     Returns: List of (state_id, state_name)
#     """
#     sql = f"SELECT DISTINCT pwcvirtualstateid, pwcvirtualstatename FROM custtable WHERE LOWER(pwcvirtualstatename) LIKE LOWER('%{name}%') AND pwcvirtualstateid IS NOT NULL"
#     try:
#         columns, rows = execute_sql_query_from_string(connection_string, sql)
#         return [(row[0], row[1]) for row in rows]
#     except Exception as e:
#         print(f"[ERROR] State lookup failed: {e}")
#         return []


# def extract_name_reference(user_question: str, param_type: str) -> str:
#     """
#     Extracts potential name references from user question.
    
#     Examples:
#     - "Top sales in Mahesh Kumar's CSO" → "Mahesh Kumar"
#     - "Sales in Bihar state" → "Bihar"
#     - "Performance in RJC01 cluster" → "RJC01" (if it's a code, returns empty)
#     """
#     import re
#     upper_q = user_question.upper()
    
#     if param_type == "cso_id":
#         # Look for patterns like "in [Name]'s CSO", "under [Name]", "for [Name] CSO"
#         patterns = [
#             r"(?:IN|UNDER|FOR)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*?)(?:'S)?\s+CSO",
#             r"CSO\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
#         ]
#         for pattern in patterns:
#             match = re.search(pattern, user_question, re.IGNORECASE)
#             if match:
#                 return match.group(1).strip()
    
#     elif param_type == "state_id":
#         # Look for state names (Bihar, Rajasthan, etc.)
#         patterns = [
#             r"(?:IN|FROM)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*?)\s+(?:STATE|TERRITORY)",
#             r"STATE\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
#         ]
#         for pattern in patterns:
#             match = re.search(pattern, user_question, re.IGNORECASE)
#             if match:
#                 name = match.group(1).strip()
#                 # Exclude if it's a 2-letter code
#                 if len(name) > 2:
#                     return name
    
#     elif param_type == "cluster_id":
#         # Look for cluster names
#         patterns = [
#             r"(?:IN|FROM)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*?)\s+CLUSTER",
#             r"CLUSTER\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
#         ]
#         for pattern in patterns:
#             match = re.search(pattern, user_question, re.IGNORECASE)
#             if match:
#                 name = match.group(1).strip()
#                 # Exclude if it looks like a code (e.g., RJC01)
#                 if not re.match(r'^[A-Z]{2,4}\d+$', name):
#                     return name
    
#     return ""
def merge_params_safely(existing: dict, incoming: dict):
    """
    Allows LLM extraction to correct pre-extracted values
    without losing already-confirmed values.
    """
    for k, v in incoming.items():
        if v in [None, "", "SKIP"]:
            continue

        if k not in existing or existing[k] in [None, "", "SKIP"]:
            existing[k] = v
        else:
            # If conflict: prefer LLM result
            if existing[k] != v:
                existing[k] = v


def run_sql_rag_agent(user_question: str, connection_string: str, chat_state: ChatState) -> str:
    try:
        clean_q = user_question.lower().strip()
        reset_keywords = ["start over", "reset", "begin again", "clear", "new question", "skip"]
        
        if any(word in clean_q for word in reset_keywords):
            chat_state.clear_all()
            return json.dumps({"bot_answer": "No problem! I've cleared everything. What new sales data can I help you find?"}) 
        
        # ========================================
        # 1️⃣ ADD USER MESSAGE TO HISTORY
        # ========================================
        chat_state.add_turn("user", user_question)
        
        # ========================================
        # 2️⃣ INTENT ROUTING
        # ========================================
        if not chat_state.pending_query_id:
            intent = INTENT_CHAIN.invoke({"question": user_question}).strip().upper()
            print(f"[DEBUG] Intent: {intent}")
               
            if intent in ["REJECT"]:
                chat_state.clear_all()
                return json.dumps({"bot_answer": DOMAIN_DECLINE_MESSAGE})

            if intent == "GREETING":
                greeting = "Hello! How can I help you with your sales data today?"
                chat_state.add_turn("assistant", greeting)
                return json.dumps({"bot_answer": greeting})

            if intent == "RESET":
                chat_state.clear_all()
                return json.dumps({"bot_answer": "All clear. What would you like to ask now?"})
            
            # TABLE MODE - Show raw values for data accuracy
            if user_question.strip().lower() in ["table", "in table", "show table", "display table", "show in table", "as table", "in table format", "table format"]:
                if chat_state.last_rows and chat_state.last_columns:
                    table_prompt = ChatPromptTemplate.from_messages([
                        ("system", "Format the following data as a markdown table. Output ONLY the table. Keep numeric values exactly as provided."),
                        ("human", "Columns: {columns}\nRows: {rows}")
                    ])
                    table_chain = table_prompt | LLM | StrOutputParser()
                    table = table_chain.invoke({
                        "columns": str(chat_state.last_columns),
                        "rows": str(chat_state.last_rows)
                    })
                    return json.dumps({"bot_answer": table})
                else:
                    return json.dumps({"bot_answer": "I don't have any previous result to display as a table."})

        # ========================================
        # 3️⃣ CONTEXT ANALYSIS (For Follow-up Detection)
        # ========================================
        inherited_params = {}
        override_hints = {}
        is_follow_up = False  # Track if this is a follow-up query
        
        if chat_state.has_context() and not chat_state.pending_query_id:
            try:
                context_input = {
                    "conversation_history": chat_state.get_history_for_llm(),
                    "last_question": chat_state.last_query_context,
                    "last_query_id": chat_state.last_query_id,
                    "last_params": json.dumps(chat_state.last_successful_params),
                    "current_message": user_question,
                    "current_date": datetime.now().strftime("%Y-%m-%d")  # Pass current date for date calculations
                }
                
                context_result = CONTEXT_CHAIN.invoke(context_input)
                
                # Robust JSON extraction - handle nested braces properly
                def extract_json(text):
                    """Extract first valid JSON object, handling nested braces."""
                    start = text.find('{')
                    if start == -1:
                        return None
                    
                    depth = 0
                    for i, char in enumerate(text[start:], start):
                        if char == '{':
                            depth += 1
                        elif char == '}':
                            depth -= 1
                            if depth == 0:
                                return text[start:i+1]
                    return None
                
                json_str = extract_json(context_result)
                if json_str:
                    # Clean up common JSON formatting issues from LLM
                    import re
                    # Remove trailing commas before } or ]
                    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
                    # Add missing commas between "value" "key" patterns
                    json_str = re.sub(r'"\s+(")', r'", \1', json_str)
                    # Fix unquoted values that should be strings
                    json_str = re.sub(r':\s*([A-Z_]+)(\s*[,}])', r': "\1"\2', json_str)
                    
                    try:
                        context_data = json.loads(json_str)
                    except json.JSONDecodeError:
                        # If still fails, try to extract key fields manually
                        context_data = {}
                        if '"FOLLOW_UP"' in json_str or "'FOLLOW_UP'" in json_str:
                            context_data['query_type'] = 'FOLLOW_UP'
                            context_data['confidence'] = 'HIGH'
                            # Inherit all previous params to maintain context
                            context_data['inherit_params'] = list(chat_state.last_successful_params.keys())
                        print(f"[DEBUG] JSON cleanup failed, using fallback: {context_data}")
                else:
                    # extract_json returned None - try parsing raw or fallback
                    try:
                        context_data = json.loads(context_result)
                    except json.JSONDecodeError:
                        # Check if it looks like a follow-up and inherit all params
                        context_data = {}
                        if 'FOLLOW_UP' in context_result.upper():
                            context_data['query_type'] = 'FOLLOW_UP'
                            context_data['confidence'] = 'HIGH'
                            context_data['inherit_params'] = list(chat_state.last_successful_params.keys())
                            print(f"[DEBUG] Fallback FOLLOW_UP from raw response, inheriting: {context_data['inherit_params']}")
                        else:
                            print(f"[DEBUG] Could not parse context, treating as NEW_QUERY")
                
                print(f"[DEBUG] Context Analysis: {context_data}")
                
                query_type = context_data.get("query_type")
                confidence = context_data.get("confidence")
                
                if confidence in ["HIGH", "MEDIUM"]:
                    # Handle ACKNOWLEDGMENT - user is just reacting, no query needed
                    if query_type == "ACKNOWLEDGMENT":
                        # Check for goodbye phrases
                        goodbye_phrases = ["bye", "goodbye", "see you", "thanks bye", "thank you bye", "cya", "later", "take care"]
                        if any(phrase in user_question.lower() for phrase in goodbye_phrases):
                            goodbye_response = "Goodbye! Feel free to return anytime you need help with sales data. Have a great day! 👋"
                            chat_state.add_turn("assistant", goodbye_response)
                            return json.dumps({"bot_answer": goodbye_response})
                        
                        ack_response = "ya! so is there anything else you'd like to know about the sales data?"
                        chat_state.add_turn("assistant", ack_response)
                        return json.dumps({"bot_answer": ack_response})
                    
                    # Handle CLARIFICATION_QUESTION - user is asking ABOUT the previous result
                    if query_type == "CLARIFICATION_QUESTION":
                        if chat_state.last_rows and chat_state.last_columns and chat_state.last_query_context:
                            # Get the sort direction that was used
                            direction_word = "lowest/bottom" if chat_state.last_sort_direction == "ASC" else "highest/top"
                            
                            # Get the first result from previous query
                            first_result = chat_state.last_rows[0] if chat_state.last_rows else []
                            result_name = first_result[0] if first_result else "unknown"
                            result_value = first_result[1] if len(first_result) > 1 else "N/A"
                            
                            clarification = f"That was the **{direction_word}** performer. {result_name} with ₹{result_value:,.0f} was the {direction_word.split('/')[0]} for the period."
                            chat_state.add_turn("assistant", clarification)
                            return json.dumps({"bot_answer": clarification})
                        else:
                            return json.dumps({"bot_answer": "I don't have a previous result to clarify. Could you ask your question again?"})
                    
                    # Inherit specified parameters from last successful query
                    # This applies to BOTH FOLLOW_UP and NEW_QUERY!
                    for param in context_data.get("inherit_params", []):
                        if param in chat_state.last_successful_params:
                            inherited_params[param] = chat_state.last_successful_params[param]
                    
                    # Capture overrides from context analysis
                    override_hints = context_data.get("override_params", {})
                    
                    # Mark as follow-up only for FOLLOW_UP type (affects query reuse)
                    if query_type == "FOLLOW_UP":
                        is_follow_up = True
                    
                    print(f"[DEBUG] Query Type: {query_type}")
                    print(f"[DEBUG] Is Follow-up (reuse query): {is_follow_up}")
                    print(f"[DEBUG] Inherited params: {inherited_params}")
                    print(f"[DEBUG] Override hints: {override_hints}")
                    
                    # ========================================
                    # CODE-LEVEL OVERRIDE: Detect product ↔ salesperson switches
                    # LLM sometimes fails to detect these, so we force NEW_QUERY behavior
                    # ========================================
                    msg_lower = user_question.lower()
                    last_qid = chat_state.last_query_id or ""
                    
                    # Detect if user is asking for salesperson but was on product query
                    asks_salesperson = any(kw in msg_lower for kw in ["salesperson", "who generated", "who made", "rank sales", "best performer", "top performer"])
                    was_product = "product_segment" in last_qid.lower()
                    
                    # Detect if user is asking for product but was on salesperson query
                    asks_product = any(kw in msg_lower for kw in ["product type", "product segment", "which product"])
                    was_salesperson = any(kw in last_qid.lower() for kw in ["salesperson", "category_performance", "category_specific"])
                    
                    if (asks_salesperson and was_product) or (asks_product and was_salesperson):
                        print(f"[DEBUG] CODE OVERRIDE: Detected query subject change, forcing NEW_QUERY behavior")
                        is_follow_up = False  # Force fresh query selection
                        # Clear persistent category filter to avoid it being inherited
                        if asks_product and "business_category" in chat_state.last_filter_params:
                            del chat_state.last_filter_params["business_category"]
                            print(f"[DEBUG] Cleared inherited business_category filter for new product query")
                    
            except (json.JSONDecodeError, Exception) as e:
                print(f"[DEBUG] Context analysis skipped: {e}")

        # ========================================
        # 4️⃣ QUERY SELECTION (Reuse for follow-ups OR semantic search)
        # ========================================
        if not chat_state.pending_query_id:
            do_semantic_search = True  # Default: do semantic search unless valid follow-up
            
            if is_follow_up and chat_state.last_query_id and chat_state.last_query_context:
                # SAFETY CHECK: Verify query type matches current question subject
                # Don't reuse a PRODUCT query for a PERSON question (or vice versa)
                last_was_product = 'product' in chat_state.last_query_id.lower() or 'segment' in chat_state.last_query_id.lower()
                current_asks_person = any(word in user_question.lower() for word in ['who', 'salesperson', 'performer', 'executive', 'sales rep'])
                
                if last_was_product and current_asks_person:
                    # Subject mismatch! Force semantic search
                    print(f"[DEBUG] FOLLOW-UP MISMATCH: Last query was product-type but current asks about person. Doing semantic search.")
                    # do_semantic_search stays True
                else:
                    # Valid FOLLOW-UP: Reuse the last successful query template
                    print(f"[DEBUG] FOLLOW-UP: Reusing query ID '{chat_state.last_query_id}'")
                    do_semantic_search = False  # Don't do semantic search
                    
                    try:
                        _, sql_template, req, opt, defaults = get_query_by_id(chat_state.last_query_id)
                    except ValueError:
                        _, sql_template, req, opt, defaults = semantic_search_sql(chat_state.last_query_context)
                    
                    chat_state.pending_query_id = chat_state.last_query_id
                    chat_state.last_query_template = sql_template
                    chat_state.optional_params = opt
                    chat_state.param_defaults = defaults
                    chat_state.original_question = chat_state.last_query_context
                    chat_state.param_collection_attempts = 0
            
            if do_semantic_search:
                # NEW QUERY or MISMATCH: Do semantic search
                qid, sql_template, req, opt, defaults = semantic_search_sql(user_question)

                chat_state.pending_query_id = qid
                chat_state.last_query_template = sql_template
                chat_state.optional_params = opt
                chat_state.param_defaults = defaults
                chat_state.original_question = user_question
                chat_state.param_collection_attempts = 0
            
            # Pre-populate with inherited params from context analysis
            chat_state.collected_params = inherited_params.copy()
            chat_state.collected_params.update(override_hints)
            
            # MUTUAL EXCLUSION: Location filters replace each other (state/cso/cluster)
            # If user specifies one, remove the others from persistent filters
            location_filters = {'state_id', 'cso_id', 'cluster_id'}
            new_location_filter = set(override_hints.keys()) & location_filters
            
            if new_location_filter:
                # User explicitly set a new location filter - remove conflicting ones
                filters_to_remove = location_filters - new_location_filter
                for f in filters_to_remove:
                    if f in chat_state.last_filter_params:
                        del chat_state.last_filter_params[f]
                        print(f"[DEBUG] Dropped conflicting filter: {f}")
            
            # CRITICAL: For follow-ups, also inherit persistent filter params 
            # (state_id, business_category, cso_id, cluster_id) from last successful query
            # These are only overridden if context explicitly provides new values
            if is_follow_up and chat_state.last_filter_params:
                for filter_key, filter_val in chat_state.last_filter_params.items():
                    # Only add if not already in collected_params (don't override explicit values)
                    if filter_key not in chat_state.collected_params:
                        chat_state.collected_params[filter_key] = filter_val
                        print(f"[DEBUG] Auto-inherited filter: {filter_key}={filter_val}")
            
            # Get required params (need to fetch again for follow-ups)
            if is_follow_up:
                _, _, req, _, _ = semantic_search_sql(chat_state.last_query_context)
            
            # Calculate truly missing params (not in inherited or overrides)
            chat_state.missing_params = [p for p in req if p not in chat_state.collected_params]

            # PRE-EXTRACTION from original question (for both new and follow-up queries)
            # This ensures "top 5" is always captured regardless of query type
            
            # Check if user explicitly mentions a time period (should override inherited dates)
            lower_q = user_question.lower()
            has_explicit_time = any(phrase in lower_q for phrase in [
                'last month', 'last year', 'last quarter', 'this month', 'this year',
                'previous month', 'previous year', 'previous quarter',
                'january', 'february', 'march', 'april', 'may', 'june',
                'july', 'august', 'september', 'october', 'november', 'december'
            ])
            
            for p in req + opt:
                # For 'n' and 'sort', ALWAYS try to extract from explicit user input
                # to override Context Analyzer defaults when user says "top 5" explicitly
                should_override = p in ['n', 'sort']
                
                # Also override dates when user explicitly mentions a time period
                if has_explicit_time and p in ['start_date', 'end_date']:
                    should_override = True
                
                if p not in chat_state.collected_params or should_override:
                    val = extract_from_original_question(user_question, p)
                    if val is not None:
                        chat_state.collected_params[p] = val
                        print(f"[DEBUG] Extracted from question: {p}={val}")
                        if p in chat_state.missing_params:
                            chat_state.missing_params.remove(p)

            # ========================================
            # EXPORT DETECTION (runs for all queries including follow-ups)
            # ========================================
            mentions_export = "export" in user_question.lower() or override_hints.get("business_type") == "export" or override_hints.get("segment") == "export" or override_hints.get("sales_type") == "export"
            if mentions_export and "export" not in chat_state.pending_query_id:
                # Check if user mentioned a specific category
                has_category = "business_category" in chat_state.collected_params or "business_category" in override_hints
                
                # First try QUERY_VARIANTS mapping
                export_key = f"{chat_state.pending_query_id}|export"
                export_query_id = QUERY_VARIANTS.get(export_key)
                
                # Special case: product_segment_domestic + category → product_segment_export_by_category
                if has_category and chat_state.pending_query_id == "product_segment_domestic":
                    export_query_id = "product_segment_export_by_category"
                
                if export_query_id:
                    print(f"[DEBUG] EXPORT DETECTED: Switching from '{chat_state.pending_query_id}' to '{export_query_id}'")
                    # Fetch the export query template directly by ID (no semantic search)
                    try:
                        _, sql_template, req, opt, defaults = get_query_by_id(export_query_id)
                        chat_state.pending_query_id = export_query_id
                        chat_state.last_query_template = sql_template
                        chat_state.optional_params = opt
                        chat_state.param_defaults = defaults
                        # Recalculate missing params for the new query
                        chat_state.missing_params = [p for p in req if p not in chat_state.collected_params]
                    except ValueError as e:
                        print(f"[DEBUG] Export query switch failed: {e}")

            # ========================================
            # DOMESTIC DETECTION (switch from export to domestic)
            # ========================================
            mentions_domestic = "domestic" in user_question.lower() or override_hints.get("sales_type") in ["domestic", "'domestic'"]
            if mentions_domestic and "export" in chat_state.pending_query_id:
                # User wants domestic - switch from export to domestic query
                domestic_key = f"{chat_state.pending_query_id}|domestic"
                domestic_query_id = QUERY_VARIANTS.get(domestic_key)
                
                if domestic_query_id:
                    print(f"[DEBUG] DOMESTIC DETECTED: Switching from '{chat_state.pending_query_id}' to '{domestic_query_id}'")
                    try:
                        _, sql_template, req, opt, defaults = get_query_by_id(domestic_query_id)
                        chat_state.pending_query_id = domestic_query_id
                        chat_state.last_query_template = sql_template
                        chat_state.optional_params = opt
                        chat_state.param_defaults = defaults
                        # Recalculate missing params for the new query
                        chat_state.missing_params = [p for p in req if p not in chat_state.collected_params]
                    except ValueError as e:
                        print(f"[DEBUG] Domestic query switch failed: {e}")

        # ========================================
        # 5️⃣ PARAMETER COLLECTION (Enhanced with context)
        # ========================================
        while chat_state.missing_params:
            chat_state.increment_attempts()

            llm_input = {
                "missing_params": ", ".join(chat_state.missing_params),
                "collected_params": json.dumps(chat_state.collected_params),
                "optional_params": ", ".join(chat_state.optional_params),
                "original_question": chat_state.original_question,
                "current_date": datetime.now().strftime("%Y-%m-%d"),
                "question": user_question,
                # NEW: Pass inherited context to parameter extractor
                "inherited_params": json.dumps(inherited_params),
                "override_hints": json.dumps(override_hints)
            }

            extraction = PARAMETER_EXTRACTION_CHAIN.invoke(llm_input)
            extracted = json.loads(extraction)

            # Direct check for "all" in business_category clarification
            if "business_category" in chat_state.missing_params:
                user_lower = user_question.lower().strip()
                if user_lower in ["all", "all categories", "everything", "all of them"]:
                    extracted["business_category"] = "'FMEG', 'Wiring Devices & Switchgear', 'Wires & Cables'"
                    print(f"[DEBUG] Direct 'all' detected for business_category - setting all 3 categories")

            # Merge extracted params safely
            merge_params_safely(chat_state.collected_params, extracted)

            # ========================================
            # DYNAMIC QUERY SWITCHING
            # ========================================
            
            # Detect "all" for business_category → set all 3 categories
            bc_value = str(chat_state.collected_params.get("business_category", "")).lower().strip()
            if bc_value in ["all", "'all'", "all categories", "everything"]:
                # User wants all categories - set all 3 business categories
                chat_state.collected_params["business_category"] = "'FMEG', 'Wiring Devices & Switchgear', 'Wires & Cables'"
                print(f"[DEBUG] 'all' detected - setting all business categories")
            
            # Detect "export" keyword → switch to export query variant
            if "export" in user_question.lower() and "domestic" not in user_question.lower():
                export_key = f"{chat_state.pending_query_id}|export"
                export_query_id = QUERY_VARIANTS.get(export_key)
                if export_query_id and export_query_id != chat_state.pending_query_id:
                    print(f"[DEBUG] Switching from '{chat_state.pending_query_id}' to export query '{export_query_id}'")
                    # Fetch the export query template directly by ID
                    try:
                        _, sql_template, req, opt, defaults = get_query_by_id(export_query_id)
                        chat_state.pending_query_id = export_query_id
                        chat_state.last_query_template = sql_template
                        chat_state.optional_params = opt
                        chat_state.param_defaults = defaults
                    except ValueError as e:
                        print(f"[DEBUG] Export query switch failed: {e}")

            chat_state.missing_params = [
                p for p in chat_state.missing_params
                if p not in chat_state.collected_params
            ]

            if chat_state.missing_params:
                return json.dumps({"bot_answer": format_missing_params_message(
                    chat_state.missing_params,
                    chat_state.param_collection_attempts,
                    chat_state.MAX_PARAM_ATTEMPTS
                )})

        # ========================================
        # 6️⃣ APPLY DEFAULTS
        # ========================================
        for p, d in chat_state.param_defaults.items():
            if p not in chat_state.collected_params:
                if isinstance(d, str) and d.startswith("__"):
                    d = calculate_date_from_placeholder(d)
                chat_state.collected_params[p] = d

        print(f"[DEBUG] All parameters collected: {chat_state.collected_params}")

        # ========================================
        # 6.5️⃣ VALIDATE QUERY SUPPORTS ALL PARAMS (Auto-switch if needed)
        # ========================================
        best_query = find_best_query_for_params(
            chat_state.collected_params, 
            chat_state.pending_query_id
        )
        
        if best_query and best_query != chat_state.pending_query_id:
            print(f"[DEBUG] Query doesn't support all filters. Switching: '{chat_state.pending_query_id}' → '{best_query}'")
            try:
                _, sql_template, req, opt, defaults = get_query_by_id(best_query)
                chat_state.pending_query_id = best_query
                chat_state.last_query_template = sql_template
                chat_state.optional_params = opt
                chat_state.param_defaults = defaults
            except ValueError as e:
                print(f"[DEBUG] Query switch failed, using original: {e}")

        # ========================================
        # 7️⃣ EXECUTION
        # ========================================
        # Ensure optional parameters have defaults before formatting SQL
        if 'n' not in chat_state.collected_params:
            chat_state.collected_params['n'] = 1  # Default to 1 result
        if 'sort' not in chat_state.collected_params:
            chat_state.collected_params['sort'] = 'DESC'  # Default to highest
        
        final_sql = chat_state.last_query_template.format(**chat_state.collected_params)
        print(f"[DEBUG] Executing Query ID: {chat_state.pending_query_id}")
        print(f"[DEBUG] Final SQL: {final_sql}")

        columns, rows = execute_sql_query_from_string(connection_string, final_sql)

        # ========================================
        # 8️⃣ RESPONSE GENERATION
        # ========================================
        # Determine question context for response generation
        response_question = user_question
        
        # Import regex for ordinal detection
        import re
        ordinal_pattern = r'\b\d+(st|nd|rd|th)\b'  # Matches 1st, 2nd, 3rd, 4th, 5th, etc.
        has_true_ordinal = bool(re.search(ordinal_pattern, user_question.lower()))
        
        # Check if user_question is a short response (likely clarification or follow-up modifier)
        words = user_question.lower().split()
        is_short_response = len(words) <= 5
        
        # Get current sort direction for accurate response
        current_sort = chat_state.collected_params.get('sort', 'DESC')
        direction_word = "highest" if current_sort == "DESC" else "lowest"
        
        # Case 1: Follow-up with context
        if is_follow_up and chat_state.last_query_context:
            if is_short_response and not has_true_ordinal:
                # Build a clearer question that reflects the actual sort direction
                # Extract date/period from original question if present
                n_value = chat_state.collected_params.get('n', 1)
                response_question = f"Show {direction_word} {n_value} performer(s) ({user_question})"
                print(f"[DEBUG] Combined question for response (follow-up): {response_question}")
        
        # Case 2: Clarification - user answered a parameter question
        # original_question is set, but this isn't a follow-up (first query's param collection)
        elif not is_follow_up and chat_state.original_question and is_short_response:
            if chat_state.original_question.lower() != user_question.lower():
                response_question = f"{chat_state.original_question} ({user_question})"
                print(f"[DEBUG] Combined question for response (clarification): {response_question}")
        
        # Build query context for the LLM to mention filters naturally
        context_parts = []
        params = chat_state.collected_params
        
        # Just pass raw values - LLM knows state codes, categories, etc.
        if params.get('state_id'):
            context_parts.append(f"State: {params['state_id']}")
        
        if params.get('business_category'):
            cat = params['business_category'].replace("'", "")
            context_parts.append(f"Category: {cat}")
        
        if params.get('sales_type'):
            context_parts.append(f"Sales Type: {params['sales_type'].capitalize()}")
        
        if params.get('cso_id'):
            context_parts.append(f"CSO: {params['cso_id']}")
        
        if params.get('cluster_id'):
            context_parts.append(f"Cluster: {params['cluster_id']}")
        
        # Format date period
        if params.get('start_date') and params.get('end_date'):
            start = params['start_date']
            end = params['end_date']
            # Try to make it more readable
            try:
                start_dt = datetime.strptime(start, '%Y-%m-%d')
                end_dt = datetime.strptime(end, '%Y-%m-%d')
                if start_dt.month == end_dt.month and start_dt.year == end_dt.year:
                    period = start_dt.strftime('%B %Y')
                else:
                    period = f"{start_dt.strftime('%b %Y')} to {end_dt.strftime('%b %Y')}"
                context_parts.append(f"Period: {period}")
            except:
                context_parts.append(f"Period: {start} to {end}")
        
        query_context = " | ".join(context_parts) if context_parts else "No specific filters"
        print(f"[DEBUG] Query Context: {query_context}")
        
        # Pre-format currency values before sending to LLM
        # This is CRITICAL - LLMs cannot reliably do arithmetic!
        formatted_rows = preformat_currency_in_rows(columns, rows)
        
        llm_input = {
            "columns": str(columns),
            "rows": str(formatted_rows),
            "question": response_question,
            "query_context": query_context
        }

        answer = (input_format | AGENT_PROMPT | LLM | StrOutputParser()).invoke(llm_input)

        # ========================================
        # 9️⃣ SAVE CONTEXT FOR NEXT TURN
        # ========================================
        chat_state.last_columns = columns
        chat_state.last_rows = rows
        
        # Save successful query context for follow-ups
        chat_state.save_successful_query(
            params=chat_state.collected_params,
            question=chat_state.original_question,
            query_id=chat_state.pending_query_id
        )
        
        # Add assistant response to history
        chat_state.add_turn("assistant", answer, chat_state.collected_params)
        
        # Clear current query state (but NOT conversation history)
        chat_state.clear_query_state()

        return json.dumps({"bot_answer": answer})

    except Exception as e:
        chat_state.clear_all()
        return json.dumps({"bot_answer": f"Unexpected error: {e}"})