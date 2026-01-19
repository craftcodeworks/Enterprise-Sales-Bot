# Enterprise-Grade Context-Aware Conversational AI Bot

## 🚀 Project Overview
Developed a sophisticated **Context-Aware SQL RAG (Retrieval-Augmented Generation) Agent** for a major manufacturing enterprise to provide real-time business insights from their SSMS (SQL Server Management Studio) data warehouse. The bot enables sales teams and management to query complex invoice and sales data using natural language, directly within **Microsoft Teams**.

### 🛠️ Key Technical Stack
-   **LLM Integration**: Groq (Llama 3 / Mixtral) via LangChain.
-   **Vector Database**: ChromaDB for semantic intent matching.
-   **Backend**: FastAPI for asynchronous message handling.
-   **Database**: Microsoft SQL Server / SSMS (via `pyodbc`).
-   **Platform**: Microsoft Teams Bot Framework integration.
-   **Data Processing**: Pydantic for state management, `python-dateutil` for advanced date resolution.

---

## 🌟 Top Features & Achievements

### 1. Advanced NL2SQL with Intent Recognition
-   **Semantic Search Layer**: Matches user questions against a curated library of high-performance SQL templates.
-   **100% Query Accuracy**: Prevents hallucinations by using verified templates rather than raw SQL generation.
-   **Dynamic Refinement**: Automatically switches between Domestic, Export, and Category-specific queries based on extracted parameters.

### 2. Multi-Turn Conversational Intelligence
-   **Parameter Filling**: Intelligent follow-ups for missing information (e.g., business category or date range).
-   **Clarification Handling**: Recognizes follow-up questions like "Highest or Lowest?" and applies them to the current context.

### 3. Custom Indian Financial Formatting (Lakhs/Crores)
-   **Accuracy First**: Built a Python-based post-processing engine to handle the Indian numbering system reliably.
-   **Localized Reporting**: Converts raw data into formatted strings like **₹26.44 Cr** or **₹55 L**.

### 4. Smart Date & Entity Resolution
-   **Relative Dates**: Resolves "last quarter," "previous fiscal year," or "last month" into precise date ranges.
-   **Fuzzy Matching**: Automatically resolves salesperson names or product categories to their database IDs.

---

## 🏗️ Technical Architecture
1.  **Incoming Message**: Teams Bot forwards request via FastAPI.
2.  **Context Analysis**: Agent extracts parameters and intent (New vs. Follow-up).
3.  **Semantic Retrieval**: ChromaDB finds the best matching template.
4.  **SQL Execution**: Executes against SSMS data warehouse.
5.  **Data Synthesis**: Formats results (Currency/Tables) and generates LLM summary.
6.  **Response**: Pushed back to the Teams chat interface.
