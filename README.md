# 🤖 Enterprise Sales Bot

A sophisticated **Context-Aware SQL RAG (Retrieval-Augmented Generation) Agent** that enables sales teams to query complex invoice and sales data using natural language, directly within **Microsoft Teams**.

## ✨ Features

- **Natural Language to SQL** - Ask questions in plain English, get accurate data
- **Multi-Turn Conversations** - Maintains context across multiple messages
- **Smart Parameter Extraction** - Automatically extracts dates, categories, and filters
- **Indian Currency Formatting** - Displays amounts in ₹ Crores/Lakhs format
- **Microsoft Teams Integration** - Works directly in your Teams chat

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Groq (LangChain) |
| Vector DB | ChromaDB |
| Backend | FastAPI + Uvicorn |
| Database | Microsoft Fabric / SQL Server |
| Platform | Microsoft Teams Bot Framework |

## 📁 Project Structure

```
├── agent.py              # Main RAG agent logic
├── db.py                 # Database & vector store functions
├── teams_C.py            # Teams bot server
├── index_C.py            # Terminal interface for testing
├── reset_chroma.py       # ChromaDB reset utility
├── queries_invoice.json  # SQL query templates
├── schemas/
│   └── state.py          # ChatState model
├── requirements.txt      # Python dependencies
└── .env.example          # Environment variables template
```

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/craftcodeworks/Enterprise-Sales-Bot.git
cd Enterprise-Sales-Bot
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
```bash
cp .env.example .env
# Edit .env with your credentials
```

### 5. Run the terminal interface (for testing)
```bash
python index_C.py
```

### 6. Run the Teams bot server
```bash
python teams_C.py
```

## ⚙️ Environment Variables

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Your Groq API key |
| `FABRIC_DB_SERVER` | Microsoft Fabric server URL |
| `FABRIC_DB_NAME` | Database name |
| `FABRIC_DB_USER` | Database username |
| `FABRIC_DB_PASSWORD` | Database password |
| `TEAMS_APP_ID` | Microsoft Bot App ID |
| `TEAMS_APP_PASSWORD` | Microsoft Bot App Password |
| `TEAMS_TENANT_ID` | Azure Tenant ID |

## 📝 Usage Examples

```
👤 User: Who is the top salesperson this month?
🤖 Bot: The top performer this month is Rahul Sharma with ₹26.44 Cr in sales.

👤 User: What about for FMEG category?
🤖 Bot: For FMEG category, Priya Patel leads with ₹8.5 Cr.

👤 User: Show bottom 5
🤖 Bot: Here are the bottom 5 performers in FMEG...
```

## 📄 License

MIT License

---

Built with ❤️ using LangChain, ChromaDB, and FastAPI
