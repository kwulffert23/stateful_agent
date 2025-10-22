# Stateful Agent with Streamlit Chat UI - Templates

A complete template for deploying a stateful AI agent with a conversational web interface using Databricks, Lakebase, and Streamlit.

## What This Is

**Agent:** LangGraph-based stateful agent that remembers conversation context across sessions  
**Storage:** Databricks Lakebase (PostgreSQL) for durable memory  
**Frontend:** Streamlit chat app with conversation history sidebar  
**Deployment:** Databricks Apps with automatic authentication

## Template Files

### Agent (Backend)
- agent_code/stateful-agent-lakebase-TEMPLATE.py` - Agent notebook with TODOs

### Streamlit App (Frontend)
- `app_template.py` - Chat UI template with TODOs
- `model_serving_utils.py` - Endpoint query utilities
- `app.yaml.template` - Databricks Apps config template
- `requirements.txt` - Python dependencies

## Quick Start

### 1. Deploy the Agent

```bash
# Open in Databricks
agent_code/stateful-agent-lakebase-TEMPLATE.py

# Search for "TODO" (5 sections)
# Fill in: Lakebase credentials, LLM endpoint, UC model name
# Run all cells
# Note your endpoint name (e.g., agents_youruser-yourmodel)
```

### 2. Deploy the Chat App

```bash
# Copy templates
cp app_template.py app.py
cp app.yaml.template app.yaml

# Edit app.yaml - search for "TODO"
# Replace: YOUR_AGENT_ENDPOINT_NAME

# Deploy
databricks apps deploy my-chatbot --source-code-path .
```

### 3. Open your app URL and start chatting.

## What You Need to Fill In

### Agent Template
1. **Lakebase Setup:** Instance name, host, Service Principal credentials
2. **Agent Config:** LLM endpoint, system prompt
3. **UC Tools:** Optional Unity Catalog functions
4. **Vector Search:** Optional retrieval indexes
5. **UC Model:** Catalog, schema, model name

### App Template 
1. **Endpoint Name:** Your deployed agent endpoint
2. **Lakebase Instance:** Only if using persistent history

## This is a session-based demo that can be extended to a persistent history in the UI retrieving it from lakebase.

## Architecture

```
┌─────────────────┐
│  Streamlit App  │ ← User interacts here
└────────┬────────┘
         │ HTTP + thread_id
         ↓
┌─────────────────┐
│ Agent Endpoint  │ ← Stateful logic
│  (Model Serving)│
└────────┬────────┘
         │ Checkpoints
         ↓
┌─────────────────┐
│  Lakebase (DB)  │ ← Persistent memory
│  (PostgreSQL)   │
└─────────────────┘
```

## Key Files Explained

### `model_serving_utils.py`
- Queries Databricks serving endpoints
- Handles both agent and chat completion endpoints
- Passes `thread_id` and `user_id` to agent
- Parses responses correctly

### `app_template.py`
- Streamlit chat interface
- Sidebar with conversation list
- Thread ID generation and management
- Error handling and fallbacks

### `app.yaml.template`
- Databricks Apps deployment config
- Serving endpoint resource declaration
- Environment variables
- Optional Lakebase resource

## Requirements

- Databricks workspace (AWS or Azure)
- Lakebase instance (for agent memory)
- Model serving endpoint access
- Unity Catalog (for agent registration)
- Service Principal with Lakebase permissions


