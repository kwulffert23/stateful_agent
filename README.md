# Stateful Agent with Lakebase - Streamlit Chat UI

A conversational AI chatbot with persistent history using Databricks Lakebase and LangGraph stateful agents.

## Architecture

```
┌─────────────────────┐
│   Streamlit UI      │  ← User interface (app.py)
└──────────┬──────────┘
           │ HTTP + thread_id
           ↓
┌─────────────────────┐
│  Agent Endpoint     │  ← Stateful agent (LangGraph)
└──────────┬──────────┘
           │ Checkpoints
           ↓
┌─────────────────────┐
│  Lakebase (Postgres)│  ← Persistent storage
│  - checkpoints      │     Agent's full state
│  - session_metadata │     UI conversation list
└─────────────────────┘
```

## Code Structure

```
├── app.py                    # Streamlit chat UI
├── app.yaml                  # Databricks Apps deployment config
├── lakebase_history_utils.py # Lakebase connection & queries
├── model_serving_utils.py    # Agent endpoint queries
├── requirements.txt          # Python dependencies
└── stateful_agent/
    └── agent_code/
        └── stateful-agent-lakebase.py  # LangGraph agent
```

## How It Works

### 1. Lakebase Connection

Uses `psycopg` with rotating OAuth tokens:

```python
# Generate token from WorkspaceClient
token = w.config.oauth_token().access_token

# Custom connection class injects fresh tokens
class RotatingTokenConnection(psycopg.Connection):
    def connect(cls, conninfo="", **kwargs):
        kwargs["password"] = token  # OAuth token as password
        return super().connect(conninfo, **kwargs)

# Connection pool for efficient reuse
pool = ConnectionPool(
    conninfo="host=... dbname=... user=...",
    connection_class=RotatingTokenConnection
)
```

### 2. Database Operations

Standard psycopg cursor pattern:

```python
with pool.connection() as conn:
    with conn.cursor() as cur:
        # Execute SQL
        cur.execute("SELECT * FROM session_metadata WHERE user_id = %s", (user_id,))
        results = cur.fetchall()
    conn.commit()
```

**Key Tables:**
- `checkpoints` - Agent's full conversation state (managed by LangGraph)
- `session_metadata` - UI metadata (thread_id, first_query, messages)

### 3. Conversation Flow

1. **New conversation**: Generate `thread_id`, send to agent
2. **Agent response**: Agent loads history from checkpoints using `thread_id`
3. **Save metadata**: Store conversation summary + messages for UI
4. **Resume conversation**: Load `thread_id` + messages, agent remembers everything

## Deployment

```bash
databricks apps deploy your-app-name --source-code-path .
```

**Requirements:**
- Databricks workspace with Apps enabled
- Lakebase instance with database user configured
- Agent deployed to Model Serving endpoint
- Permissions: CAN_QUERY on endpoint, CAN_USE on Lakebase

## Configuration

Set in `app.yaml`:
- `SERVING_ENDPOINT` - Agent endpoint name
- `PGHOST`, `PGUSER`, `PGDATABASE` - Lakebase connection details
- `LAKEBASE_INSTANCE_NAME` - For OAuth token generation

Authentication uses workspace OAuth tokens, automatically refreshed on each connection.
