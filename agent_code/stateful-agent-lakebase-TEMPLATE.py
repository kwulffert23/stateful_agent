# Databricks notebook source
# MAGIC %md
# MAGIC # Mosaic AI Agent Framework: Author and deploy a Stateful Agent with Thread-scoped memory using Databricks Lakebase and LangGraph
# MAGIC 
# MAGIC **TEMPLATE VERSION WITH TODO MARKERS**
# MAGIC 
# MAGIC This is a template version of the stateful agent notebook with TODO markers for all sensitive/customizable information.
# MAGIC Before using:
# MAGIC 1. Search for "TODO" comments
# MAGIC 2. Replace all placeholder values with your actual values
# MAGIC 3. Remove this template notice
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC This notebook demonstrates how to build a stateful agent using the Mosaic AI Agent Framework and LangGraph, with Lakebase as the agent's durable memory and checkpoint store. Threads allow you to store conversational state in Lakebase so you can pass in thread IDs into your agent instead of needing to send the full conversation history.
# MAGIC In this notebook, you will:
# MAGIC 1. Author a Stateful Agent graph with Lakebase (the new Postgres database in Databricks) and Langgraph to manage state using thread ids in a Databricks Agent 
# MAGIC 2. Wrap the LangGraph agent with `ResponsesAgent` interface to ensure compatibility with Databricks features
# MAGIC 3. Test the agent's behavior locally
# MAGIC 4. Register model to Unity Catalog, log and deploy the agent for use in apps and Playground
# MAGIC
# MAGIC We are using [PostgresSaver in Langgraph](https://api.python.langchain.com/en/latest/checkpoint/langchain_postgres.checkpoint.PostgresSaver.html) to open a connection with our Lakebase Postgres database.
# MAGIC
# MAGIC ## Why use Lakebase?
# MAGIC Stateful agents need a place to persist, resume, and inspect their work. Lakebase provides a managed, UC-governed store for agent state:
# MAGIC - Durable, resumable state. Automatically capture threads, intermediate checkpoints, tool outputs, and node state after each graph step so you can resume, branch, or replay any point in time.
# MAGIC - Queryable & observable. Because state lands in the Lakehouse, you can use SQL (or notebooks) to audit conversations and build upon other Databricks functionality like dashboards
# MAGIC - Governed by Unity Catalog. Apply data permissions, lineage, and auditing to AI state, just like any other table.
# MAGIC
# MAGIC ## What are Stateful Agents?
# MAGIC Unlike stateless LLM calls, a stateful agent keeps and reuses context across steps and sessions. Each new conversation is tracked with a thread ID, which represents the logical task or dialogue stream. Pick up an existing thread at any time to continue the conversation without having to pass in the entire conversation history.
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Create a Lakebase instance, see Databricks documentation ([AWS](https://docs.databricks.com/aws/en/oltp/create/) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/oltp/create/)). 
# MAGIC - You can create a Lakebase instance by going to SQL Warehouses -> Lakebase Postgres -> Create database instance. You will need to retrieve values from the "Connection details" section of your Lakebase to fill out this notebook.
# MAGIC - Complete all the "TODO"s throughout this notebook

# COMMAND ----------

# MAGIC %md
# MAGIC ### Install dependencies

# COMMAND ----------

# DBTITLE 1,or
# MAGIC %pip install -U -qqqq databricks-langchain langgraph==0.5.3 uv databricks-agents mlflow-skinny[databricks] \
# MAGIC   langgraph-checkpoint-postgres==2.0.21 psycopg[binary,pool]
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## First time setup only: Set up checkpointer with your Lakebase instance

# COMMAND ----------

import os
import uuid
from databricks.sdk import WorkspaceClient
from psycopg_pool import ConnectionPool
import psycopg
from langgraph.checkpoint.postgres import PostgresSaver

# ============================================================================
# TODO SECTION 1: LAKEBASE SETUP
# ============================================================================
# Fill in your lakebase instance details here.
# 
# How to get these values:
# 1. Go to Databricks UI -> Compute -> Database Instances
# 2. Find your Lakebase instance
# 3. Click on it and go to "Connection details"
# 
# For the username (SP_CLIENT_ID), create a Service Principal:
# 1. Go to Settings -> Developer -> Service Principals
# 2. Create a new Service Principal
# 3. Grant it databricks_superuser permissions on the lakebase instance
# 4. Generate a secret for it
# 5. Use the client_id as SP_CLIENT_ID and secret as SP_CLIENT_SECRET
# 
# See Service Principal Documentation for more information:
# https://docs.databricks.com/en/admin/users-groups/service-principals
# ============================================================================

# TODO: Replace with your Lakebase instance name
DB_INSTANCE_NAME = "YOUR_LAKEBASE_INSTANCE_NAME"  # e.g., "my-lakebase"

# TODO: Replace with your database name (usually databricks_postgres)
DB_NAME = "databricks_postgres"

# TODO: Replace with your Service Principal client ID
SP_CLIENT_ID = "YOUR_SERVICE_PRINCIPAL_CLIENT_ID"  # e.g., "12345678-1234-1234-1234-123456789012"

# TODO: Replace with your Service Principal client secret
SP_CLIENT_SECRET = "YOUR_SERVICE_PRINCIPAL_CLIENT_SECRET"  # e.g., "dosxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# SSL mode (usually "require", don't change unless you know what you're doing)
SSL_MODE = "require"

# TODO: Replace with your Lakebase host
# Get this from: Databricks UI -> Compute -> Database Instances -> Your Instance -> Connection details
DB_HOST = "instance-xxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx.database.azuredatabricks.net"

# TODO: Port (usually 5432, don't change unless you know what you're doing)
DB_PORT = 5432

# TODO: Replace with your Databricks workspace URL
WORKSPACE_HOST = "https://adb-xxxxxxxxxxxx.xx.azuredatabricks.net"

# ============================================================================

w = WorkspaceClient(
  host = WORKSPACE_HOST,
  client_id = SP_CLIENT_ID,
  client_secret = SP_CLIENT_SECRET
)

def db_password_provider() -> str:
    """
    Ask Databricks to mint a fresh DB credential for this instance.
    """
    cred = w.database.generate_database_credential(
        request_id=str(uuid.uuid4()),
        instance_names=[DB_INSTANCE_NAME],
    )
    return cred.token

class CustomConnection(psycopg.Connection):
    """
    A psycopg Connection subclass that injects a fresh password
    *at connection time* (only when the pool creates a new connection).
    """
    @classmethod
    def connect(cls, conninfo="", **kwargs):
        # Append the new password to kwargs
        kwargs["password"] = db_password_provider()
        # Call the superclass's connect method with updated kwargs
        return super().connect(conninfo, **kwargs)

pool = ConnectionPool(
    conninfo=f"dbname={DB_NAME} user={SP_CLIENT_ID} host={DB_HOST} port={DB_PORT} sslmode={SSL_MODE}",
    connection_class=CustomConnection,
    min_size=1,
    max_size=10,
)

# Test the connection by initializing the PostgresSaver
checkpointer = PostgresSaver(pool)
checkpointer.setup()
print("✅ Checkpointer setup complete! Lakebase tables initialized.")

# COMMAND ----------

# MAGIC %md
# MAGIC # Define Agent and Register as UC Model

# COMMAND ----------

# DBTITLE 1,Imports
from typing import Literal
from databricks_langchain import ChatDatabricks
from langchain_core.tools import tool, StructuredTool
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode
from databricks.agents import VectorSearchRetrieverTool
from mlflow.langchain.databricks_dependencies import (
    _detect_databricks_dependencies,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define all hyperparameters and tools

# COMMAND ----------

# ============================================================================
# TODO SECTION 2: AGENT CONFIGURATION
# ============================================================================

# TODO: Replace with your model serving endpoint
# You can find available endpoints at: Compute -> Serving -> Model Serving
# For Databricks Foundation Models: https://docs.databricks.com/en/machine-learning/foundation-models/
LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"  # or "databricks-meta-llama-3-1-70b-instruct", etc.

# TODO: Update with your system prompt
# This defines the agent's behavior and personality
SYSTEM_PROMPT = "You are a helpful assistant. Use the available tools to answer questions."

# TODO: Fill in Lakebase config values here (should match the values above)
LAKEBASE_CONFIG = {
    "instance_name": "YOUR_LAKEBASE_INSTANCE_NAME",  # Same as DB_INSTANCE_NAME
    "conn_host": "instance-xxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx.database.azuredatabricks.net",  # Same as DB_HOST
    "conn_db_name": "databricks_postgres",  # Same as DB_NAME
    "conn_ssl_mode": "require",
}

# ============================================================================

###############################################################################
## Define tools for your agent,enabling it to retrieve data or take actions
## beyond text generation
## To create and see usage examples of more tools, see
## https://docs.databricks.com/en/generative-ai/agent-framework/agent-tool.html
###############################################################################

tools = []

# ============================================================================
# TODO SECTION 3: UNITY CATALOG TOOLS (Optional)
# ============================================================================
# If you have Unity Catalog functions you want to use as tools, add them here
# Example:
# UC_TOOL_NAMES: list[str] = [
#     "my_catalog.my_schema.my_function",
#     "my_catalog.my_schema.another_function",
# ]
# ============================================================================

UC_TOOL_NAMES: list[str] = []
if UC_TOOL_NAMES:
    uc_toolkit = UCFunctionToolkit(function_names=UC_TOOL_NAMES)
    tools.extend(uc_toolkit.tools)

# ============================================================================
# TODO SECTION 4: VECTOR SEARCH TOOLS (Optional)
# ============================================================================
# If you have Vector Search indexes for retrieval, add them here
# See https://docs.databricks.com/en/generative-ai/agent-framework/unstructured-retrieval-tools.html
# 
# How to add:
# 1. Create a Vector Search index in Databricks
# 2. Note the full index name: catalog.schema.index_name
# 3. Add it below with a descriptive tool_description
# 
# Example:
# VECTOR_SEARCH_TOOLS.append(
#     VectorSearchRetrieverTool(
#         index_name="my_catalog.my_schema.my_knowledge_index",
#         tool_description="knowledge base of company documentation"
#     )
# )
# ============================================================================

VECTOR_SEARCH_TOOLS = []

# TODO: Add your vector search tools here (uncomment and customize)
# VECTOR_SEARCH_TOOLS.append(
#     VectorSearchRetrieverTool(
#         index_name="YOUR_CATALOG.YOUR_SCHEMA.YOUR_INDEX_NAME",
#         tool_description="Description of what this knowledge base contains"
#     )
# )

tools.extend(VECTOR_SEARCH_TOOLS)

#####################
## Define agent logic
#####################

llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)
llm_with_tools = llm.bind_tools(tools)

# Define the function that calls the model
def call_model(state: MessagesState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define the function that determines whether to continue or not
def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state["messages"]
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, we stop (reply to the user)
    return END


# Define a new graph
workflow = StateGraph(MessagesState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.add_edge(START, "agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("tools", "agent")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Wrap agent as a `ResponsesAgent` and run locally

# COMMAND ----------

import uuid
from typing import Iterator, Optional
from databricks.agents.databricks import ResponsesAgent
from mlflow.types.llm import ChatCompletionRequest, ChatCompletionResponse, ChatMessage
from mlflow.types.agents import (
    AgentOutput,
    AgentOutputText,
    ResponsesAgentRequest,
    ResponsesAgentResponse,
)


class LangGraphResponsesAgent(ResponsesAgent):
    """
    A ResponsesAgent implementation that wraps a LangGraph graph and uses
    Lakebase (PostgreSQL) for checkpointing.
    """

    def __init__(self, lakebase_config: dict):
        """
        Initialize the agent with Lakebase configuration.

        Args:
            lakebase_config: Dict containing:
                - instance_name (str): Lakebase instance name
                - conn_host (str): Connection host
                - conn_db_name (str): Database name (default: databricks_postgres)
                - conn_ssl_mode (str): SSL mode (default: require)
        """
        self.lakebase_config = lakebase_config

    def _get_username(self) -> str:
        """Get the username for the database connection."""
        return w.current_user.me().user_name

    def _setup_connection_pool(self) -> ConnectionPool:
        """
        Set up a connection pool with auto-rotating credentials.
        """
        
        username = self._get_username()
        host = self.lakebase_config["conn_host"]
        database = self.lakebase_config.get("conn_db_name", "databricks_postgres")

        class RotatingTokenConnection(psycopg.Connection):
            """
            A psycopg Connection subclass that injects a fresh DB token
            *at connection time* (only when the pool creates a new connection).
            """

            @classmethod
            def fun_generate_token(cls, instance_name: str) -> str:
                """Generate a fresh OAuth token for Lakebase access."""
                cred = w.database.generate_database_credential(
                    request_id=str(uuid.uuid4()), instance_names=[instance_name]
                )
                return cred.token

            @classmethod
            def connect(cls, conninfo: str = "", **kwargs):
                """
                Override connect to inject a fresh token as the password.
                """
                instance_name = kwargs.pop("_instance_name")
                kwargs["password"] = cls.fun_generate_token(instance_name)
                kwargs.setdefault("sslmode", "require")
                return super().connect(conninfo, **kwargs)

        # Build the connection string
        conninfo = f"host={host} dbname={database} user={username}"

        # Create the pool with the custom connection class
        pool = ConnectionPool(
            conninfo=conninfo,
            connection_class=RotatingTokenConnection,
            kwargs={"_instance_name": self.lakebase_config["instance_name"]},
            min_size=1,
            max_size=5,
            open=True,
        )
        return pool

    def _get_thread_id(self, request: ResponsesAgentRequest) -> str:
        """
        Determine which thread_id to use for this request.
        Precedence:
        1. Use thread_id from custom_inputs if present
        2. Use conversation_id from chat context if available
        3. Generate a new UUID
        
        Returns:
            thread_id: The thread identifier to use for this conversation
        """
        ci = dict(request.custom_inputs or {})
        
        if "thread_id" in ci:
            return ci["thread_id"]
        
        # using conversation id from chat context as thread id
        # https://mlflow.org/docs/latest/api_reference/python_api/mlflow.types.html#mlflow.types.agent.ChatContext
        if request.context and getattr(request.context, "conversation_id", None):
            return request.context.conversation_id
        
        # Generate new thread_id
        return str(uuid.uuid4())
    
    def predict_stream(self, request: ResponsesAgentRequest) -> Iterator[ResponsesAgentResponse]:
        """
        Handles streaming predictions with Lakebase checkpointing.
        """
        pool = self._setup_connection_pool()
        checkpointer = PostgresSaver(pool)
        # Compile the workflow with the checkpointer
        graph = workflow.compile(checkpointer=checkpointer)

        # Get thread_id
        thread_id = self._get_thread_id(request)
        
        # Store thread_id in custom_inputs for downstream use
        ci = dict(request.custom_inputs or {})
        ci["thread_id"] = thread_id
        request.custom_inputs = ci

        outputs = [
            event.item
            for message in request.input
            for event in graph.stream(
                {"messages": [("user", message["content"])]},
                config={"configurable": {"thread_id": thread_id}},
                stream_mode="values",
            )
        ]

        # Accumulate output messages
        out_messages = outputs[-1]["messages"]
        text_outputs = [
            AgentOutputText(text=m.content) for m in out_messages if m.content
        ]
        
        yield ResponsesAgentResponse(
            output=text_outputs,
            custom_outputs={"thread_id": thread_id},
        )

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """
        Non-streaming prediction.
        """
        pool = self._setup_connection_pool()
        checkpointer = PostgresSaver(pool)
        graph = workflow.compile(checkpointer=checkpointer)

        thread_id = self._get_thread_id(request)
        
        # Store thread_id in custom_inputs
        ci = dict(request.custom_inputs or {})
        ci["thread_id"] = thread_id
        request.custom_inputs = ci
        
        # Convert incoming Responses messages to ChatCompletions format
        # LangChain will automatically convert from ChatCompletions to LangChain format
        messages = [(msg["role"], msg["content"]) for msg in request.input]

        # Invoke the graph with the thread_id as the configuration
        final_state = graph.invoke(
            {"messages": messages},
            config={"configurable": {"thread_id": thread_id}},
        )

        # Extract outputs
        out_messages = final_state["messages"]
        text_outputs = [
            AgentOutputText(text=m.content) for m in out_messages if m.content
        ]

        return ResponsesAgentResponse(
            output=text_outputs,
            custom_outputs={"thread_id": thread_id},
        )

# Initialize the agent
AGENT = LangGraphResponsesAgent(lakebase_config=LAKEBASE_CONFIG)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test the agent

# COMMAND ----------

# MAGIC %md
# MAGIC Test the agent with a given thread id. Notice that you can resume conversation at any time by re-using the same thread id. 

# COMMAND ----------

from mlflow.types.agents import ResponsesAgentRequest, ChatContext

thread_id = str(uuid.uuid4())
print(f"Using thread_id: {thread_id}")

req = ResponsesAgentRequest(
    input=[{"role": "user", "content": "Hi, my name is John"}],
    custom_inputs={"thread_id": thread_id}
)
result = AGENT.predict(req)

print(result.model_dump(exclude_none=True))

# COMMAND ----------

# Continue the conversation with the same thread_id
conversation_id = result.custom_outputs["thread_id"]

req = ResponsesAgentRequest(input=[{"role": "user", "content": "What is my name?"}], custom_inputs={"thread_id": thread_id})
response2 = AGENT.predict(req)
print("Response 2:", response2.model_dump(exclude_none=True))

# COMMAND ----------

# Can also use streaming
for chunk in AGENT.predict_stream(ResponsesAgentRequest(
    input=[{"role": "user", "content": "Tell me more"}],
    custom_inputs={"thread_id": thread_id}
}):
    print("Chunk:", chunk.model_dump(exclude_none=True))

# COMMAND ----------

# Or initiate without a thread_id - agent will create a new one
req = ResponsesAgentRequest(
    input=[{"role": "user", "content": "What am I asking about?"}],
    context=ChatContext(
        conversation_id=conversation_id,
        user_id="email@databricks.com"
    )
)
result = AGENT.predict(req)

print(result.model_dump(exclude_none=True))
thread_id = result.custom_outputs["thread_id"]
print(f"Resolved thread_id from agent: {thread_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Log the agent as an MLflow model
# MAGIC Log the agent as code from the agent.py file. See [MLflow - Models from Code](https://mlflow.org/docs/latest/models.html#models-from-code).
# MAGIC

# COMMAND ----------

import mlflow

# Inference table is the destination for the payloads, trace data, and assessments
mlflow.set_registry_uri("databricks-uc")

agent_config = {
    "lakebase_config": LAKEBASE_CONFIG,
    "llm_endpoint_name": LLM_ENDPOINT_NAME,
    "system_prompt": SYSTEM_PROMPT,
}

# ============================================================================
# TODO SECTION 5: UNITY CATALOG MODEL REGISTRATION
# ============================================================================
# Define where to register your agent in Unity Catalog
# Format: catalog.schema.model_name
# 
# Requirements:
# - You must have CREATE MODEL permissions on the schema
# - The catalog and schema must already exist
# 
# Example:
# catalog = "my_team"
# schema = "agents"
# model_name = "customer-support-agent"
# ============================================================================

# TODO: define the catalog, schema, and model name for your UC model
catalog = "YOUR_CATALOG_NAME"  # e.g., "my_team", "production"
schema = "YOUR_SCHEMA_NAME"  # e.g., "agents", "ml_models"
model_name = "YOUR_MODEL_NAME"  # e.g., "stateful-agent-v1", "support-bot"

UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"

# ============================================================================

# Enable trace logging
mlflow.langchain.autolog(log_models=True)

# Log the model
with mlflow.start_run(run_name="LangGraph Agent with Lakebase"):
    logged_agent_info = mlflow.pyfunc.log_model(
        python_model=AGENT,
        artifact_path="agent",
        model_config=agent_config,
        # TODO: Optionally customize the input example
        input_example={
            "messages": [{"role": "user", "content": "Hello"}]
        },
        signature=False,
        # Declare Lakebase as a resource to enable automatic auth passthrough
        resources=[
            {
                "databricks_lakebase": {
                    "instance_name": LAKEBASE_CONFIG["instance_name"],
                }
            }
        ],
        extra_pip_requirements=[
            "langchain-core==0.3.29",
            "langchain==0.3.14",
            "langgraph==0.5.3",
            "langgraph-checkpoint==2.0.11",
            "langgraph-checkpoint-postgres==2.0.21",
            "psycopg[binary,pool]>=3.1.0",
            "databricks-langchain>=0.3.1",
        ],
        # TODO: Optionally add example_no_conversion if needed
        # example_no_conversion=True,
    )

# COMMAND ----------

# register the model to UC
uc_registered_model_info = mlflow.register_model(
    model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy the agent

# COMMAND ----------

from databricks import agents
agents.deploy(UC_MODEL_NAME, uc_registered_model_info.version, tags = {"endpointSource": "docs"})

# COMMAND ----------

# MAGIC %md
# MAGIC # Query your conversations from Lakebase
# MAGIC
# MAGIC Since all conversation state is stored in Lakebase, you can query it directly via SQL.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- TODO: Update the table name if you're using a different database or schema
# MAGIC -- Default: databricks_postgres.public.checkpoints
# MAGIC
# MAGIC -- View all conversation threads and their activity
# MAGIC SELECT 
# MAGIC     thread_id,
# MAGIC     COUNT(*) as checkpoint_count,
# MAGIC     MIN((checkpoint::json->>'ts')::timestamptz) as first_message,
# MAGIC     MAX((checkpoint::json->>'ts')::timestamptz) as last_message
# MAGIC FROM databricks_postgres.public.checkpoints
# MAGIC GROUP BY thread_id
# MAGIC ORDER BY last_message DESC
# MAGIC LIMIT 20;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- TODO: Replace 'your-thread-id-here' with an actual thread_id from the query above
# MAGIC
# MAGIC -- View conversation details for a specific thread
# MAGIC SELECT 
# MAGIC     thread_id,
# MAGIC     checkpoint,
# MAGIC     (checkpoint::json->>'ts')::timestamptz as timestamp
# MAGIC FROM databricks_postgres.public.checkpoints
# MAGIC WHERE thread_id = 'your-thread-id-here'
# MAGIC ORDER BY timestamp;

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Setup Complete!
# MAGIC
# MAGIC Your stateful agent is now deployed and ready to use. You can:
# MAGIC 1. Test it in the Databricks AI Playground
# MAGIC 2. Call it via the Model Serving API
# MAGIC 3. Integrate it into apps using the endpoint name
# MAGIC 4. Query conversation history via SQL
# MAGIC
# MAGIC ### Next Steps:
# MAGIC - Deploy a Streamlit app to provide a UI for your agent
# MAGIC - Set up monitoring dashboards using conversation data from Lakebase
# MAGIC - Implement feedback loops and evaluation metrics
# MAGIC
# MAGIC ### Important Notes:
# MAGIC - All conversations are stored in Lakebase (databricks_postgres.public.checkpoints)
# MAGIC - Thread IDs uniquely identify each conversation
# MAGIC - The agent automatically creates checkpoints after each step
# MAGIC - You can resume any conversation by passing the same thread_id

