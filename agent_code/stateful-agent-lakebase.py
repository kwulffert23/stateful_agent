# Databricks notebook source
# MAGIC %md
# MAGIC # Mosaic AI Agent Framework: Author and deploy a Stateful Agent with Thread-scoped memory using Databricks Lakebase and LangGraph
# MAGIC This notebook demonstrates how to build a stateful agent using the Mosaic AI Agent Framework and LangGraph, with Lakebase as the agent’s durable memory and checkpoint store. Threads allow you to store conversational state in Lakebase so you can pass in thread IDs into your agent instead of needing to send the full conversation history.
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

# TODO: Fill in your lakebase instance details here. For the username, create a
# Service Principal and grant it databricks_superuser permissions onto the lakebase instance.
# See Service Principal Documentation for more information:
# https://docs.databricks.com/en/admin/users-groups/service-principals
# Use the Service Principal client id and secret as the SP_CLIENT_ID/SP_CLIENT_SECRET
# This will help initialize the checkpointers
DB_INSTANCE_NAME = "<TODO>"  
DB_NAME          = "databricks_postgres"
SP_CLIENT_ID      = "<TODO>"
SP_CLIENT_SECRET      = "<TODO>"
SSL_MODE         = "require"
DB_HOST = "<TODO>"
DB_PORT = 5432
WORKSPACE_HOST = "<TODO>"

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
    open=True,
)

# Use the pool to initialize your checkpoint tables
with pool.connection() as conn:
    conn.autocommit = True   # disable transaction wrapping
    checkpointer = PostgresSaver(conn)
    checkpointer.setup()
    conn.autocommit = False  # restore default if you want transactions later

    with conn.cursor() as cur:
        cur.execute("select 1")
    print("✅ Pool connected and checkpoint tables are ready.")

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # Define the agent in code
# MAGIC
# MAGIC ## Write agent code to file agent.py
# MAGIC Define the agent code in a single cell below. This lets you write the agent code to a local Python file, using the `%%writefile` magic command, for subsequent logging and deployment.
# MAGIC
# MAGIC ## Wrap the LangGraph agent using the ResponsesAgent interface
# MAGIC For compatibility with Databricks AI features, the `LangGraphResponsesAgent` class implements the `ResponsesAgent` interface to wrap the LangGraph agent.
# MAGIC
# MAGIC Databricks recommends using `ResponsesAgent` as it simplifies authoring multi-turn conversational agents using an open source standard. See MLflow's [ResponsesAgent documentation](https://www.mlflow.org/docs/latest/llms/responses-agent-intro/).

# COMMAND ----------

# MAGIC %%writefile agent.py
# MAGIC import json
# MAGIC import logging
# MAGIC import os
# MAGIC import time
# MAGIC import urllib.parse
# MAGIC import uuid
# MAGIC from threading import Lock
# MAGIC from typing import Annotated, Any, Generator, Optional, Sequence, TypedDict
# MAGIC
# MAGIC import mlflow
# MAGIC from databricks_langchain import (
# MAGIC     ChatDatabricks,
# MAGIC     VectorSearchRetrieverTool,
# MAGIC     DatabricksFunctionClient,
# MAGIC     UCFunctionToolkit,
# MAGIC )
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC from langchain_core.messages import (
# MAGIC     AIMessage,
# MAGIC     AIMessageChunk,
# MAGIC     BaseMessage,
# MAGIC )
# MAGIC from langchain_core.runnables import RunnableConfig, RunnableLambda
# MAGIC from langgraph.checkpoint.postgres import PostgresSaver
# MAGIC from langgraph.graph import END, StateGraph
# MAGIC from langgraph.graph.message import add_messages
# MAGIC from langgraph.prebuilt.tool_node import ToolNode
# MAGIC from mlflow.pyfunc import ResponsesAgent
# MAGIC from mlflow.types.responses import (
# MAGIC     ResponsesAgentRequest,
# MAGIC     ResponsesAgentResponse,
# MAGIC     ResponsesAgentStreamEvent,
# MAGIC )
# MAGIC import psycopg
# MAGIC from psycopg_pool import ConnectionPool
# MAGIC from psycopg.rows import dict_row
# MAGIC from contextlib import contextmanager
# MAGIC
# MAGIC logger = logging.getLogger(__name__)
# MAGIC logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
# MAGIC
# MAGIC
# MAGIC ############################################
# MAGIC # Define your LLM endpoint and system prompt
# MAGIC ############################################
# MAGIC # TODO: Replace with your model serving endpoint
# MAGIC LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"
# MAGIC
# MAGIC # TODO: Update with your system prompt
# MAGIC SYSTEM_PROMPT = "You are a helpful assistant. Use the available tools to answer questions."
# MAGIC
# MAGIC # TODO: Fill in Lakebase config values here
# MAGIC LAKEBASE_CONFIG = {
# MAGIC     "instance_name": "<TODO>",
# MAGIC     "conn_host": "<TODO>",
# MAGIC     "conn_db_name": "databricks_postgres",
# MAGIC     "conn_ssl_mode": "require",
# MAGIC }
# MAGIC
# MAGIC ###############################################################################
# MAGIC ## Define tools for your agent,enabling it to retrieve data or take actions
# MAGIC ## beyond text generation
# MAGIC ## To create and see usage examples of more tools, see
# MAGIC ## https://docs.databricks.com/en/generative-ai/agent-framework/agent-tool.html
# MAGIC ###############################################################################
# MAGIC
# MAGIC tools = []
# MAGIC
# MAGIC # Example UC tools; add your own as needed
# MAGIC UC_TOOL_NAMES: list[str] = []
# MAGIC if UC_TOOL_NAMES:
# MAGIC     uc_toolkit = UCFunctionToolkit(function_names=UC_TOOL_NAMES)
# MAGIC     tools.extend(uc_toolkit.tools)
# MAGIC
# MAGIC # Use Databricks vector search indexes as tools
# MAGIC # See https://docs.databricks.com/en/generative-ai/agent-framework/unstructured-retrieval-tools.html#locally-develop-vector-search-retriever-tools-with-ai-bridge
# MAGIC # List to store vector search tool instances for unstructured retrieval.
# MAGIC VECTOR_SEARCH_TOOLS = []
# MAGIC
# MAGIC # To add vector search retriever tools,
# MAGIC # use VectorSearchRetrieverTool and create_tool_info,
# MAGIC # then append the result to TOOL_INFOS.
# MAGIC # Example:
# MAGIC VECTOR_SEARCH_TOOLS.append(
# MAGIC     VectorSearchRetrieverTool(
# MAGIC         index_name="<TODO>",
# MAGIC         tool_description="<TODO>s"
# MAGIC     )
# MAGIC )
# MAGIC
# MAGIC tools.extend(VECTOR_SEARCH_TOOLS)
# MAGIC
# MAGIC #####################
# MAGIC ## Define agent logic
# MAGIC #####################
# MAGIC
# MAGIC
# MAGIC class AgentState(TypedDict):
# MAGIC     messages: Annotated[Sequence[BaseMessage], add_messages]
# MAGIC     custom_inputs: Optional[dict[str, Any]]
# MAGIC     custom_outputs: Optional[dict[str, Any]]
# MAGIC
# MAGIC
# MAGIC class CredentialConnection(psycopg.Connection):
# MAGIC     """Custom connection class that generates fresh OAuth tokens with caching."""
# MAGIC     
# MAGIC     workspace_client = None
# MAGIC     instance_name = None
# MAGIC     
# MAGIC     # Cache attributes
# MAGIC     _cached_credential = None
# MAGIC     _cache_timestamp = None
# MAGIC     _cache_duration = 3000  # 50 minutes in seconds (50 * 60)
# MAGIC     _cache_lock = Lock()
# MAGIC     
# MAGIC     @classmethod
# MAGIC     def connect(cls, conninfo='', **kwargs):
# MAGIC         """Override connect to inject OAuth token with 50-minute caching"""
# MAGIC         if cls.workspace_client is None or cls.instance_name is None:
# MAGIC             raise ValueError("workspace_client and instance_name must be set on CredentialConnection class")
# MAGIC         
# MAGIC         # Get cached or fresh credential and append the new password to kwargs
# MAGIC         credential_token = cls._get_cached_credential()
# MAGIC         kwargs['password'] = credential_token
# MAGIC         
# MAGIC         # Call the superclass's connect method with updated kwargs
# MAGIC         return super().connect(conninfo, **kwargs)
# MAGIC     
# MAGIC     @classmethod
# MAGIC     def _get_cached_credential(cls):
# MAGIC         """Get credential from cache or generate a new one if cache is expired"""
# MAGIC         with cls._cache_lock:
# MAGIC             current_time = time.time()
# MAGIC             
# MAGIC             # Check if we have a valid cached credential
# MAGIC             if (cls._cached_credential is not None and 
# MAGIC                 cls._cache_timestamp is not None and 
# MAGIC                 current_time - cls._cache_timestamp < cls._cache_duration):
# MAGIC                 return cls._cached_credential
# MAGIC             
# MAGIC             # Generate new credential
# MAGIC             credential = cls.workspace_client.database.generate_database_credential(
# MAGIC                 request_id=str(uuid.uuid4()),
# MAGIC                 instance_names=[cls.instance_name]
# MAGIC             )
# MAGIC             
# MAGIC             # Cache the new credential
# MAGIC             cls._cached_credential = credential.token
# MAGIC             cls._cache_timestamp = current_time
# MAGIC             
# MAGIC             return cls._cached_credential
# MAGIC
# MAGIC
# MAGIC class LangGraphResponsesAgent(ResponsesAgent):
# MAGIC     """Stateful agent using ResponsesAgent with Lakebase PostgreSQL checkpointing.
# MAGIC     
# MAGIC     Features:
# MAGIC     - Connection pooling with credential rotation and caching
# MAGIC     - Thread-based conversation state persistence
# MAGIC     - Tool support with UC functions
# MAGIC     """
# MAGIC
# MAGIC     def __init__(self, lakebase_config: dict[str, Any]):
# MAGIC         self.lakebase_config = lakebase_config
# MAGIC         self.workspace_client = WorkspaceClient()
# MAGIC         
# MAGIC         # Model and tools
# MAGIC         self.model = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)
# MAGIC         self.system_prompt = SYSTEM_PROMPT
# MAGIC         self.model_with_tools = self.model.bind_tools(tools) if tools else self.model
# MAGIC         
# MAGIC         # Connection pool configuration
# MAGIC         self.pool_min_size = int(os.getenv("DB_POOL_MIN_SIZE", "1"))
# MAGIC         self.pool_max_size = int(os.getenv("DB_POOL_MAX_SIZE", "10"))
# MAGIC         self.pool_timeout = float(os.getenv("DB_POOL_TIMEOUT", "30.0"))
# MAGIC         
# MAGIC         # Token cache duration (in minutes, can be overridden via env var)
# MAGIC         cache_duration_minutes = int(os.getenv("DB_TOKEN_CACHE_MINUTES", "50"))
# MAGIC         CredentialConnection._cache_duration = cache_duration_minutes * 60
# MAGIC         
# MAGIC         # Initialize the connection pool with rotating credentials
# MAGIC         self._connection_pool = self._create_rotating_pool()
# MAGIC         
# MAGIC         mlflow.langchain.autolog()
# MAGIC
# MAGIC     def _get_username(self) -> str:
# MAGIC         """Get the username for database connection"""
# MAGIC         try:
# MAGIC             sp = self.workspace_client.current_service_principal.me()
# MAGIC             return sp.application_id
# MAGIC         except Exception:
# MAGIC             user = self.workspace_client.current_user.me()
# MAGIC             return user.user_name
# MAGIC
# MAGIC     def _create_rotating_pool(self) -> ConnectionPool:
# MAGIC         """Create a connection pool that automatically rotates credentials with caching"""
# MAGIC         # Set the workspace client and instance name on the custom connection class
# MAGIC         CredentialConnection.workspace_client = self.workspace_client
# MAGIC         CredentialConnection.instance_name = self.lakebase_config["instance_name"]
# MAGIC         
# MAGIC         username = self._get_username()
# MAGIC         host = self.lakebase_config["conn_host"]
# MAGIC         database = self.lakebase_config.get("conn_db_name", "databricks_postgres")
# MAGIC         
# MAGIC         # Create pool with custom connection class
# MAGIC         pool = ConnectionPool(
# MAGIC             conninfo=f"dbname={database} user={username} host={host} sslmode=require",
# MAGIC             connection_class=CredentialConnection,
# MAGIC             min_size=self.pool_min_size,
# MAGIC             max_size=self.pool_max_size,
# MAGIC             timeout=self.pool_timeout,
# MAGIC             open=True,
# MAGIC             kwargs={
# MAGIC                 "autocommit": True, # Required for the .setup() method to properly commit the checkpoint tables to the database
# MAGIC                 "row_factory": dict_row, # Required because the PostgresSaver implementation accesses database rows using dictionary-style syntax
# MAGIC                 "keepalives": 1,
# MAGIC                 "keepalives_idle": 30,
# MAGIC                 "keepalives_interval": 10,
# MAGIC                 "keepalives_count": 5,
# MAGIC             }
# MAGIC         )
# MAGIC         
# MAGIC         # Test the pool
# MAGIC         try:
# MAGIC             with pool.connection() as conn:
# MAGIC                 with conn.cursor() as cursor:
# MAGIC                     cursor.execute("SELECT 1")
# MAGIC             logger.info(
# MAGIC                 f"Connection pool with rotating credentials created successfully "
# MAGIC                 f"(min={self.pool_min_size}, max={self.pool_max_size}, "
# MAGIC                 f"token_cache={CredentialConnection._cache_duration / 60:.0f} minutes)"
# MAGIC             )
# MAGIC         except Exception as e:
# MAGIC             pool.close()
# MAGIC             raise ConnectionError(f"Failed to create connection pool: {e}")
# MAGIC         
# MAGIC         return pool
# MAGIC     
# MAGIC     @contextmanager
# MAGIC     def get_connection(self):
# MAGIC         """Context manager to get a connection from the pool"""
# MAGIC         with self._connection_pool.connection() as conn:
# MAGIC             yield conn
# MAGIC     
# MAGIC     def _langchain_to_responses(self, messages: list[BaseMessage]) -> list[dict[str, Any]]:
# MAGIC         """Convert from LangChain messages to Responses API format"""
# MAGIC         responses = []
# MAGIC         for message in messages:
# MAGIC             message_dict = message.model_dump()
# MAGIC             msg_type = message_dict["type"]
# MAGIC             
# MAGIC             if msg_type == "ai":
# MAGIC                 if tool_calls := message_dict.get("tool_calls"):
# MAGIC                     for tool_call in tool_calls:
# MAGIC                         responses.append(
# MAGIC                             self.create_function_call_item(
# MAGIC                                 id=message_dict.get("id") or str(uuid.uuid4()),
# MAGIC                                 call_id=tool_call["id"],
# MAGIC                                 name=tool_call["name"],
# MAGIC                                 arguments=json.dumps(tool_call["args"]),
# MAGIC                             )
# MAGIC                         )
# MAGIC                 else:
# MAGIC                     responses.append(
# MAGIC                         self.create_text_output_item(
# MAGIC                             text=message_dict.get("content", ""),
# MAGIC                             id=message_dict.get("id") or str(uuid.uuid4()),
# MAGIC                         )
# MAGIC                     )
# MAGIC             elif msg_type == "tool":
# MAGIC                 responses.append(
# MAGIC                     self.create_function_call_output_item(
# MAGIC                         call_id=message_dict["tool_call_id"],
# MAGIC                         output=message_dict["content"],
# MAGIC                     )
# MAGIC                 )
# MAGIC             elif msg_type == "human":
# MAGIC                 responses.append({
# MAGIC                     "role": "user",
# MAGIC                     "content": message_dict.get("content", "")
# MAGIC                 })
# MAGIC         
# MAGIC         return responses
# MAGIC     
# MAGIC     def _create_graph(self, checkpointer: PostgresSaver):
# MAGIC         """Create the LangGraph workflow"""
# MAGIC         def should_continue(state: AgentState):
# MAGIC             messages = state["messages"]
# MAGIC             last_message = messages[-1]
# MAGIC             if isinstance(last_message, AIMessage) and last_message.tool_calls:
# MAGIC                 return "continue"
# MAGIC             return "end"
# MAGIC         
# MAGIC         if self.system_prompt:
# MAGIC             preprocessor = RunnableLambda(
# MAGIC                 lambda state: [{"role": "system", "content": self.system_prompt}] + state["messages"]
# MAGIC             )
# MAGIC         else:
# MAGIC             preprocessor = RunnableLambda(lambda state: state["messages"])
# MAGIC         
# MAGIC         model_runnable = preprocessor | self.model_with_tools
# MAGIC         
# MAGIC         def call_model(state: AgentState, config: RunnableConfig):
# MAGIC             response = model_runnable.invoke(state, config)
# MAGIC             return {"messages": [response]}
# MAGIC         
# MAGIC         workflow = StateGraph(AgentState)
# MAGIC         workflow.add_node("agent", RunnableLambda(call_model))
# MAGIC         
# MAGIC         if tools:
# MAGIC             workflow.add_node("tools", ToolNode(tools))
# MAGIC             workflow.add_conditional_edges(
# MAGIC                 "agent",
# MAGIC                 should_continue,
# MAGIC                 {"continue": "tools", "end": END}
# MAGIC             )
# MAGIC             workflow.add_edge("tools", "agent")
# MAGIC         else:
# MAGIC             workflow.add_edge("agent", END)
# MAGIC         
# MAGIC         workflow.set_entry_point("agent")
# MAGIC         
# MAGIC         return workflow.compile(checkpointer=checkpointer)
# MAGIC
# MAGIC     def _get_or_create_thread_id(self, request: ResponsesAgentRequest) -> str:
# MAGIC         """Get thread_id from request or create a new one.
# MAGIC         
# MAGIC         Priority:
# MAGIC         1. Use thread_id from custom_inputs if present
# MAGIC         2. Use conversation_id from chat context if available
# MAGIC         3. Generate a new UUID
# MAGIC         
# MAGIC         Returns:
# MAGIC             thread_id: The thread identifier to use for this conversation
# MAGIC         """
# MAGIC         ci = dict(request.custom_inputs or {})
# MAGIC         
# MAGIC         if "thread_id" in ci:
# MAGIC             return ci["thread_id"]
# MAGIC         
# MAGIC         # using conversation id from chat context as thread id
# MAGIC         # https://mlflow.org/docs/latest/api_reference/python_api/mlflow.types.html#mlflow.types.agent.ChatContext
# MAGIC         if request.context and getattr(request.context, "conversation_id", None):
# MAGIC             return request.context.conversation_id
# MAGIC         
# MAGIC         # Generate new thread_id
# MAGIC         return str(uuid.uuid4())
# MAGIC     
# MAGIC     def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
# MAGIC         """Non-streaming prediction"""
# MAGIC         thread_id = self._get_or_create_thread_id(request)
# MAGIC
# MAGIC         ci = dict(request.custom_inputs or {})
# MAGIC         ci["thread_id"] = thread_id
# MAGIC         request.custom_inputs = ci
# MAGIC
# MAGIC         outputs = [
# MAGIC             event.item
# MAGIC             for event in self.predict_stream(request)
# MAGIC             if event.type == "response.output_item.done"
# MAGIC         ]
# MAGIC         return ResponsesAgentResponse(output=outputs, custom_outputs={"thread_id": ci["thread_id"]})
# MAGIC     
# MAGIC     def predict_stream(
# MAGIC         self,
# MAGIC         request: ResponsesAgentRequest,
# MAGIC     ) -> Generator[ResponsesAgentStreamEvent, None, None]:
# MAGIC         """Streaming prediction with PostgreSQL checkpointing"""
# MAGIC         thread_id = self._get_or_create_thread_id(request)
# MAGIC         
# MAGIC         ci = dict(request.custom_inputs or {})
# MAGIC         ci["thread_id"] = thread_id
# MAGIC         request.custom_inputs = ci
# MAGIC         
# MAGIC         # Convert incoming Responses messages to ChatCompletions format
# MAGIC         # LangChain will automatically convert from ChatCompletions to LangChain format
# MAGIC         cc_msgs = self.prep_msgs_for_cc_llm([i.model_dump() for i in request.input])
# MAGIC         langchain_msgs = cc_msgs
# MAGIC         
# MAGIC         checkpoint_config = {"configurable": {"thread_id": thread_id}}
# MAGIC         
# MAGIC         # Use connection from pool
# MAGIC         with self.get_connection() as conn:            
# MAGIC             # Create checkpointer and graph
# MAGIC             checkpointer = PostgresSaver(conn)
# MAGIC             graph = self._create_graph(checkpointer)
# MAGIC             
# MAGIC             # Stream the graph execution
# MAGIC             for event in graph.stream(
# MAGIC                 {"messages": langchain_msgs},
# MAGIC                 checkpoint_config,
# MAGIC                 stream_mode=["updates", "messages"]
# MAGIC             ):
# MAGIC                 if event[0] == "updates":
# MAGIC                     for node_data in event[1].values():
# MAGIC                         for item in self._langchain_to_responses(node_data["messages"]):
# MAGIC                             yield ResponsesAgentStreamEvent(
# MAGIC                                 type="response.output_item.done",
# MAGIC                                 item=item
# MAGIC                             )
# MAGIC                 # Stream message chunks for real-time text generation
# MAGIC                 elif event[0] == "messages":
# MAGIC                     try:
# MAGIC                         chunk = event[1][0]
# MAGIC                         if isinstance(chunk, AIMessageChunk) and chunk.content:
# MAGIC                             yield ResponsesAgentStreamEvent(
# MAGIC                                 **self.create_text_delta(
# MAGIC                                     delta=chunk.content,
# MAGIC                                     item_id=chunk.id
# MAGIC                                 ),
# MAGIC                             )
# MAGIC                     except Exception as e:
# MAGIC                         logger.error(f"Error streaming chunk: {e}")
# MAGIC
# MAGIC
# MAGIC # ----- Export model -----
# MAGIC AGENT = LangGraphResponsesAgent(LAKEBASE_CONFIG)
# MAGIC mlflow.models.set_model(AGENT)

# COMMAND ----------

# MAGIC %md
# MAGIC # Test the Agent locally

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from agent import AGENT
# Message 1, don't include thread_id (creates new thread)
result = AGENT.predict({
    "input": [{"role": "user", "content": "how long does the anti-rust paint work for"}]
})
print(result.model_dump(exclude_none=True))
thread_id = result.custom_outputs["thread_id"]

# COMMAND ----------

# Message 2, include thread ID and notice how agent remembers context from previous predict message
response2 = AGENT.predict({
    "input": [{"role": "user", "content": "What am I asking about?"}],
    "custom_inputs": {"thread_id": thread_id}
})
print("Response 2:", response2.model_dump(exclude_none=True))

# COMMAND ----------

# Example calling agent without passing in thread id - notice it does not retain the memory
response3 = AGENT.predict({
    "input": [{"role": "user", "content": "What am I asking about?"}],
})
print("Response 3 No thread id passed:", response3.model_dump(exclude_none=True))

# COMMAND ----------

# predict stream example
for chunk in AGENT.predict_stream({
    "input": [{"role": "user", "content": "What am I asking about?"}],
    "custom_inputs": {"thread_id": thread_id}
}):
    print("Chunk:", chunk.model_dump(exclude_none=True))

# COMMAND ----------

result.custom_outputs

# COMMAND ----------

# example using conversation_id from ChatContext as thread_id
# https://mlflow.org/docs/latest/api_reference/python_api/mlflow.types.html#mlflow.types.agent.ChatContext
from agent import AGENT
import mlflow
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ChatContext
)

conversation_id = result.custom_outputs["thread_id"]

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
# MAGIC ## Enable automatic authentication for Databricks resources
# MAGIC For the most common Databricks resource types, Databricks supports and recommends declaring resource dependencies for the agent upfront during logging. This enables automatic authentication passthrough when you deploy the agent. With automatic authentication passthrough, Databricks automatically provisions, rotates, and manages short-lived credentials to securely access these resource dependencies from within the agent endpoint.
# MAGIC
# MAGIC To enable automatic authentication, specify the dependent Databricks resources when calling `mlflow.pyfunc.log_model()`.
# MAGIC
# MAGIC **TODO:** 
# MAGIC - Add lakebase as a resource type
# MAGIC - If your Unity Catalog tool queries a [vector search index](https://docs.databricks.com/docs%20link) or leverages [external functions](https://docs.databricks.com/docs%20link), you need to include the dependent vector search index and UC connection objects, respectively, as resources. See docs ([AWS](https://docs.databricks.com/generative-ai/agent-framework/log-agent.html#specify-resources-for-automatic-authentication-passthrough) | [Azure](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/log-agent#resources)).

# COMMAND ----------

# Determine Databricks resources to specify for automatic auth passthrough at deployment time
import mlflow
from agent import tools, LLM_ENDPOINT_NAME, LAKEBASE_CONFIG
from databricks_langchain import VectorSearchRetrieverTool
from mlflow.models.resources import DatabricksFunction, DatabricksServingEndpoint, DatabricksLakebase,DatabricksVectorSearchIndex
from unitycatalog.ai.langchain.toolkit import UnityCatalogTool
from pkg_resources import get_distribution

resources = [
    DatabricksServingEndpoint(LLM_ENDPOINT_NAME), 
    DatabricksVectorSearchIndex(index_name="<TODO>"),
    DatabricksLakebase(database_instance_name=LAKEBASE_CONFIG["instance_name"])
    ]

for tool in tools:
    if isinstance(tool, VectorSearchRetrieverTool):
        resources.extend(tool.resources)
    elif isinstance(tool, UnityCatalogTool):
        resources.append(DatabricksFunction(function_name=tool.uc_function_name))

input_example = {
    "input": [
        {
            "role": "user",
            "content": "What is an LLM agent?"
        }
    ],
    "custom_inputs": {"thread_id": "example-thread-123"},
}

with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model="agent.py",
        input_example=input_example,
        resources=resources,
        pip_requirements=[
            "databricks-langchain",
            f"databricks-connect=={get_distribution('databricks-connect').version}",
            f"langgraph=={get_distribution('langgraph').version}",
            f"langgraph-checkpoint-postgres=={get_distribution('langgraph-checkpoint-postgres').version}",
            f"psycopg[binary,pool]",
            f"pydantic=={get_distribution('pydantic').version}",
        ]
    )

# COMMAND ----------

# MAGIC %md
# MAGIC # Evaluate the agent with Agent Evaluation
# MAGIC Use Mosaic AI Agent Evaluation to evalaute the agent's responses based on expected responses and other evaluation criteria. Use the evaluation criteria you specify to guide iterations, using MLflow to track the computed quality metrics. See Databricks documentation ([AWS](https://docs.databricks.com/(https://docs.databricks.com/aws/generative-ai/agent-evaluation) | [Azure](https://learn.microsoft.com/azure/databricks/generative-ai/agent-evaluation/)).
# MAGIC
# MAGIC To evaluate your tool calls, add custom metrics. See Databricks documentation ([AWS](https://docs.databricks.com/en/generative-ai/agent-evaluation/custom-metrics.html#evaluating-tool-calls) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-evaluation/custom-metrics#evaluating-tool-calls)).

# COMMAND ----------

import mlflow
from mlflow.genai.scorers import RelevanceToQuery, RetrievalGroundedness, RetrievalRelevance, Safety

eval_dataset = [
    {
        "inputs": {"input": [{"role": "user", "content": "What are the features and benefits of Rain Shield Exterior paint?"}]},
        "expected_response": "Rain Shield Exterior is designed as a premium, long-lasting solution for exterior surfaces that need reliable protection against weather, UV rays, and mildew while maintaining excellent appearance for up to a decade.",
    }
]

eval_results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=lambda input: AGENT.predict({"input": input}),
    scorers=[RelevanceToQuery(), Safety()],  # add more scorers here if they're applicable
)

# Review the evaluation results in the MLfLow UI (see console output)

# COMMAND ----------

# MAGIC %md
# MAGIC # Pre-deployment agent validation
# MAGIC Before registering and deploying the agent, perform pre-deployment checks using the mlflow.models.predict() API.

# COMMAND ----------

mlflow.models.predict(
    model_uri=f"runs:/{logged_agent_info.run_id}/agent",
    input_data={"input": [{"role": "user", "content": "What am I asking about"}]},
    env_manager="uv",
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Register the model to Unity Catalog
# MAGIC Update the `catalog`, `schema`, and `model_name` below to register the MLflow model to Unity Catalog.

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# TODO: define the catalog, schema, and model name for your UC model
catalog = "<TODO>"
schema = "<TODO>"
model_name = "<TODO>"

UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"

# register the model to UC
uc_registered_model_info = mlflow.register_model(
    model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME
)

# COMMAND ----------

# MAGIC %md
# MAGIC Deploy the agent

# COMMAND ----------

from databricks import agents
agents.deploy(UC_MODEL_NAME, uc_registered_model_info.version, tags = {"endpointSource": "docs"})

# COMMAND ----------

# MAGIC %md
# MAGIC # Next steps
# MAGIC It will take around 15 minutes for you to finish deploying your agent. After your agent is deployed, you can chat with it in AI playground to perform additional checks, share it with SMEs in your organization for feedback, or embed it in a production application. 
# MAGIC
# MAGIC Now, with your stateful agent, you can pick up past threads and continue the conversation.
# MAGIC
# MAGIC You can query your Lakebase instance to see a record of your conversation at various threads/checkpoints. Here is a basic query to see 10 checkpoints:
# MAGIC ```
# MAGIC -- See all conversation threads with their metadata
# MAGIC SELECT 
# MAGIC     *
# MAGIC FROM checkpoints
# MAGIC LIMIT 10;
# MAGIC ```
# MAGIC
# MAGIC Check most recently logged checkpoints:
# MAGIC ```
# MAGIC SELECT
# MAGIC     c.*,
# MAGIC     (c.checkpoint::json->>'ts')::timestamptz AS ts
# MAGIC FROM checkpoints c
# MAGIC ORDER BY ts DESC
# MAGIC LIMIT 10;
# MAGIC ```