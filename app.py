import logging
import os
import streamlit as st
from model_serving_utils import query_endpoint, is_endpoint_supported
from lakebase_history_utils import (
    get_thread_list, 
    update_session_metadata, 
    create_metadata_table,
    check_metadata_table_exists,
    get_thread_messages
)
from datetime import datetime
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure environment variable is set correctly
SERVING_ENDPOINT = os.getenv('SERVING_ENDPOINT')
assert SERVING_ENDPOINT, \
    ("Unable to determine serving endpoint to use for chatbot app. If developing locally, "
     "set the SERVING_ENDPOINT environment variable to the name of your serving endpoint. If "
     "deploying to a Databricks app, include a serving endpoint resource named "
     "'serving_endpoint' with CAN_QUERY permissions, as described in "
     "https://docs.databricks.com/aws/en/generative-ai/agent-framework/chat-app#deploy-the-databricks-app")

# Lakebase configuration
LAKEBASE_INSTANCE_NAME = os.getenv('LAKEBASE_INSTANCE_NAME', '<TODO>')
LAKEBASE_HOST = os.getenv('LAKEBASE_HOST', '<TODO>')
LAKEBASE_ENABLED = os.getenv('LAKEBASE_ENABLED', 'true').lower() == 'true'

# Check if the endpoint is supported
endpoint_supported = is_endpoint_supported(SERVING_ENDPOINT)

# Check if metadata table exists and ensure schema is up to date
metadata_table_exists = check_metadata_table_exists() if LAKEBASE_ENABLED else False
if LAKEBASE_ENABLED and metadata_table_exists:
    # Run migration to ensure messages column exists
    create_metadata_table()
logger.info(f"Lakebase enabled: {LAKEBASE_ENABLED}, Metadata table exists: {metadata_table_exists}")

def get_user_info():
    headers = st.context.headers
    return dict(
        user_name=headers.get("X-Forwarded-Preferred-Username"),
        user_email=headers.get("X-Forwarded-Email"),
        user_id=headers.get("X-Forwarded-User"),
    )

user_info = get_user_info()

# Initialize session state
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for conversation management
with st.sidebar:
    st.header("Conversation")
    
    # Button to start a new conversation
    if st.button("+ New Conversation", use_container_width=True):
        st.session_state.thread_id = None
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    # Lakebase setup section
    if LAKEBASE_ENABLED:
        if not metadata_table_exists:
            st.warning("⚠️ Metadata table not found")
            if st.button("🔧 Create Metadata Table", use_container_width=True):
                with st.spinner("Creating table..."):
                    if create_metadata_table():
                        st.success("✅ Table created successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to create table. Check logs.")
            st.divider()
        else:
            # Show conversation history from Lakebase
            st.subheader("📜 Your Conversations")
            
            try:
                threads = get_thread_list(user_info.get('user_id', 'unknown'), limit=10)
                
                if threads:
                    for thread in threads:
                        # Truncate first_query for display
                        display_text = thread['first_query'][:50] + "..." if thread['first_query'] and len(thread['first_query']) > 50 else thread['first_query']
                        
                        # Create a button for each conversation
                        if st.button(
                            f"💬 {display_text or 'Untitled'}",
                            key=f"load_{thread['thread_id']}",
                            use_container_width=True,
                            help=f"Last updated: {thread['last_updated']}"
                        ):
                            st.session_state.thread_id = thread['thread_id']
                            # Load messages from metadata for UI display
                            st.session_state.messages = get_thread_messages(thread['thread_id'])
                            st.rerun()
                else:
                    st.info("No previous conversations found")
            except Exception as e:
                logger.error(f"Error loading thread list: {e}")
                st.error("Failed to load conversation history")
            
            st.divider()
    
    # Info about history
    st.subheader("About Chat History")
    if LAKEBASE_ENABLED and metadata_table_exists:
        st.info(
            "✅ Your conversations are automatically saved to Lakebase. "
            "Click any conversation above to resume it!"
        )
    else:
        st.info(
            "Your conversations are tracked with unique thread IDs. "
            "History will be available once Lakebase is set up."
        )

# Main app
st.title("Stateful Chatbot with History")

# Check if endpoint is supported and show appropriate UI
if not endpoint_supported:
    st.error("Unsupported Endpoint Type")
    st.markdown(
        f"The endpoint `{SERVING_ENDPOINT}` is not compatible with this basic chatbot template.\n\n"
        "This template only supports chat completions-compatible endpoints.\n\n"
        "For a richer chatbot template that supports all conversational endpoints on Databricks, "
        "please see the [Databricks documentation](https://docs.databricks.com/aws/en/generative-ai/agent-framework/chat-app)."
    )
else:
    st.markdown(
        "Enhanced with chat history from Lakebase. See "
        "[Databricks docs](https://docs.databricks.com/aws/en/generative-ai/agent-framework/chat-app) "
        "for more information."
    )
    
    # Show thread status
    if st.session_state.thread_id:
        st.info(f"Continuing conversation (Thread: {st.session_state.thread_id[:8]}...)")
    else:
        st.info("Starting a new conversation")

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Generate thread_id if this is a new conversation
        if not st.session_state.thread_id:
            st.session_state.thread_id = str(uuid.uuid4())
            logger.info(f"Created new thread_id: {st.session_state.thread_id}")
        
        # Determine if this is the first message (for metadata)
        is_first_message = len(st.session_state.messages) == 0
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            # Query the Databricks serving endpoint with thread_id
            try:
                # For stateful agents, pass thread_id in custom_inputs
                response = query_endpoint(
                    endpoint_name=SERVING_ENDPOINT,
                    messages=st.session_state.messages,
                    max_tokens=400,
                    thread_id=st.session_state.thread_id  # Pass thread_id for stateful agents
                )
                assistant_response = response["content"]
                st.markdown(assistant_response)
            except Exception as e:
                logger.error(f"Error querying endpoint: {e}")
                # Fallback without custom_inputs if endpoint doesn't support it
                try:
                    assistant_response = query_endpoint(
                        endpoint_name=SERVING_ENDPOINT,
                        messages=st.session_state.messages,
                        max_tokens=400,
                    )["content"]
                    st.markdown(assistant_response)
                except Exception as e2:
                    st.error(f"Failed to get response: {e2}")
                    assistant_response = "Sorry, I encountered an error."

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        
        # Update session metadata in Lakebase (if enabled and table exists)
        if LAKEBASE_ENABLED and metadata_table_exists:
            try:
                # Use the first user message as the conversation title
                first_query = prompt if is_first_message else ""
                update_session_metadata(
                    thread_id=st.session_state.thread_id,
                    first_query=first_query,
                    user_id=user_info.get('user_id', 'unknown'),
                    messages=st.session_state.messages  # Store messages for UI display
                )
                logger.info(f"Updated metadata for thread {st.session_state.thread_id}")
            except Exception as e:
                logger.error(f"Failed to update session metadata: {e}")
