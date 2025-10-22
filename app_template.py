"""
Stateful Chatbot App - TEMPLATE VERSION
========================================

This is a template with TODO markers for customization.
Replace all TODO items with your specific configuration.

Features:
- Session-based conversation history
- Stateful agent with thread_id support
- User privacy with user_id tracking
- Conversational interface with sidebar history

NOTE: This version uses SESSION-BASED history (lost on browser close).
For persistent history across sessions, see app_with_persistent_history.py
"""

import logging
import os
import uuid
import streamlit as st
from model_serving_utils import query_endpoint, is_endpoint_supported

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TODO: Set your serving endpoint name
# Option 1: Set via environment variable (recommended for Databricks Apps)
SERVING_ENDPOINT = os.getenv('SERVING_ENDPOINT')

# Option 2: Hardcode for local testing (uncomment and replace)
# SERVING_ENDPOINT = "agents_youruser-yourmodel-name"

assert SERVING_ENDPOINT, \
    ("Unable to determine serving endpoint to use for chatbot app. If developing locally, "
     "set the SERVING_ENDPOINT environment variable to the name of your serving endpoint. If "
     "deploying to a Databricks app, include a serving endpoint resource named "
     "'serving_endpoint' with CAN_QUERY permissions, as described in "
     "https://docs.databricks.com/aws/en/generative-ai/agent-framework/chat-app#deploy-the-databricks-app")

# Check if the endpoint is supported
endpoint_supported = is_endpoint_supported(SERVING_ENDPOINT)

def get_user_info():
    """
    Extract user information from Databricks authentication headers.
    
    When deployed as a Databricks App, these headers are automatically provided.
    For local testing, these will be None.
    """
    headers = st.context.headers
    return dict(
        user_name=headers.get("X-Forwarded-Preferred-Username"),
        user_email=headers.get("X-Forwarded-Email"),
        user_id=headers.get("X-Forwarded-User"),
    )

user_info = get_user_info()

def _save_to_history():
    """
    Save current conversation to session history.
    
    NOTE: This is session-based storage (lost on browser close).
    For persistent storage, see app_with_persistent_history.py
    """
    if not st.session_state.thread_id or not st.session_state.messages:
        return
    
    # Get the first user message as the title
    first_user_msg = next(
        (msg["content"] for msg in st.session_state.messages if msg["role"] == "user"),
        "New conversation"
    )
    
    # TODO: Customize title length (currently 50 chars)
    MAX_TITLE_LENGTH = 50
    title = first_user_msg[:MAX_TITLE_LENGTH] + "..." if len(first_user_msg) > MAX_TITLE_LENGTH else first_user_msg
    
    # Check if this conversation is already in history
    existing_index = next(
        (i for i, conv in enumerate(st.session_state.conversation_history) 
         if conv["thread_id"] == st.session_state.thread_id),
        None
    )
    
    conv_data = {
        "thread_id": st.session_state.thread_id,
        "title": title,
        "messages": st.session_state.messages.copy(),  # Store the full conversation
    }
    
    if existing_index is not None:
        # Update existing conversation
        st.session_state.conversation_history[existing_index] = conv_data
    else:
        # Add new conversation to the beginning of the list
        st.session_state.conversation_history.insert(0, conv_data)
    
    # TODO: Customize max conversations to keep in session (currently 20)
    MAX_HISTORY_SIZE = 20
    st.session_state.conversation_history = st.session_state.conversation_history[:MAX_HISTORY_SIZE]

# Initialize session state
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

# Initialize conversation history (for current session)
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Sidebar for thread management
with st.sidebar:
    # TODO: Customize sidebar header
    st.header("Chats")
    
    # New conversation button
    if st.button("+ New Conversation", use_container_width=True):
        # Save current conversation to history before starting new one
        if st.session_state.thread_id and st.session_state.messages:
            _save_to_history()
        
        st.session_state.thread_id = None
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    # Show recent conversations (from current session)
    if st.session_state.conversation_history:
        st.caption(f"{len(st.session_state.conversation_history)} conversations in this session")
        
        for conv in st.session_state.conversation_history:
            thread_id = conv["thread_id"]
            title = conv["title"]
            
            # Check if this is the current conversation
            is_current = thread_id == st.session_state.thread_id
            
            # Show conversation button
            if st.button(
                title,
                key=f"conv_{thread_id}",
                use_container_width=True,
                disabled=is_current,
                type="primary" if is_current else "secondary"
            ):
                # Save current conversation before switching
                if st.session_state.thread_id and st.session_state.messages:
                    _save_to_history()
                
                # Load selected conversation (with messages)
                st.session_state.thread_id = thread_id
                st.session_state.messages = conv.get("messages", []).copy()
                st.rerun()
    else:
        # TODO: Customize empty state message
        st.info("No conversations yet")

# Main app
# TODO: Customize app title
st.title("Stateful Chatbot")

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
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    # TODO: Customize input placeholder
    if prompt := st.chat_input("What is up?"):
        # Generate thread_id if this is a new conversation
        if not st.session_state.thread_id:
            st.session_state.thread_id = str(uuid.uuid4())
            logger.info(f"Created new thread_id: {st.session_state.thread_id}")
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            try:
                # TODO: Customize max_tokens (currently 400)
                MAX_TOKENS = 400
                
                # Query the Databricks serving endpoint with thread_id and user_id
                response = query_endpoint(
                    endpoint_name=SERVING_ENDPOINT,
                    messages=st.session_state.messages,
                    max_tokens=MAX_TOKENS,
                    thread_id=st.session_state.thread_id,  # Pass thread_id for statefulness
                    user_id=user_info.get("user_id")  # Pass user_id for privacy
                )
                assistant_response = response.get("content", "")
                
                # Ensure we have a valid response
                if not assistant_response or not assistant_response.strip():
                    # TODO: Customize fallback message
                    assistant_response = "I apologize, but I couldn't generate a response. Please try again."
                
                st.markdown(assistant_response)
                
                # Add assistant response to chat history (only on success)
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                
                # Save conversation to history
                _save_to_history()
                
            except Exception as e:
                logger.error(f"Error querying endpoint: {e}")
                st.error(f"Failed to get response: {e}")
                # Remove the user message from history since we failed to get a response
                st.session_state.messages.pop()
                # TODO: Customize error message
                st.warning("Your message was not sent. Please try again or start a new conversation.")

