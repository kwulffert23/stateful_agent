from mlflow.deployments import get_deploy_client
from databricks.sdk import WorkspaceClient

def _get_endpoint_task_type(endpoint_name: str) -> str:
    """Get the task type of a serving endpoint."""
    try:
        w = WorkspaceClient()
        ep = w.serving_endpoints.get(endpoint_name)
        return ep.task if ep.task else "unknown"
    except Exception as e:
        print(f"Warning: Could not determine endpoint task type: {e}")
        return "unknown"

def is_endpoint_supported(endpoint_name: str) -> bool:
    """Check if the endpoint has a supported task type."""
    task_type = _get_endpoint_task_type(endpoint_name)
    
    # If endpoint name starts with "agents_", it's an agent endpoint - always supported
    if endpoint_name.startswith("agents_"):
        return True
    
    # Support various agent and chat endpoint types
    supported_task_types = [
        "agent/v1/chat",
        "agent/v2/chat", 
        "llm/v1/chat",
        "chat",
        "conversational"
    ]
    
    # Also allow if task_type is unknown (we'll try to query it anyway)
    if task_type == "unknown":
        return True
    
    # Check if task type contains "agent" or "chat"
    if task_type and ("agent" in task_type.lower() or "chat" in task_type.lower()):
        return True
        
    return task_type in supported_task_types

def _validate_endpoint_task_type(endpoint_name: str) -> None:
    """Validate that the endpoint has a supported task type."""
    if not is_endpoint_supported(endpoint_name):
        task_type = _get_endpoint_task_type(endpoint_name)
        raise Exception(
            f"Detected unsupported endpoint type '{task_type}' for endpoint '{endpoint_name}'. "
            f"This chatbot template supports: agent/v1/chat, agent/v2/chat, llm/v1/chat, chat, conversational. "
            f"For a richer chatbot template with support for all conversational endpoints on Databricks, "
            f"see https://docs.databricks.com/aws/en/generative-ai/agent-framework/chat-app"
        )

def _query_endpoint(endpoint_name: str, messages: list[dict[str, str]], max_tokens, thread_id: str = None, user_id: str = None) -> list[dict[str, str]]:
    """Calls a model serving endpoint with optional thread_id and user_id for stateful agents."""
    _validate_endpoint_task_type(endpoint_name)
    
    # Determine if this is an agent endpoint or chat completions endpoint
    # Agent endpoints (deployed via agents framework) use the Responses Agent schema
    is_agent_endpoint = endpoint_name.startswith("agents_")
    
    if is_agent_endpoint:
        # Use Responses Agent schema
        inputs = {
            'input': messages,  # Agents use 'input' not 'messages'
            'max_output_tokens': max_tokens  # Agents use 'max_output_tokens' not 'max_tokens'
        }
        
        # Add thread_id and user_id to custom_inputs if provided (for stateful agents)
        custom_inputs = {}
        if thread_id:
            custom_inputs['thread_id'] = thread_id
        if user_id:
            custom_inputs['user_id'] = user_id
        if custom_inputs:
            inputs['custom_inputs'] = custom_inputs
    else:
        # Use chat completions schema
        inputs = {
            'messages': messages,
            'max_tokens': max_tokens
        }
        
        # For non-agent endpoints, thread_id and user_id might not be supported
        custom_inputs = {}
        if thread_id:
            custom_inputs['thread_id'] = thread_id
        if user_id:
            custom_inputs['user_id'] = user_id
        if custom_inputs:
            inputs['custom_inputs'] = custom_inputs
    
    res = get_deploy_client('databricks').predict(
        endpoint=endpoint_name,
        inputs=inputs,
    )
    
    # Handle Responses Agent format (agent endpoints)
    if "output" in res:
        # Agent endpoints return output as a list of items
        output_items = res["output"]
        messages = []
        
        for item in output_items:
            # Handle different response formats
            if item.get("role") == "assistant":
                content = item.get("content", "")
                
                # If content is a list of structured objects (like [{'text': '...', 'type': 'output_text'}])
                if isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict):
                            # Extract text from structured content
                            if "text" in part:
                                text_parts.append(part["text"])
                            elif "content" in part:
                                text_parts.append(part["content"])
                    content = "".join(text_parts)
                
                messages.append({
                    "role": "assistant",
                    "content": content
                })
        
        return messages if messages else [{"role": "assistant", "content": ""}]
    
    # Handle chat completions format
    elif "messages" in res:
        return res["messages"]
    
    # Handle OpenAI-style choices format
    elif "choices" in res:
        choice_message = res["choices"][0]["message"]
        choice_content = choice_message.get("content")
        
        # Case 1: The content is a list of structured objects
        if isinstance(choice_content, list):
            combined_content = "".join([part.get("text", "") for part in choice_content if part.get("type") == "text"])
            reformatted_message = {
                "role": choice_message.get("role"),
                "content": combined_content
            }
            return [reformatted_message]
        
        # Case 2: The content is a simple string
        elif isinstance(choice_content, str):
            return [choice_message]
    
    raise Exception(
        "Unexpected response format from endpoint. "
        "This app supports:"
        "\n1) Databricks agent endpoints (Responses Agent format)"
        "\n2) Chat completions endpoints (messages format)"
        "\n3) OpenAI-style endpoints (choices format)"
        f"\nReceived response keys: {list(res.keys())}"
    )

def query_endpoint(endpoint_name, messages, max_tokens, thread_id: str = None, user_id: str = None):
    """
    Query a chat-completions or agent serving endpoint with optional thread_id and user_id for stateful agents.
    If querying an agent serving endpoint that returns multiple messages, this method
    returns the last message.
    
    Args:
        endpoint_name: Name of the serving endpoint
        messages: List of message dictionaries
        max_tokens: Maximum tokens for response
        thread_id: Optional thread ID for stateful agents
        user_id: Optional user ID for privacy and tracking
    """
    return _query_endpoint(endpoint_name, messages, max_tokens, thread_id, user_id)[-1]
