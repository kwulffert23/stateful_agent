"""
Lakebase History Utilities
===========================

Provides persistent conversation history across browser sessions using Databricks Lakebase.

This module:
- Creates and manages a metadata table in Lakebase
- Stores conversation summaries (thread_id, user_id, first_query, last_updated)
- Queries conversation history filtered by user
- Handles authentication via Databricks WorkspaceClient

Architecture:
- Metadata table: Lightweight summaries for sidebar display
- Checkpoints table: Full conversation data (managed by agent)
"""

import logging
import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

import streamlit as st
import psycopg
from psycopg_pool import ConnectionPool
from databricks.sdk import WorkspaceClient

logger = logging.getLogger(__name__)

# Initialize WorkspaceClient
w = None
try:
    w = WorkspaceClient()
    logger.info("WorkspaceClient initialized successfully")
except Exception as e:
    logger.warning(f"WorkspaceClient initialization failed: {e}. Lakebase features will be disabled.")

class RotatingTokenConnection(psycopg.Connection):
    """
    Custom psycopg Connection that injects fresh OAuth tokens for Lakebase authentication.
    
    Based on Databricks Apps Cookbook pattern for OLTP database connections.
    Tokens are generated on-demand for each connection to ensure they're always fresh.
    """

    @classmethod
    def fun_generate_credential(cls, instance_name: str) -> tuple:
        """
        Generate fresh OAuth token for Lakebase using WorkspaceClient.
        
        Uses the workspace OAuth token directly (same approach as working Lakebase apps).
        Username is read from environment variables auto-injected by Databricks Apps.
        
        Returns:
            tuple: (username, password/token)
        """
        if w is None:
            raise ValueError("WorkspaceClient not initialized. Cannot generate token.")
        
        try:
            # Get OAuth token - this is the working pattern from Databricks Apps examples
            token = w.config.oauth_token().access_token
            logger.debug("Generated fresh OAuth token for Lakebase")
            
            # Username is provided by Databricks Apps via PGUSER environment variable
            username = os.getenv('PGUSER') or os.getenv('LAKEBASE_USER')
            
            if not username:
                logger.error("No username found in PGUSER or LAKEBASE_USER environment variable")
                logger.error("Please ensure Lakebase resource is properly declared in app.yaml")
                raise ValueError(
                    "Unable to determine database username. PGUSER environment variable not set. "
                    "Ensure Lakebase resource is declared in app.yaml"
                )
            
            logger.info(f"Using username from environment: {username}")
            return username, token
            
        except AttributeError as e:
            logger.error(f"Failed to get OAuth token: {e}")
            raise ValueError(f"Could not obtain OAuth token: {e}")
        except Exception as e:
            logger.error(f"Failed to generate credentials: {e}")
            raise ValueError(f"Could not obtain Lakebase credentials: {e}")

    @classmethod
    def connect(cls, conninfo: str = "", **kwargs):
        """
        Override connect to inject username and password/token.
        
        The _instance_name is passed via kwargs and used to get credentials.
        """
        instance_name = kwargs.pop("_instance_name", None)
        if instance_name:
            username, password = cls.fun_generate_credential(instance_name)
            kwargs["user"] = username
            kwargs["password"] = password
        kwargs.setdefault("sslmode", "require")
        return super().connect(conninfo, **kwargs)

@st.cache_resource
def get_connection_pool(instance_name: str, host: str, user: str, database: str) -> Optional[ConnectionPool]:
    """
    Build and cache a connection pool for Lakebase.
    
    Args:
        instance_name: Name of the Lakebase instance
        host: Lakebase host URL
        user: Database user (may be overridden by credential)
        database: Database name (default: databricks_postgres)
    
    Returns:
        ConnectionPool or None if setup fails
    
    The pool is cached using @st.cache_resource to avoid recreating connections
    on every Streamlit rerun.
    """
    if w is None:
        logger.error("Cannot build connection pool: WorkspaceClient not initialized.")
        return None
    
    try:
        # Don't include user in conninfo - let RotatingTokenConnection inject it
        conninfo = f"host={host} dbname={database}"
        pool = ConnectionPool(
            conninfo=conninfo,
            connection_class=RotatingTokenConnection,
            kwargs={"_instance_name": instance_name},
            min_size=1,
            max_size=5,
            open=True,
        )
        
        # Test connection
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        
        logger.info("Successfully created Lakebase connection pool.")
        return pool
    
    except Exception as e:
        logger.error(f"Failed to create connection pool: {e}")
        return None

def _get_lakebase_config():
    """
    Retrieve Lakebase configuration from environment variables.
    
    When a Lakebase resource is declared in app.yaml, Databricks Apps automatically
    injects PG* environment variables (PGHOST, PGUSER, PGDATABASE, PGPORT, etc.).
    
    Returns:
        tuple: (instance_name, host, user, database, port)
    """
    instance_name = os.getenv('LAKEBASE_INSTANCE_NAME')
    
    # Try to get from PG* environment variables (auto-injected by Databricks Apps)
    host = os.getenv('PGHOST') or os.getenv('LAKEBASE_HOST')
    database = os.getenv('PGDATABASE') or os.getenv('LAKEBASE_DATABASE', 'databricks_postgres')
    user = os.getenv('PGUSER') or os.getenv('LAKEBASE_USER')
    port = os.getenv('PGPORT') or os.getenv('LAKEBASE_PORT', '5432')

    if not instance_name:
        logger.error("Missing LAKEBASE_INSTANCE_NAME environment variable.")
        return None, None, None, None, None
    
    if not host:
        logger.error("Missing PGHOST or LAKEBASE_HOST environment variable.")
        logger.error("Please ensure Lakebase resource is declared in app.yaml")
        return None, None, None, None, None
    
    if not user:
        logger.error("Missing PGUSER or LAKEBASE_USER environment variable.")
        logger.error("Please ensure Lakebase resource is declared in app.yaml")
        return None, None, None, None, None
    
    if w is None:
        logger.warning("WorkspaceClient not initialized. Token-based auth may not work.")
    
    logger.info(f"Lakebase config: instance={instance_name}, host={host}, user={user}, db={database}, port={port}")
    return instance_name, host, user, database, port

def create_metadata_table() -> bool:
    """
    Create the session_metadata table if it doesn't exist, and ensure it has all required columns.
    
    Table schema:
        - thread_id (VARCHAR, PRIMARY KEY): Unique conversation identifier
        - user_id (VARCHAR): User who owns this conversation
        - first_query (TEXT): First user message (used as title)
        - messages (TEXT): JSON array of conversation messages for UI display
        - last_updated (TIMESTAMP): When conversation was last active
        - created_at (TIMESTAMP): When conversation was created
    
    Returns:
        bool: True if successful, False otherwise
    """
    instance_name, host, user, database, _ = _get_lakebase_config()
    if not all([instance_name, host, user, database]):
        logger.error("Incomplete Lakebase configuration. Cannot create metadata table.")
        return False

    pool = get_connection_pool(instance_name, host, user, database)
    if pool is None:
        logger.error("Cannot create metadata table - no connection pool")
        return False

    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                create_table_sql = """
                CREATE TABLE IF NOT EXISTS session_metadata (
                    thread_id VARCHAR(255) PRIMARY KEY,
                    user_id VARCHAR(255) NOT NULL,
                    first_query TEXT,
                    messages TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
                cur.execute(create_table_sql)
                
                # Add messages column if it doesn't exist (for existing tables)
                alter_table_sql = """
                DO $$ 
                BEGIN 
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns 
                        WHERE table_name='session_metadata' AND column_name='messages'
                    ) THEN
                        ALTER TABLE session_metadata ADD COLUMN messages TEXT;
                    END IF;
                END $$;
                """
                cur.execute(alter_table_sql)
                
                # Create index separately (PostgreSQL syntax)
                create_index_sql = """
                CREATE INDEX IF NOT EXISTS idx_user_updated 
                ON session_metadata (user_id, last_updated DESC)
                """
                cur.execute(create_index_sql)
            conn.commit()
        logger.info("session_metadata table created/updated successfully.")
        return True
    except Exception as e:
        logger.error(f"Error creating session_metadata table: {e}")
        return False

def get_thread_list(user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Fetch a list of recent conversation threads from the metadata table.
    
    Args:
        user_id: Filter conversations by this user
        limit: Maximum number of conversations to return (default: 20)
    
    Returns:
        List of dicts with keys: thread_id, first_query, last_updated
        Empty list if query fails or no data found
    
    Results are ordered by last_updated DESC (most recent first).
    """
    instance_name, host, user, database, _ = _get_lakebase_config()
    if not all([instance_name, host, user, database]):
        logger.warning("Incomplete Lakebase configuration. Returning empty history.")
        return []

    pool = get_connection_pool(instance_name, host, user, database)
    if pool is None:
        logger.warning("Cannot get thread list - no connection pool")
        return []

    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                query = """
                SELECT thread_id, first_query, last_updated
                FROM session_metadata
                WHERE user_id = %s
                ORDER BY last_updated DESC
                LIMIT %s
                """
                cur.execute(query, (user_id, limit))
                rows = cur.fetchall()
                
                threads = []
                for row in rows:
                    threads.append({
                        'thread_id': row[0],
                        'first_query': row[1],
                        'last_updated': row[2].isoformat() if row[2] else None
                    })
                
                logger.info(f"Retrieved {len(threads)} threads for user {user_id}")
                return threads
    
    except Exception as e:
        logger.error(f"Error fetching thread list from metadata table: {e}")
        return []

def update_session_metadata(thread_id: str, first_query: str, user_id: str, messages: list = None):
    """
    Upsert (insert or update) session metadata.
    
    Args:
        thread_id: Conversation identifier
        first_query: First user message (for display in sidebar)
        user_id: User who owns this conversation
        messages: Full conversation messages for UI display
    
    Behavior:
        - If thread_id exists: Update last_updated timestamp and messages
        - If thread_id is new: Insert new row with all fields
        - Uses COALESCE to preserve existing first_query on updates
    
    Note: Messages are stored for UI display. The agent also has full history
    in its checkpoint table.
    
    This is called after each message exchange to keep the metadata fresh.
    """
    instance_name, host, user, database, _ = _get_lakebase_config()
    if not all([instance_name, host, user, database]):
        logger.warning("Cannot update session metadata - missing Lakebase config")
        return

    pool = get_connection_pool(instance_name, host, user, database)
    if pool is None:
        logger.warning("Cannot update session metadata - no connection pool")
        return

    try:
        import json
        messages_json = json.dumps(messages) if messages else None
        
        with pool.connection() as conn:
            with conn.cursor() as cur:
                upsert_sql = """
                INSERT INTO session_metadata (thread_id, user_id, first_query, messages, last_updated, created_at)
                VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT (thread_id) DO UPDATE
                SET last_updated = CURRENT_TIMESTAMP,
                    first_query = COALESCE(session_metadata.first_query, EXCLUDED.first_query),
                    messages = EXCLUDED.messages
                """
                cur.execute(upsert_sql, (thread_id, user_id, first_query, messages_json))
            conn.commit()
        logger.debug(f"Upserted metadata for thread_id: {thread_id}")
    
    except Exception as e:
        logger.error(f"Error updating session metadata for thread {thread_id}: {e}")

def get_thread_messages(thread_id: str) -> list:
    """
    Retrieve messages for a specific thread from metadata table.
    
    Args:
        thread_id: The conversation thread ID
    
    Returns:
        List of message dicts with 'role' and 'content' keys
    """
    instance_name, host, user, database, _ = _get_lakebase_config()
    if not all([instance_name, host, user, database]):
        return []

    pool = get_connection_pool(instance_name, host, user, database)
    if pool is None:
        return []

    try:
        import json
        with pool.connection() as conn:
            with conn.cursor() as cur:
                query = "SELECT messages FROM session_metadata WHERE thread_id = %s"
                cur.execute(query, (thread_id,))
                row = cur.fetchone()
                
                if row and row[0]:
                    return json.loads(row[0])
                return []
    except Exception as e:
        logger.error(f"Error fetching messages for thread {thread_id}: {e}")
        return []

def check_metadata_table_exists() -> bool:
    """
    Check if the session_metadata table exists in Lakebase.
    
    Returns:
        bool: True if table exists, False otherwise
    
    Useful for showing a "Create Table" button in the UI if the table hasn't been set up yet.
    """
    instance_name, host, user, database, _ = _get_lakebase_config()
    if not all([instance_name, host, user, database]):
        return False

    pool = get_connection_pool(instance_name, host, user, database)
    if pool is None:
        return False

    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'session_metadata'
                    )
                """)
                result = cur.fetchone()
                return result[0] if result else False
    
    except Exception as e:
        logger.error(f"Error checking if metadata table exists: {e}")
        return False

