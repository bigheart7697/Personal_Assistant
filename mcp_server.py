#!/usr/bin/env python3
"""
MCP Server for Strategic Note-Taking Assistant
Implements the Model Context Protocol using FastMCP SDK

Production-grade implementation with:
- FastMCP SDK for protocol handling (eliminates boilerplate)
- Pydantic validation for robust input validation
- Explicit Roots security boundaries
- OpenTelemetry tracing for observability
- Structured logging
- Metrics collection
- Audit logging
- Enhanced security validation
- Configuration management
- Resources primitive for note reading
- Prompts primitive for common workflows
- Elicitation for user confirmations
- Rate limiting per agent/session
- OAuth 2.0 authorization support
"""

import json
import os
import logging
import time
import re
import pickle
import hashlib
import secrets
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from dataclasses import dataclass

# FastMCP and Pydantic imports
from fastmcp import FastMCP
from pydantic import BaseModel, Field, field_validator

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Status, StatusCode

# Day 2: Semantic Memory & Vector Embeddings
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import faiss
    SEMANTIC_SEARCH_AVAILABLE = True
except ImportError:
    SEMANTIC_SEARCH_AVAILABLE = False
    logger = logging.getLogger("mcp_server")
    logger.warning("Semantic search dependencies not available. Install sentence-transformers, numpy, and faiss-cpu for Day 2 features.")


class ErrorCategory(Enum):
    """Categorization of errors for metrics and analysis"""
    VALIDATION_ERROR = "validation_error"
    IO_ERROR = "io_error"
    PROTOCOL_ERROR = "protocol_error"
    SECURITY_ERROR = "security_error"
    INTERNAL_ERROR = "internal_error"


class ServerConfig:
    """Configuration management for different environments"""
    
    def __init__(self):
        self.environment = os.environ.get("MCP_ENV", "development")
        self.notes_path = os.environ.get("NOTES_PATH", "./notes")
        self.memory_path = os.environ.get("MEMORY_PATH", "./memory")
        self.log_level = os.environ.get("LOG_LEVEL", "INFO")
        self.enable_tracing = os.environ.get("ENABLE_TRACING", "true").lower() == "true"
        self.enable_audit_log = os.environ.get("ENABLE_AUDIT_LOG", "true").lower() == "true"
        self.audit_log_path = os.environ.get("AUDIT_LOG_PATH", "./audit.log")
        self.max_file_size_mb = int(os.environ.get("MAX_FILE_SIZE_MB", "10"))
        self.embedding_model = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        
        # Rate limiting configuration
        self.enable_rate_limiting = os.environ.get("ENABLE_RATE_LIMITING", "true").lower() == "true"
        self.rate_limit_requests = int(os.environ.get("RATE_LIMIT_REQUESTS", "100"))
        self.rate_limit_window_seconds = int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "60"))
        
        # OAuth 2.0 configuration
        self.enable_oauth = os.environ.get("ENABLE_OAUTH", "false").lower() == "true"
        self.oauth_client_id = os.environ.get("OAUTH_CLIENT_ID", "")
        self.oauth_client_secret = os.environ.get("OAUTH_CLIENT_SECRET", "")
        self.oauth_token_url = os.environ.get("OAUTH_TOKEN_URL", "")
        self.oauth_scopes = os.environ.get("OAUTH_SCOPES", "notes:read notes:write").split()
        
        # Elicitation (user confirmation) configuration
        self.enable_elicitation = os.environ.get("ENABLE_ELICITATION", "true").lower() == "true"
        self.require_confirmation_for_overwrite = os.environ.get("REQUIRE_CONFIRMATION_OVERWRITE", "true").lower() == "true"
        
    def is_production(self) -> bool:
        return self.environment == "production"
    
    def is_development(self) -> bool:
        return self.environment == "development"


@dataclass
class OAuthToken:
    """OAuth 2.0 token information"""
    access_token: str
    token_type: str
    expires_at: datetime
    scopes: List[str]
    agent_id: str
    
    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) >= self.expires_at
    
    def has_scope(self, required_scope: str) -> bool:
        return required_scope in self.scopes


class OAuthManager:
    """
    OAuth 2.0 Authorization Manager
    Implements authorization code grant flow as recommended in MCP whitepaper (page 39)
    """
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.tokens: Dict[str, OAuthToken] = {}  # agent_id -> token
        self.authorization_codes: Dict[str, Dict[str, Any]] = {}  # code -> {agent_id, scopes, expires}
    
    def is_enabled(self) -> bool:
        return self.config.enable_oauth
    
    def generate_authorization_code(self, agent_id: str, scopes: List[str]) -> str:
        """Generate a new authorization code"""
        code = secrets.token_urlsafe(32)
        self.authorization_codes[code] = {
            "agent_id": agent_id,
            "scopes": scopes,
            "expires_at": datetime.now(timezone.utc) + timedelta(minutes=10)
        }
        return code
    
    def exchange_code_for_token(self, code: str) -> Optional[OAuthToken]:
        """Exchange authorization code for access token"""
        if code not in self.authorization_codes:
            return None
        
        code_data = self.authorization_codes[code]
        if datetime.now(timezone.utc) >= code_data["expires_at"]:
            del self.authorization_codes[code]
            return None
        
        # Generate access token
        token = OAuthToken(
            access_token=secrets.token_urlsafe(32),
            token_type="Bearer",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            scopes=code_data["scopes"],
            agent_id=code_data["agent_id"]
        )
        
        self.tokens[token.access_token] = token
        del self.authorization_codes[code]
        return token
    
    def validate_token(self, access_token: str) -> Optional[OAuthToken]:
        """Validate and return token information"""
        if not self.is_enabled():
            # If OAuth is disabled, return a permissive token
            return OAuthToken(
                access_token="dev-token",
                token_type="Bearer",
                expires_at=datetime.now(timezone.utc) + timedelta(days=365),
                scopes=["notes:read", "notes:write", "memory:read", "memory:write"],
                agent_id="development"
            )
        
        token = self.tokens.get(access_token)
        if not token or token.is_expired():
            return None
        return token
    
    def check_scope(self, access_token: str, required_scope: str) -> bool:
        """Check if token has required scope"""
        token = self.validate_token(access_token)
        if not token:
            return False
        return token.has_scope(required_scope)


class RateLimiter:
    """
    Rate limiting per agent/session
    Addresses whitepaper concern about DoS attacks and resource exhaustion
    """
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.requests: Dict[str, List[datetime]] = defaultdict(list)
    
    def is_enabled(self) -> bool:
        return self.config.enable_rate_limiting
    
    def check_rate_limit(self, agent_id: str) -> Tuple[bool, Optional[str]]:
        """
        Check if agent is within rate limits
        Returns: (is_allowed, error_message)
        """
        if not self.is_enabled():
            return True, None
        
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(seconds=self.config.rate_limit_window_seconds)
        
        # Clean up old requests
        self.requests[agent_id] = [
            req_time for req_time in self.requests[agent_id]
            if req_time > window_start
        ]
        
        # Check limit
        if len(self.requests[agent_id]) >= self.config.rate_limit_requests:
            return False, (
                f"Rate limit exceeded. Maximum {self.config.rate_limit_requests} requests "
                f"per {self.config.rate_limit_window_seconds} seconds allowed."
            )
        
        # Record this request
        self.requests[agent_id].append(now)
        return True, None


class ElicitationManager:
    """
    Manages user confirmations for sensitive operations
    Implements MCP Elicitation primitive (whitepaper page 33)
    """
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.pending_confirmations: Dict[str, Dict[str, Any]] = {}
    
    def is_enabled(self) -> bool:
        return self.config.enable_elicitation
    
    def requires_confirmation(self, operation: str, context: Dict[str, Any]) -> bool:
        """Check if operation requires user confirmation"""
        if not self.is_enabled():
            return False
        
        if operation == "file_overwrite" and self.config.require_confirmation_for_overwrite:
            return True
        
        return False
    
    def create_confirmation_request(self, operation: str, context: Dict[str, Any]) -> str:
        """Create a confirmation request and return its ID"""
        confirmation_id = secrets.token_urlsafe(16)
        self.pending_confirmations[confirmation_id] = {
            "operation": operation,
            "context": context,
            "created_at": datetime.now(timezone.utc),
            "expires_at": datetime.now(timezone.utc) + timedelta(minutes=5)
        }
        return confirmation_id
    
    def check_confirmation(self, confirmation_id: str) -> Optional[Dict[str, Any]]:
        """Check if confirmation exists and is valid"""
        if confirmation_id not in self.pending_confirmations:
            return None
        
        confirmation = self.pending_confirmations[confirmation_id]
        if datetime.now(timezone.utc) >= confirmation["expires_at"]:
            del self.pending_confirmations[confirmation_id]
            return None
        
        return confirmation
    
    def consume_confirmation(self, confirmation_id: str) -> bool:
        """Consume a confirmation (one-time use)"""
        if confirmation_id in self.pending_confirmations:
            del self.pending_confirmations[confirmation_id]
            return True
        return False


class MemoryBank:
    """
    Dual-layer memory system for context engineering.
    
    Short-term (Session): Current context like job postings being reviewed
    Long-term (Entity): Career history, skills, preferences, tone of voice
    """
    
    def __init__(self, memory_path: Path):
        self.memory_path = memory_path
        self.memory_path.mkdir(parents=True, exist_ok=True)
        
        self.short_term_file = self.memory_path / "short_term.json"
        self.long_term_file = self.memory_path / "long_term.json"
        
        self.short_term_memory = self._load_memory(self.short_term_file)
        self.long_term_memory = self._load_memory(self.long_term_file)
    
    def _load_memory(self, file_path: Path) -> Dict[str, Any]:
        """Load memory from file or create empty dict"""
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save_memory(self, memory: Dict[str, Any], file_path: Path):
        """Save memory to file"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(memory, f, indent=2, ensure_ascii=False)
    
    def update_short_term(self, key: str, value: Any):
        """Update short-term (session) memory"""
        self.short_term_memory[key] = {
            "value": value,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        self._save_memory(self.short_term_memory, self.short_term_file)
    
    def update_long_term(self, key: str, value: Any):
        """Update long-term (entity) memory"""
        self.long_term_memory[key] = {
            "value": value,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        self._save_memory(self.long_term_memory, self.long_term_file)
    
    def get_short_term(self, key: Optional[str] = None) -> Dict[str, Any]:
        """Get short-term memory (all or specific key)"""
        if key:
            return self.short_term_memory.get(key, {})
        return self.short_term_memory
    
    def get_long_term(self, key: Optional[str] = None) -> Dict[str, Any]:
        """Get long-term memory (all or specific key)"""
        if key:
            return self.long_term_memory.get(key, {})
        return self.long_term_memory
    
    def get_all_memory(self) -> Dict[str, Any]:
        """Get both short-term and long-term memory"""
        return {
            "short_term": self.short_term_memory,
            "long_term": self.long_term_memory
        }
    
    def clear_short_term(self):
        """Clear short-term memory (session reset)"""
        self.short_term_memory = {}
        self._save_memory(self.short_term_memory, self.short_term_file)


class VectorEmbeddingManager:
    """
    Manages vector embeddings for semantic search.
    Uses sentence-transformers for creating embeddings and FAISS for similarity search.
    """
    
    def __init__(self, memory_path: Path, model_name: str = "all-MiniLM-L6-v2"):
        self.memory_path = memory_path
        self.index_file = memory_path / "vector_index.faiss"
        self.metadata_file = memory_path / "vector_metadata.pkl"
        
        if not SEMANTIC_SEARCH_AVAILABLE:
            self.model = None
            self.index = None
            self.metadata = []
            return
        
        # Load embedding model (lightweight and fast)
        self.model = SentenceTransformer(model_name)
        self.dimension = 384  # all-MiniLM-L6-v2 produces 384-dim embeddings
        
        # Load or create FAISS index
        if self.index_file.exists() and self.metadata_file.exists():
            self.index = faiss.read_index(str(self.index_file))
            with open(self.metadata_file, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)  # L2 distance
            self.metadata = []
    
    def add_note_to_index(self, note_path: Path, content: str):
        """Add a note to the vector index"""
        if not SEMANTIC_SEARCH_AVAILABLE or self.model is None:
            return
        
        # Create embedding
        embedding = self.model.encode([content])[0]
        
        # Add to FAISS index
        self.index.add(np.array([embedding], dtype='float32'))
        
        # Store metadata
        self.metadata.append({
            "path": str(note_path),
            "content": content,
            "indexed_at": datetime.now(timezone.utc).isoformat()
        })
        
        # Save index and metadata
        self._save_index()
    
    def search_similar_notes(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for semantically similar notes"""
        if not SEMANTIC_SEARCH_AVAILABLE or self.model is None:
            return []
        
        if self.index.ntotal == 0:
            return []
        
        # Create query embedding
        query_embedding = self.model.encode([query])[0]
        
        # Search FAISS index
        distances, indices = self.index.search(
            np.array([query_embedding], dtype='float32'),
            min(top_k, self.index.ntotal)
        )
        
        # Build results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['similarity_score'] = float(1 / (1 + distances[0][i]))  # Convert distance to similarity
                results.append(result)
        
        return results
    
    def rebuild_index(self, notes_base_path: Path):
        """Rebuild the entire vector index from scratch"""
        if not SEMANTIC_SEARCH_AVAILABLE or self.model is None:
            return
        
        # Reset index
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
        
        # Index all markdown files
        for note_file in notes_base_path.rglob("*.md"):
            try:
                content = note_file.read_text(encoding='utf-8')
                self.add_note_to_index(note_file, content)
            except Exception as e:
                logger = logging.getLogger("mcp_server")
                logger.warning(f"Failed to index {note_file}: {e}")
        
        self._save_index()
    
    def _save_index(self):
        """Save FAISS index and metadata to disk"""
        if not SEMANTIC_SEARCH_AVAILABLE:
            return
        
        faiss.write_index(self.index, str(self.index_file))
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)


class KnowledgeExtractor:
    """
    Extracts key facts and insights from notes for memory consolidation.
    Uses pattern matching and heuristics to identify important information.
    """
    
    @staticmethod
    def extract_skills(content: str) -> List[str]:
        """Extract technical skills and technologies mentioned"""
        # Common patterns for skills
        skill_patterns = [
            r'\b(Python|Java|JavaScript|TypeScript|C\+\+|C#|Ruby|Go|Rust|PHP)\b',
            r'\b(React|Vue|Angular|Django|Flask|FastAPI|Spring|Node\.js)\b',
            r'\b(SQL|PostgreSQL|MySQL|MongoDB|Redis|DynamoDB)\b',
            r'\b(AWS|Azure|GCP|Docker|Kubernetes|Jenkins|GitHub Actions)\b',
            r'\b(Machine Learning|ML|AI|Data Science|NLP|Computer Vision)\b'
        ]
        
        skills = set()
        for pattern in skill_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            skills.update(matches)
        
        return list(skills)
    
    @staticmethod
    def extract_preferences(content: str) -> List[Dict[str, str]]:
        """Extract user preferences and opinions"""
        preferences = []
        
        # Pattern: "I prefer X over Y"
        prefer_pattern = r'I prefer ([^\.]+?) over ([^\.]+)'
        matches = re.findall(prefer_pattern, content, re.IGNORECASE)
        for match in matches:
            preferences.append({
                "type": "preference",
                "preferred": match[0].strip(),
                "over": match[1].strip()
            })
        
        # Pattern: "I like/love/enjoy X"
        like_pattern = r'I (?:like|love|enjoy) ([^\.]+)'
        matches = re.findall(like_pattern, content, re.IGNORECASE)
        for match in matches:
            preferences.append({
                "type": "positive",
                "subject": match.strip()
            })
        
        # Pattern: "I dislike/hate X"
        dislike_pattern = r'I (?:dislike|hate|avoid) ([^\.]+)'
        matches = re.findall(dislike_pattern, content, re.IGNORECASE)
        for match in matches:
            preferences.append({
                "type": "negative",
                "subject": match.strip()
            })
        
        return preferences
    
    @staticmethod
    def extract_key_facts(content: str) -> Dict[str, Any]:
        """Extract all key facts from content"""
        return {
            "skills": KnowledgeExtractor.extract_skills(content),
            "preferences": KnowledgeExtractor.extract_preferences(content),
            "word_count": len(content.split()),
            "extracted_at": datetime.now(timezone.utc).isoformat()
        }


class Metrics:
    """Simple in-memory metrics collector"""
    
    def __init__(self):
        self.tool_calls = {}  # {tool_name: count}
        self.tool_errors = {}  # {tool_name: count}
        self.tool_latencies = {}  # {tool_name: [latencies]}
        self.error_categories = {}  # {category: count}
        
    def record_tool_call(self, tool_name: str):
        self.tool_calls[tool_name] = self.tool_calls.get(tool_name, 0) + 1
    
    def record_tool_error(self, tool_name: str, category: ErrorCategory):
        self.tool_errors[tool_name] = self.tool_errors.get(tool_name, 0) + 1
        self.error_categories[category.value] = self.error_categories.get(category.value, 0) + 1
    
    def record_tool_latency(self, tool_name: str, latency_ms: float):
        if tool_name not in self.tool_latencies:
            self.tool_latencies[tool_name] = []
        self.tool_latencies[tool_name].append(latency_ms)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current metrics statistics"""
        stats = {
            "tool_calls": self.tool_calls,
            "tool_errors": self.tool_errors,
            "error_categories": self.error_categories,
            "tool_latencies": {}
        }
        
        # Calculate latency statistics
        for tool_name, latencies in self.tool_latencies.items():
            if latencies:
                stats["tool_latencies"][tool_name] = {
                    "count": len(latencies),
                    "avg_ms": sum(latencies) / len(latencies),
                    "min_ms": min(latencies),
                    "max_ms": max(latencies)
                }
        
        return stats


# Pydantic models for input validation
class WriteNoteInput(BaseModel):
    """Validated input schema for write_note tool"""
    folder_name: str = Field(
        ...,
        description="The name of the category folder where the note will be saved",
        min_length=1,
        max_length=255
    )
    file_name: str = Field(
        ...,
        description="The name of the note file (should end with .md extension)",
        min_length=1,
        max_length=255
    )
    content: str = Field(
        ...,
        description="The full text content of the note in markdown format",
        min_length=1
    )
    access_token: Optional[str] = Field(
        default=None,
        description="OAuth 2.0 access token (required if OAuth is enabled)"
    )
    confirmation_id: Optional[str] = Field(
        default=None,
        description="Confirmation ID for file overwrites (if elicitation is enabled)"
    )
    
    @field_validator('folder_name')
    @classmethod
    def validate_folder_name(cls, v: str) -> str:
        """Validate folder name for security"""
        if not v or not v.strip():
            raise ValueError("folder_name cannot be empty")
        
        # Check for path traversal attempts
        if ".." in v or "/" in v or "\\" in v:
            raise ValueError("Invalid folder name: path traversal detected")
        
        # Check for hidden directories
        if v.startswith("."):
            raise ValueError("Invalid folder name: hidden directories not allowed")
        
        # Check for invalid characters
        invalid_chars = ['<', '>', ':', '"', '|', '?', '*']
        if any(char in v for char in invalid_chars):
            raise ValueError(f"Invalid folder name: contains invalid characters")
        
        return v.strip()
    
    @field_validator('file_name')
    @classmethod
    def validate_file_name(cls, v: str) -> str:
        """Validate and normalize file name"""
        if not v or not v.strip():
            raise ValueError("file_name cannot be empty")
        
        v = v.strip()
        
        # Ensure .md extension
        if not v.endswith('.md'):
            v = f"{v}.md"
        
        # Check for path traversal
        if ".." in v or "/" in v or "\\" in v:
            raise ValueError("Invalid file name: path traversal detected")
        
        return v


class FolderInfo(BaseModel):
    """Information about a note folder"""
    name: str
    note_count: int


class SearchNotesInput(BaseModel):
    """Validated input schema for search_notes tool"""
    query: str = Field(
        ...,
        description="The search query (semantic search will find related concepts)",
        min_length=1,
        max_length=500
    )
    max_results: Optional[int] = Field(
        default=5,
        description="Maximum number of results to return",
        ge=1,
        le=20
    )
    folder_name: Optional[str] = Field(
        default=None,
        description="Optional: Search within a specific folder only"
    )


class UpdateMemoryInput(BaseModel):
    """Validated input schema for update_memory tool"""
    memory_type: str = Field(
        ...,
        description="Type of memory to update: 'short_term' or 'long_term'"
    )
    key: str = Field(
        ...,
        description="Memory key (e.g., 'current_job_posting', 'skills', 'preferences')",
        min_length=1,
        max_length=100
    )
    value: str = Field(
        ...,
        description="Memory value to store",
        min_length=1
    )
    
    @field_validator('memory_type')
    @classmethod
    def validate_memory_type(cls, v: str) -> str:
        if v not in ['short_term', 'long_term']:
            raise ValueError("memory_type must be either 'short_term' or 'long_term'")
        return v


class GetMemoryInput(BaseModel):
    """Validated input schema for get_memory tool"""
    memory_type: str = Field(
        ...,
        description="Type of memory to retrieve: 'short_term', 'long_term', or 'all'"
    )
    key: Optional[str] = Field(
        default=None,
        description="Optional: Specific memory key to retrieve"
    )
    
    @field_validator('memory_type')
    @classmethod
    def validate_memory_type(cls, v: str) -> str:
        if v not in ['short_term', 'long_term', 'all']:
            raise ValueError("memory_type must be 'short_term', 'long_term', or 'all'")
        return v


# Initialize configuration and global components
config = ServerConfig()
metrics = Metrics()
oauth_manager = OAuthManager(config)
rate_limiter = RateLimiter(config)
elicitation_manager = ElicitationManager(config)

# Setup structured logging
logger = logging.getLogger("mcp_server")
logger.setLevel(getattr(logging, config.log_level))

# Use stderr for logs (stdout is reserved for JSON-RPC)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(handler)


def setup_tracing() -> trace.Tracer:
    """Setup OpenTelemetry tracing"""
    if not config.enable_tracing:
        return trace.get_tracer(__name__)
    
    resource = Resource.create({
        "service.name": "mcp-note-server",
        "service.version": "3.0.0",
        "deployment.environment": config.environment
    })
    
    provider = TracerProvider(resource=resource)
    
    if config.is_development():
        exporter = ConsoleSpanExporter()
    else:
        exporter = ConsoleSpanExporter()
    
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    
    return trace.get_tracer(__name__)


def setup_audit_logging() -> Optional[logging.Logger]:
    """Setup audit logging for compliance"""
    if not config.enable_audit_log:
        return None
    
    audit_logger = logging.getLogger("mcp_server.audit")
    audit_logger.setLevel(logging.INFO)
    
    try:
        audit_handler = logging.FileHandler(config.audit_log_path)
        audit_formatter = logging.Formatter('%(asctime)s - %(message)s')
        audit_handler.setFormatter(audit_formatter)
        
        if not audit_logger.handlers:
            audit_logger.addHandler(audit_handler)
        
        return audit_logger
    except Exception as e:
        logger.error(f"Failed to setup audit logging: {e}")
        return None


def audit_log(action: str, details: Dict[str, Any], agent_id: Optional[str] = None):
    """Log an auditable action"""
    if not audit_logger:
        return
    
    audit_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "agent_id": agent_id or "unknown",
        "details": details
    }
    
    audit_logger.info(json.dumps(audit_entry))


def check_authorization(access_token: Optional[str], required_scope: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Check OAuth authorization for the request
    Returns: (is_authorized, agent_id, error_message)
    """
    if not oauth_manager.is_enabled():
        # OAuth disabled, allow with development agent
        return True, "development", None
    
    if not access_token:
        return False, None, "Missing access token. OAuth is enabled."
    
    token = oauth_manager.validate_token(access_token)
    if not token:
        return False, None, "Invalid or expired access token."
    
    if not token.has_scope(required_scope):
        return False, token.agent_id, f"Insufficient permissions. Required scope: {required_scope}"
    
    return True, token.agent_id, None


def sanitize_output(data: Any, max_preview_length: int = 200) -> Any:
    """
    Sanitize output to prevent information leaks
    Addresses whitepaper concern about sensitive information exposure (page 45)
    """
    if isinstance(data, str):
        # Truncate long strings to prevent leaking full file contents
        if len(data) > max_preview_length:
            return data[:max_preview_length] + "... [truncated]"
        return data
    elif isinstance(data, dict):
        return {k: sanitize_output(v, max_preview_length) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_output(item, max_preview_length) for item in data]
    return data


def validate_path_within_root(folder_path: Path, root_path: Path) -> bool:
    """
    Validate that the folder path is within the root boundary.
    Implements the "Roots" security pattern.
    """
    try:
        # Resolve both paths to absolute
        resolved_folder = folder_path.resolve()
        resolved_root = root_path.resolve()
        
        # Check if folder is within root
        # Use is_relative_to() for Python 3.9+, or string comparison for older versions
        try:
            return resolved_folder.is_relative_to(resolved_root)
        except AttributeError:
            # Fallback for Python < 3.9
            return str(resolved_folder).startswith(str(resolved_root))
    except Exception:
        return False


# Initialize global components
tracer = setup_tracing()
audit_logger = setup_audit_logging()

# Ensure notes directory exists
notes_base_path = Path(config.notes_path)
notes_base_path.mkdir(parents=True, exist_ok=True)

# Day 2: Initialize memory bank and vector embedding system
memory_path = Path(config.memory_path)
memory_path.mkdir(parents=True, exist_ok=True)

memory_bank = MemoryBank(memory_path)
vector_manager = VectorEmbeddingManager(memory_path, config.embedding_model)
knowledge_extractor = KnowledgeExtractor()

# Initialize FastMCP
mcp = FastMCP(
    "strategic-note-taking-mcp-server",
    version="5.0.0"  # Enhanced with Resources, Prompts, Elicitation, Rate Limiting, OAuth
)

# Roots Security Pattern: Documented root boundary
# The server is confined to the notes_base_path directory through
# runtime validation in validate_path_within_root() function.
# All file operations are validated to ensure they stay within this boundary.
ROOT_BOUNDARY_URI = f"file://{notes_base_path.absolute()}"

logger.info(
    "MCP Server initialized with FastMCP",
    extra={
        "version": "5.0.0",
        "environment": config.environment,
        "notes_path": str(notes_base_path.absolute()),
        "memory_path": str(memory_path.absolute()),
        "root_boundary": ROOT_BOUNDARY_URI,
        "semantic_search_enabled": SEMANTIC_SEARCH_AVAILABLE,
        "oauth_enabled": oauth_manager.is_enabled(),
        "rate_limiting_enabled": rate_limiter.is_enabled(),
        "elicitation_enabled": elicitation_manager.is_enabled()
    }
)


# ============================================================================
# RESOURCES PRIMITIVE - MCP Whitepaper Pages 31
# Expose notes as readable resources with URI scheme: note://{folder}/{file}
# ============================================================================

@mcp.resource("note://{folder}/{filename}")
def read_note_resource(folder: str, filename: str) -> str:
    """
    Read a note as a resource.
    Resources provide direct read access to notes without custom tools.
    URI format: note://folder-name/file-name.md
    """
    try:
        # Validate folder and filename
        if ".." in folder or "/" in folder or "\\" in folder:
            raise ValueError("Invalid folder name")
        if ".." in filename or "/" in filename or "\\" in filename:
            raise ValueError("Invalid filename")
        
        folder_path = notes_base_path / folder
        note_path = folder_path / filename
        
        # Validate within root boundary
        if not validate_path_within_root(note_path, notes_base_path):
            raise ValueError("Path outside root boundary")
        
        if not note_path.exists():
            raise ValueError(f"Note not found: {folder}/{filename}")
        
        content = note_path.read_text(encoding='utf-8')
        
        logger.info(f"Resource accessed: note://{folder}/{filename}")
        
        return content
    
    except Exception as e:
        logger.error(f"Error reading resource: {e}")
        raise


@mcp.resource("note://{folder}/")
def list_notes_in_folder_resource(folder: str) -> str:
    """
    List all notes in a folder as a resource.
    URI format: note://folder-name/
    """
    try:
        # Validate folder
        if ".." in folder or "/" in folder or "\\" in folder:
            raise ValueError("Invalid folder name")
        
        folder_path = notes_base_path / folder
        
        # Validate within root boundary
        if not validate_path_within_root(folder_path, notes_base_path):
            raise ValueError("Path outside root boundary")
        
        if not folder_path.exists():
            raise ValueError(f"Folder not found: {folder}")
        
        # List markdown files
        notes = []
        for note_file in folder_path.glob("*.md"):
            notes.append({
                "name": note_file.name,
                "size": note_file.stat().st_size,
                "modified": datetime.fromtimestamp(note_file.stat().st_mtime, tz=timezone.utc).isoformat(),
                "uri": f"note://{folder}/{note_file.name}"
            })
        
        result = {
            "folder": folder,
            "notes": notes,
            "count": len(notes)
        }
        
        logger.info(f"Resource accessed: note://{folder}/")
        
        return json.dumps(result, indent=2)
    
    except Exception as e:
        logger.error(f"Error listing folder resource: {e}")
        raise


# ============================================================================
# PROMPTS PRIMITIVE - MCP Whitepaper Pages 31
# Predefined prompts for common workflows
# ============================================================================

@mcp.prompt()
def job_application_workflow() -> str:
    """
    Prompt: Job Application Review Workflow
    
    A structured workflow for reviewing and tracking job applications.
    This guides the AI through the complete process of analyzing a job posting,
    comparing it with user skills, and creating appropriate notes.
    """
    return """You are helping the user review and track a job application.

Follow this workflow:

1. **Gather Information**
   - Ask the user to share the job posting details
   - Get the memory context using get_memory to retrieve their skills and preferences

2. **Analyze the Job**
   - Identify key requirements and responsibilities
   - Compare with user's skills from long-term memory
   - Note any skill gaps or growth opportunities
   - Assess cultural and preference fit

3. **Create Documentation**
   - Use write_note to save the analysis to the job-applications folder
   - Include: company name, position, key requirements, fit assessment, action items
   - Update short-term memory with this active application

4. **Follow-up Actions**
   - Suggest next steps (tailor resume, prepare for interviews, etc.)
   - Offer to create additional notes for preparation

Remember: Be strategic, thorough, and supportive throughout the process."""


@mcp.prompt()
def project_idea_brainstorm() -> str:
    """
    Prompt: Project Idea Brainstorming Session
    
    A creative workflow for capturing and organizing project ideas.
    """
    return """You are helping the user brainstorm and document project ideas.

Follow this workflow:

1. **Exploration Phase**
   - Ask about the project domain or area of interest
   - Explore what technologies or skills they want to use/learn
   - Review their existing skills from long-term memory

2. **Ideation**
   - Generate 3-5 concrete project ideas
   - For each idea, outline:
     * Core functionality
     * Technical stack
     * Learning outcomes
     * Estimated complexity

3. **Documentation**
   - Save the most promising ideas to the work folder
   - Structure each note with: Overview, Technical Details, Learning Goals, Next Steps

4. **Prioritization**
   - Help rank ideas by feasibility and learning value
   - Update short-term memory with selected project focus

Be creative, practical, and encouraging!"""


@mcp.prompt()
def daily_reflection() -> str:
    """
    Prompt: Daily Reflection and Note Review
    
    A reflective workflow for reviewing notes and consolidating knowledge.
    """
    return """You are facilitating the user's daily reflection and note review.

Follow this workflow:

1. **Context Retrieval**
   - Get all memory to understand current focus
   - Use search_notes to find recent notes

2. **Review Session**
   - Summarize key themes from recent notes
   - Identify patterns or recurring topics
   - Note any action items or follow-ups

3. **Memory Consolidation**
   - Extract important facts, skills, or insights
   - Update long-term memory with new learnings
   - Clear or update short-term memory as needed

4. **Forward Planning**
   - Suggest areas for tomorrow's focus
   - Identify knowledge gaps to address
   - Offer to create planning notes

Be reflective, insightful, and forward-thinking!"""


# ============================================================================
# TOOL IMPLEMENTATIONS
# ============================================================================

def _list_note_folders_impl() -> str:
    """Implementation of list_note_folders tool"""
    start_time = time.time()
    
    with tracer.start_as_current_span("list_note_folders") as span:
        try:
            metrics.record_tool_call("list_note_folders")
            
            folders = []
            
            # List all directories in notes path
            if notes_base_path.exists():
                for item in notes_base_path.iterdir():
                    if item.is_dir():
                        # Verify within root boundary
                        if not validate_path_within_root(item, notes_base_path):
                            logger.warning(f"Folder {item} outside root boundary, skipping")
                            continue
                        
                        # Count notes in each folder
                        note_count = len(list(item.glob("*.md")))
                        folders.append({
                            "name": item.name,
                            "note_count": note_count
                        })
            
            span.set_attribute("folder_count", len(folders))
            span.set_attribute("notes_path", str(notes_base_path))
            span.set_status(Status(StatusCode.OK))
            
            latency_ms = (time.time() - start_time) * 1000
            metrics.record_tool_latency("list_note_folders", latency_ms)
            
            logger.info(
                "Listed note folders",
                extra={
                    "folder_count": len(folders),
                    "latency_ms": latency_ms
                }
            )
            
            return json.dumps({
                "folders": folders,
                "total_categories": len(folders),
                "notes_path": str(notes_base_path.absolute())
            }, indent=2)
        
        except Exception as e:
            metrics.record_tool_error("list_note_folders", ErrorCategory.IO_ERROR)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            
            logger.error("Error listing folders", exc_info=True)
            raise


def _write_note_impl(input: WriteNoteInput, access_token: Optional[str] = None, 
                     confirmation_id: Optional[str] = None) -> str:
    """
    Saves a text note into a specific folder. Use this only AFTER
    identifying the correct target folder via list_note_folders.
    This tool requires three pieces of information: the folder name
    (which must exist), the file name (should end in .md), and the
    complete note content in markdown format. Includes security
    validation and size limits for production safety.
    
    Enhanced with:
    - OAuth 2.0 authorization (requires notes:write scope)
    - Rate limiting per agent
    - Elicitation for file overwrites
    - Output sanitization
    
    Args:
        input: Validated input containing folder_name, file_name, and content
        access_token: OAuth 2.0 access token (optional if OAuth disabled)
        confirmation_id: Confirmation ID for file overwrites (if elicitation enabled)
    
    Returns:
        JSON string with success confirmation and metadata
    """
    start_time = time.time()
    
    with tracer.start_as_current_span("write_note") as span:
        try:
            metrics.record_tool_call("write_note")
            
            # 1. AUTHORIZATION CHECK (OAuth 2.0)
            is_authorized, agent_id, auth_error = check_authorization(access_token, "notes:write")
            if not is_authorized:
                metrics.record_tool_error("write_note", ErrorCategory.SECURITY_ERROR)
                span.set_status(Status(StatusCode.ERROR, "Authorization failed"))
                raise ValueError(f"Authorization error: {auth_error}")
            
            # 2. RATE LIMITING CHECK
            is_allowed, rate_error = rate_limiter.check_rate_limit(agent_id)
            if not is_allowed:
                metrics.record_tool_error("write_note", ErrorCategory.VALIDATION_ERROR)
                span.set_status(Status(StatusCode.ERROR, "Rate limit exceeded"))
                raise ValueError(rate_error)
            
            # Add span attributes
            span.set_attribute("folder_name", input.folder_name)
            span.set_attribute("file_name", input.file_name)
            span.set_attribute("content_size_bytes", len(input.content.encode('utf-8')))
            span.set_attribute("agent_id", agent_id)
            
            # Validate file size
            content_size_mb = len(input.content.encode('utf-8')) / (1024 * 1024)
            if content_size_mb > config.max_file_size_mb:
                metrics.record_tool_error("write_note", ErrorCategory.VALIDATION_ERROR)
                span.set_status(Status(StatusCode.ERROR, "File too large"))
                raise ValueError(
                    f"File size ({content_size_mb:.2f}MB) exceeds maximum allowed size ({config.max_file_size_mb}MB)"
                )
            
            # Build target folder path
            target_folder = notes_base_path / input.folder_name
            
            # Validate within root boundary (Roots security)
            if not validate_path_within_root(target_folder, notes_base_path):
                metrics.record_tool_error("write_note", ErrorCategory.SECURITY_ERROR)
                span.set_status(Status(StatusCode.ERROR, "Root boundary violation"))
                
                logger.warning(
                    "Root boundary violation detected",
                    extra={"folder_name": input.folder_name}
                )
                
                raise ValueError(
                    "Security error: Attempted to access path outside root boundary"
                )
            
            # Check if folder exists
            if not target_folder.exists():
                available_folders = [f.name for f in notes_base_path.iterdir() if f.is_dir()]
                
                metrics.record_tool_error("write_note", ErrorCategory.VALIDATION_ERROR)
                span.set_status(Status(StatusCode.ERROR, "Folder not found"))
                
                raise ValueError(
                    f"Folder '{input.folder_name}' not found. "
                    f"Available folders: {', '.join(available_folders) if available_folders else 'None'}. "
                    f"Use list_note_folders to see valid options."
                )
            
            # Prepare note path
            note_path = target_folder / input.file_name
            
            # Final validation that note path is within root
            if not validate_path_within_root(note_path, notes_base_path):
                metrics.record_tool_error("write_note", ErrorCategory.SECURITY_ERROR)
                raise ValueError("Security error: Note path outside root boundary")
            
            # 3. ELICITATION CHECK (User confirmation for overwrites)
            file_exists = note_path.exists()
            if file_exists and elicitation_manager.requires_confirmation("file_overwrite", {"path": str(note_path)}):
                # Check if confirmation was provided
                if not confirmation_id:
                    # Create confirmation request
                    new_confirmation_id = elicitation_manager.create_confirmation_request(
                        "file_overwrite",
                        {
                            "path": str(note_path),
                            "folder": input.folder_name,
                            "file": input.file_name,
                            "agent_id": agent_id
                        }
                    )
                    
                    return json.dumps({
                        "success": False,
                        "requires_confirmation": True,
                        "confirmation_id": new_confirmation_id,
                        "message": f"File '{input.folder_name}/{input.file_name}' already exists. "
                                   f"Please confirm overwrite by calling this tool again with confirmation_id.",
                        "context": {
                            "operation": "file_overwrite",
                            "existing_file": str(note_path)
                        }
                    }, indent=2)
                else:
                    # Validate confirmation
                    confirmation = elicitation_manager.check_confirmation(confirmation_id)
                    if not confirmation:
                        raise ValueError("Invalid or expired confirmation ID")
                    
                    # Check that confirmation matches this operation
                    if confirmation["context"]["path"] != str(note_path):
                        raise ValueError("Confirmation ID does not match this operation")
                    
                    # Consume confirmation
                    elicitation_manager.consume_confirmation(confirmation_id)
                    logger.info(f"File overwrite confirmed for {note_path}")
            
            # Write the note
            note_path.write_text(input.content, encoding='utf-8')
            
            # Day 2: Extract key facts from the note
            extracted_facts = knowledge_extractor.extract_key_facts(input.content)
            
            # Day 2: Update vector index for semantic search
            if SEMANTIC_SEARCH_AVAILABLE:
                try:
                    vector_manager.add_note_to_index(note_path, input.content)
                    logger.info(f"Added note to vector index: {note_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to add note to vector index: {e}")
            
            # Day 2: Consolidate extracted facts into long-term memory
            if extracted_facts.get('skills'):
                # Update skills in long-term memory
                existing_skills = memory_bank.get_long_term('skills').get('value', [])
                if isinstance(existing_skills, list):
                    # Merge new skills with existing ones
                    all_skills = list(set(existing_skills + extracted_facts['skills']))
                    memory_bank.update_long_term('skills', all_skills)
            
            if extracted_facts.get('preferences'):
                # Store preferences in long-term memory
                existing_prefs = memory_bank.get_long_term('preferences').get('value', [])
                if isinstance(existing_prefs, list):
                    all_prefs = existing_prefs + extracted_facts['preferences']
                    memory_bank.update_long_term('preferences', all_prefs)
            
            # Record latency
            latency_ms = (time.time() - start_time) * 1000
            metrics.record_tool_latency("write_note", latency_ms)
            
            # Audit log
            audit_log(
                action="write_note",
                details={
                    "folder": input.folder_name,
                    "file": input.file_name,
                    "size_bytes": len(input.content.encode('utf-8')),
                    "path": str(note_path.absolute()),
                    "extracted_skills": len(extracted_facts.get('skills', [])),
                    "extracted_preferences": len(extracted_facts.get('preferences', [])),
                    "file_overwritten": file_exists
                },
                agent_id=agent_id
            )
            
            span.set_status(Status(StatusCode.OK))
            
            logger.info(
                "Note written successfully",
                extra={
                    "folder": input.folder_name,
                    "file": input.file_name,
                    "size_bytes": len(input.content.encode('utf-8')),
                    "latency_ms": latency_ms,
                    "extracted_skills": len(extracted_facts.get('skills', [])),
                    "extracted_preferences": len(extracted_facts.get('preferences', []))
                }
            )
            
            # Prepare response with sanitization
            response_data = {
                "success": True,
                "message": f"Note successfully saved to {input.folder_name}/{input.file_name}",
                "path": str(note_path.absolute()),
                "size_bytes": len(input.content.encode('utf-8')),
                "extracted_facts": {
                    "skills_found": len(extracted_facts.get('skills', [])),
                    "preferences_found": len(extracted_facts.get('preferences', [])),
                    "indexed_for_search": SEMANTIC_SEARCH_AVAILABLE
                },
                "file_overwritten": file_exists
            }
            
            # Apply output sanitization
            sanitized_response = sanitize_output(response_data)
            
            return json.dumps(sanitized_response, indent=2)
        
        except ValueError as e:
            # Re-raise validation errors (already logged)
            raise
        except Exception as e:
            metrics.record_tool_error("write_note", ErrorCategory.IO_ERROR)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            
            logger.error(
                "Error writing note",
                extra={
                    "folder": input.folder_name,
                    "file": input.file_name,
                    "error": str(e)
                },
                exc_info=True
            )
            raise


def _get_health_status_impl() -> str:
    """
    Returns the current health status of the MCP server including metrics,
    performance statistics, and system status. Useful for monitoring,
    debugging, and understanding server behavior.
    
    Returns:
        JSON string with comprehensive health information
    """
    try:
        # Check if notes directory is accessible
        notes_accessible = notes_base_path.exists() and os.access(notes_base_path, os.W_OK)
        
        health_data = {
            "status": "healthy" if notes_accessible else "degraded",
            "version": "5.0.0",  # Enhanced version with full MCP primitives
            "environment": config.environment,
            "notes_path": str(notes_base_path.absolute()),
            "notes_accessible": notes_accessible,
            "roots_configured": True,
            "root_boundary": ROOT_BOUNDARY_URI,
            "security_features": {
                "pydantic_validation": True,
                "path_security": True,
                "root_boundaries": True,
                "audit_logging": config.enable_audit_log,
                "oauth_enabled": oauth_manager.is_enabled(),
                "rate_limiting_enabled": rate_limiter.is_enabled(),
                "elicitation_enabled": elicitation_manager.is_enabled(),
                "output_sanitization": True
            },
            "mcp_primitives": {
                "tools": True,
                "resources": True,
                "prompts": True,
                "elicitation": elicitation_manager.is_enabled()
            },
            "day2_features": {
                "semantic_search": SEMANTIC_SEARCH_AVAILABLE,
                "memory_bank": True,
                "vector_index_size": vector_manager.index.ntotal if SEMANTIC_SEARCH_AVAILABLE and vector_manager.index else 0,
                "short_term_memory_keys": len(memory_bank.short_term_memory),
                "long_term_memory_keys": len(memory_bank.long_term_memory)
            },
            "rate_limiting": {
                "enabled": rate_limiter.is_enabled(),
                "requests_per_window": config.rate_limit_requests,
                "window_seconds": config.rate_limit_window_seconds
            },
            "oauth": {
                "enabled": oauth_manager.is_enabled(),
                "active_tokens": len(oauth_manager.tokens) if oauth_manager.is_enabled() else 0
            },
            "metrics": metrics.get_stats(),
            "uptime_check": datetime.now(timezone.utc).isoformat()
        }
        
        return json.dumps(health_data, indent=2)
    
    except Exception as e:
        logger.error("Error checking health", exc_info=True)
        raise


def _search_notes_impl(input: SearchNotesInput) -> str:
    """
    Search notes using semantic similarity (vector embeddings).
    Unlike keyword search, this finds conceptually related notes even if 
    they don't contain the exact search terms.
    
    Args:
        input: Validated input containing query, max_results, and optional folder_name
    
    Returns:
        JSON string with search results and metadata
    """
    start_time = time.time()
    
    with tracer.start_as_current_span("search_notes") as span:
        try:
            metrics.record_tool_call("search_notes")
            
            span.set_attribute("query", input.query)
            span.set_attribute("max_results", input.max_results)
            
            # Check if semantic search is available
            if not SEMANTIC_SEARCH_AVAILABLE:
                return json.dumps({
                    "success": False,
                    "error": "Semantic search not available. Install: pip install sentence-transformers numpy faiss-cpu",
                    "results": []
                }, indent=2)
            
            # Rebuild index if empty
            if vector_manager.index.ntotal == 0:
                logger.info("Vector index empty, rebuilding...")
                vector_manager.rebuild_index(notes_base_path)
            
            # Perform semantic search
            results = vector_manager.search_similar_notes(input.query, input.max_results)
            
            # Filter by folder if specified
            if input.folder_name:
                results = [r for r in results if input.folder_name in r['path']]
            
            # Add more context to results
            enriched_results = []
            for result in results:
                note_path = Path(result['path'])
                if note_path.exists():
                    # Get note preview (first 200 chars)
                    content = result.get('content', '')
                    preview = content[:200] + "..." if len(content) > 200 else content
                    
                    enriched_results.append({
                        "file": note_path.name,
                        "folder": note_path.parent.name,
                        "path": str(note_path),
                        "similarity_score": result['similarity_score'],
                        "preview": preview,
                        "indexed_at": result.get('indexed_at', 'unknown')
                    })
            
            latency_ms = (time.time() - start_time) * 1000
            metrics.record_tool_latency("search_notes", latency_ms)
            
            span.set_attribute("results_found", len(enriched_results))
            span.set_status(Status(StatusCode.OK))
            
            logger.info(
                "Search completed",
                extra={
                    "query": input.query,
                    "results": len(enriched_results),
                    "latency_ms": latency_ms
                }
            )
            
            return json.dumps({
                "success": True,
                "query": input.query,
                "results": enriched_results,
                "total_results": len(enriched_results),
                "search_type": "semantic",
                "latency_ms": latency_ms
            }, indent=2)
        
        except Exception as e:
            metrics.record_tool_error("search_notes", ErrorCategory.INTERNAL_ERROR)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            
            logger.error("Error searching notes", exc_info=True)
            raise


def _update_memory_impl(input: UpdateMemoryInput) -> str:
    """
    Update the memory bank (short-term or long-term).
    
    Short-term memory is for session-specific context (e.g., current job posting).
    Long-term memory is for persistent user information (e.g., skills, preferences).
    
    Args:
        input: Validated input containing memory_type, key, and value
    
    Returns:
        JSON string with success confirmation
    """
    start_time = time.time()
    
    with tracer.start_as_current_span("update_memory") as span:
        try:
            metrics.record_tool_call("update_memory")
            
            span.set_attribute("memory_type", input.memory_type)
            span.set_attribute("key", input.key)
            
            if input.memory_type == "short_term":
                memory_bank.update_short_term(input.key, input.value)
            else:
                memory_bank.update_long_term(input.key, input.value)
            
            latency_ms = (time.time() - start_time) * 1000
            metrics.record_tool_latency("update_memory", latency_ms)
            
            span.set_status(Status(StatusCode.OK))
            
            # Audit log
            audit_log(
                action="update_memory",
                details={
                    "memory_type": input.memory_type,
                    "key": input.key,
                    "value_length": len(input.value)
                }
            )
            
            logger.info(
                "Memory updated",
                extra={
                    "memory_type": input.memory_type,
                    "key": input.key,
                    "latency_ms": latency_ms
                }
            )
            
            return json.dumps({
                "success": True,
                "message": f"Updated {input.memory_type} memory: {input.key}",
                "memory_type": input.memory_type,
                "key": input.key,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }, indent=2)
        
        except Exception as e:
            metrics.record_tool_error("update_memory", ErrorCategory.INTERNAL_ERROR)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            
            logger.error("Error updating memory", exc_info=True)
            raise


def _get_memory_impl(input: GetMemoryInput) -> str:
    """
    Retrieve memory from the memory bank.
    
    Can retrieve short-term, long-term, or all memory.
    Optionally filter by a specific key.
    
    Args:
        input: Validated input containing memory_type and optional key
    
    Returns:
        JSON string with memory contents
    """
    start_time = time.time()
    
    with tracer.start_as_current_span("get_memory") as span:
        try:
            metrics.record_tool_call("get_memory")
            
            span.set_attribute("memory_type", input.memory_type)
            if input.key:
                span.set_attribute("key", input.key)
            
            if input.memory_type == "short_term":
                memory_data = memory_bank.get_short_term(input.key)
            elif input.memory_type == "long_term":
                memory_data = memory_bank.get_long_term(input.key)
            else:  # all
                memory_data = memory_bank.get_all_memory()
            
            latency_ms = (time.time() - start_time) * 1000
            metrics.record_tool_latency("get_memory", latency_ms)
            
            span.set_status(Status(StatusCode.OK))
            
            logger.info(
                "Memory retrieved",
                extra={
                    "memory_type": input.memory_type,
                    "key": input.key,
                    "latency_ms": latency_ms
                }
            )
            
            return json.dumps({
                "success": True,
                "memory_type": input.memory_type,
                "key": input.key,
                "data": memory_data,
                "retrieved_at": datetime.now(timezone.utc).isoformat()
            }, indent=2)
        
        except Exception as e:
            metrics.record_tool_error("get_memory", ErrorCategory.INTERNAL_ERROR)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            
            logger.error("Error retrieving memory", exc_info=True)
            raise


# Register tools with FastMCP (decorators wrap the implementations)
@mcp.tool()
def list_note_folders() -> str:
    """
    Lists all available note category folders in the notes directory.
    Use this tool to 'Scan the Scene' and discover what categories exist
    before attempting to save a note. This prevents errors from assuming
    folder names and enables the model to make informed decisions about
    note categorization.
    
    Returns:
        JSON string containing list of folders with metadata
    """
    return _list_note_folders_impl()


@mcp.tool()
def write_note(input: WriteNoteInput) -> str:
    """
    Saves a text note into a specific folder. Use this only AFTER
    identifying the correct target folder via list_note_folders.
    
    This tool requires:
    - folder_name: The category folder (must exist)
    - file_name: The note filename (should end in .md)
    - content: The complete note content in markdown format
    - access_token: OAuth 2.0 token (optional if OAuth disabled)
    - confirmation_id: For file overwrites (if elicitation enabled)
    
    Enhanced security features:
    - OAuth 2.0 authorization (requires notes:write scope)
    - Rate limiting per agent
    - User confirmation for file overwrites
    - Output sanitization to prevent info leaks
    
    Args:
        input: Validated input containing all required fields
    
    Returns:
        JSON string with success confirmation and metadata, or
        confirmation request if file exists and elicitation is enabled
    """
    return _write_note_impl(input, input.access_token, input.confirmation_id)


@mcp.tool()
def get_health_status() -> str:
    """
    Returns the current health status of the MCP server including metrics,
    performance statistics, and system status. Useful for monitoring,
    debugging, and understanding server behavior.
    
    Returns:
        JSON string with comprehensive health information
    """
    return _get_health_status_impl()


@mcp.tool()
def search_notes(input: SearchNotesInput) -> str:
    """
    Search notes using semantic similarity (vector embeddings).
    
    This is NOT a simple keyword search. It uses AI embeddings to find
    conceptually related notes even if they don't contain the exact words
    you're searching for.
    
    Example: Searching for "database issue" will find notes about 
    "SQL performance problems" even if the word "issue" isn't mentioned.
    
    This enables true memory recall based on meaning, not just matching text.
    
    Args:
        input: Validated input containing query, max_results, and optional folder_name
    
    Returns:
        JSON string with search results ranked by semantic similarity
    """
    return _search_notes_impl(input)


@mcp.tool()
def update_memory(input: UpdateMemoryInput) -> str:
    """
    Update the dual-layer memory bank.
    
    Memory Types:
    - short_term: Session-specific context (e.g., current job posting being reviewed)
      Use this for temporary context that should be cleared between sessions.
    
    - long_term: Persistent entity memory (e.g., career history, skills, preferences)
      Use this for information that should be remembered across all sessions.
    
    Common Keys:
    - Short-term: 'current_job_posting', 'active_project', 'today_focus'
    - Long-term: 'skills', 'preferences', 'career_history', 'tone_of_voice'
    
    Args:
        input: Validated input containing memory_type, key, and value
    
    Returns:
        JSON string with success confirmation
    """
    return _update_memory_impl(input)


@mcp.tool()
def get_memory(input: GetMemoryInput) -> str:
    """
    Retrieve memory from the dual-layer memory bank.
    
    This allows the AI to access persistent information about the user,
    preventing the need to re-ask questions or lose context between sessions.
    
    Memory Types:
    - short_term: Current session context
    - long_term: Persistent user profile
    - all: Both layers of memory
    
    Optional key parameter allows retrieving a specific memory entry.
    
    Args:
        input: Validated input containing memory_type and optional key
    
    Returns:
        JSON string with memory contents
    """
    return _get_memory_impl(input)


@mcp.tool()
def request_authorization() -> str:
    """
    Request OAuth 2.0 authorization.
    
    This initiates the authorization flow by generating an authorization code.
    In a production environment, this would redirect the user to an OAuth provider.
    For this implementation, it generates a code that can be exchanged for a token.
    
    Returns:
        JSON with authorization code and instructions
    """
    try:
        if not oauth_manager.is_enabled():
            return json.dumps({
                "success": False,
                "message": "OAuth is not enabled. Set ENABLE_OAUTH=true to enable authorization."
            }, indent=2)
        
        # In production, agent_id would come from user authentication
        # For now, generate a unique agent ID
        agent_id = f"agent_{secrets.token_urlsafe(8)}"
        
        # Request all available scopes by default
        code = oauth_manager.generate_authorization_code(
            agent_id=agent_id,
            scopes=config.oauth_scopes
        )
        
        logger.info(f"Authorization code generated for agent: {agent_id}")
        
        return json.dumps({
            "success": True,
            "authorization_code": code,
            "agent_id": agent_id,
            "scopes": config.oauth_scopes,
            "expires_in_seconds": 600,
            "instructions": "Use exchange_token with this authorization code to get an access token."
        }, indent=2)
    
    except Exception as e:
        logger.error(f"Error requesting authorization: {e}", exc_info=True)
        raise


@mcp.tool()
def exchange_token(authorization_code: str) -> str:
    """
    Exchange authorization code for access token.
    
    This completes the OAuth 2.0 authorization flow by exchanging
    an authorization code for an access token that can be used with
    other tools.
    
    Args:
        authorization_code: The code received from request_authorization
    
    Returns:
        JSON with access token and token details
    """
    try:
        if not oauth_manager.is_enabled():
            return json.dumps({
                "success": False,
                "message": "OAuth is not enabled."
            }, indent=2)
        
        token = oauth_manager.exchange_code_for_token(authorization_code)
        
        if not token:
            return json.dumps({
                "success": False,
                "error": "Invalid or expired authorization code"
            }, indent=2)
        
        logger.info(f"Access token issued for agent: {token.agent_id}")
        
        audit_log(
            action="token_issued",
            details={
                "agent_id": token.agent_id,
                "scopes": token.scopes,
                "expires_at": token.expires_at.isoformat()
            },
            agent_id=token.agent_id
        )
        
        return json.dumps({
            "success": True,
            "access_token": token.access_token,
            "token_type": token.token_type,
            "expires_in_seconds": int((token.expires_at - datetime.now(timezone.utc)).total_seconds()),
            "scopes": token.scopes,
            "agent_id": token.agent_id,
            "instructions": "Include this access_token in the access_token field when calling protected tools."
        }, indent=2)
    
    except Exception as e:
        logger.error(f"Error exchanging token: {e}", exc_info=True)
        raise


# Export implementation functions for testing (bypasses FastMCP wrappers)
list_note_folders_impl = _list_note_folders_impl
write_note_impl = _write_note_impl
get_health_status_impl = _get_health_status_impl
search_notes_impl = _search_notes_impl
update_memory_impl = _update_memory_impl
get_memory_impl = _get_memory_impl


if __name__ == "__main__":
    """
    Entry point for the MCP server.
    FastMCP handles all protocol details, request routing, and response formatting.
    """
    logger.info("Starting MCP Server with FastMCP")
    mcp.run()
