# Strategic Note-Taking MCP Server

A **Production-Grade Level 2 Strategic Problem-Solver** AI assistant powered by the Model Context Protocol (MCP). This system demonstrates how to move beyond simple pattern prediction to strategic tool usage with the **Think-Act-Observe** loop.

**Version 5.0.0** - Full MCP implementation with Resources, Prompts, Elicitation, OAuth 2.0, and Rate Limiting aligned with [Google's MCP Whitepaper](https://research.google/pubs/agent-tools-interoperability-with-model-context-protocol-mcp/).

## 🎯 Project Overview

This project implements a production-grade MCP server that enables AI assistants to:
- **Discover** the current state of a notes directory dynamically
- **Reason** about the best categorization for new notes
- **Execute** note-saving actions with proper validation
- **Monitor** performance with OpenTelemetry tracing and metrics
- **Read** notes directly via Resources primitive (URI-based access)
- **Follow** structured workflows using Prompts primitive
- **Confirm** sensitive operations with Elicitation (user confirmations)
- **Authenticate** using OAuth 2.0 authorization framework
- **Limit** usage with rate limiting per agent/session

### Key Design Principles

1. **No Hardcoding**: The model discovers folder structure at runtime
2. **Granular Tools**: Single-purpose tools for specific operations
3. **Context Engineering**: Comprehensive docstrings guide model behavior
4. **Error Handling**: Clear feedback for strategic decision-making
5. **Zero Boilerplate**: FastMCP SDK handles all protocol details
6. **Robust Validation**: Pydantic models ensure type safety
7. **Security First**: Explicit Roots boundaries prevent unauthorized access
8. **MCP Compliant**: Full implementation of MCP primitives (Tools, Resources, Prompts, Elicitation)
9. **Enterprise Ready**: OAuth 2.0, rate limiting, audit logging, and output sanitization
10. **Whitepaper Aligned**: Follows Google's MCP security recommendations and best practices

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   AI Assistant                       │
│            (Claude, GPT-4, etc.)                     │
└───────────────────┬─────────────────────────────────┘
                    │
                    │ JSON-RPC 2.0 over stdio
                    │
┌───────────────────▼─────────────────────────────────┐
│          MCP Server (v5.0.0)                         │
│  ┌─────────────────────────────────────────────┐   │
│  │  FastMCP SDK Layer                          │   │
│  │  • Protocol handling (zero boilerplate)     │   │
│  │  • Request routing                           │   │
│  │  • Pydantic validation                       │   │
│  └─────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────┐   │
│  │  MCP Primitives:                             │   │
│  │  • Tools (list, write, search, memory)      │   │
│  │  • Resources (note://folder/file URIs)      │   │
│  │  • Prompts (workflows & templates)          │   │
│  │  • Elicitation (user confirmations)         │   │
│  └─────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────┐   │
│  │  Security & Observability:                   │   │
│  │  • OAuth 2.0 authorization                   │   │
│  │  • Rate limiting per agent                   │   │
│  │  • Roots boundaries (file://notes/)         │   │
│  │  • Output sanitization                       │   │
│  │  • OpenTelemetry tracing                     │   │
│  │  • Metrics collection                        │   │
│  │  • Audit logging                             │   │
│  └─────────────────────────────────────────────┘   │
└───────────────────┬─────────────────────────────────┘
                    │
                    │ File System (Confined to Root)
                    │
┌───────────────────▼─────────────────────────────────┐
│       Notes Directory (Root Boundary)                │
│  ./notes/                                            │
│    ├── personal/                                     │
│    ├── work/                                         │
│    └── job-applications/                            │
└─────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Required dependencies:
  - `fastmcp` - MCP SDK (eliminates boilerplate)
  - `pydantic` - Input validation
  - `opentelemetry-*` - Observability (optional)

### Installation

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\Activate.ps1
# Unix/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the MCP Server

```bash
# Development mode (default)
python mcp_server.py

# Production mode with all features
MCP_ENV=production \
NOTES_PATH=/path/to/notes \
LOG_LEVEL=INFO \
ENABLE_TRACING=true \
ENABLE_AUDIT_LOG=true \
python mcp_server.py
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_ENV` | `development` | Environment mode (development/production) |
| `NOTES_PATH` | `./notes` | Root directory for notes storage |
| `MEMORY_PATH` | `./memory` | Directory for memory bank storage |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG/INFO/WARNING/ERROR) |
| `ENABLE_TRACING` | `true` | Enable OpenTelemetry tracing |
| `ENABLE_AUDIT_LOG` | `true` | Enable audit logging |
| `AUDIT_LOG_PATH` | `./audit.log` | Path to audit log file |
| `MAX_FILE_SIZE_MB` | `10` | Maximum file size in megabytes |
| `ENABLE_RATE_LIMITING` | `true` | Enable rate limiting per agent |
| `RATE_LIMIT_REQUESTS` | `100` | Maximum requests per window |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Rate limit window in seconds |
| `ENABLE_OAUTH` | `false` | Enable OAuth 2.0 authorization |
| `OAUTH_CLIENT_ID` | `` | OAuth client ID |
| `OAUTH_CLIENT_SECRET` | `` | OAuth client secret |
| `OAUTH_TOKEN_URL` | `` | OAuth token endpoint URL |
| `OAUTH_SCOPES` | `notes:read notes:write` | Space-separated OAuth scopes |
| `ENABLE_ELICITATION` | `true` | Enable user confirmations |
| `REQUIRE_CONFIRMATION_OVERWRITE` | `true` | Require confirmation for file overwrites |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model for embeddings |

The server communicates via stdin/stdout using JSON-RPC 2.0 protocol.

## 🔷 MCP Primitives Implementation

This server implements all major MCP primitives as defined in the [MCP specification](https://modelcontextprotocol.io/) and aligned with Google's whitepaper recommendations:

### 1. **Tools** (Full CRUD Operations)
Standard tool calling interface for actions that modify state or perform operations.

### 2. **Resources** (URI-Based Read Access)
Direct read access to notes using URI scheme: `note://folder/filename.md`

**Examples:**
- `note://personal/example-note.md` - Read a specific note
- `note://work/` - List all notes in the work folder

**Benefits:**
- No need for custom read tools
- Standard URI-based addressing
- Efficient for large-scale note retrieval

### 3. **Prompts** (Workflow Templates)
Predefined structured prompts for common workflows:

- **`job_application_workflow`** - Complete job application review process
- **`project_idea_brainstorm`** - Creative ideation and documentation
- **`daily_reflection`** - Note review and knowledge consolidation

**Benefits:**
- Consistent, repeatable processes
- Reduces prompt engineering burden
- Guides AI through complex multi-step tasks

### 4. **Elicitation** (User Confirmations)
Server-initiated requests for user confirmation on sensitive operations:

- File overwrites (when enabled)
- Large deletions (future)
- Permanent modifications (future)

**Flow:**
1. Tool detects sensitive operation
2. Returns confirmation request with `confirmation_id`
3. User approves/denies
4. Tool called again with `confirmation_id` to complete

### 5. **Authorization** (OAuth 2.0)
OAuth 2.0 authorization code grant flow for secure access:

**Scopes:**
- `notes:read` - Read notes via Resources
- `notes:write` - Create/update notes
- `memory:read` - Access memory bank
- `memory:write` - Modify memory

**Flow:**
1. Call `request_authorization` to get auth code
2. Call `exchange_token` with auth code to get access token
3. Include `access_token` in tool calls

### 6. **Rate Limiting**
Per-agent rate limiting prevents abuse:
- Default: 100 requests per 60 seconds
- Configurable via environment variables
- Per-agent tracking (not per-IP)

## 📋 Available Tools

### 1. `list_note_folders`

**Purpose:** Discovery tool that lists all category folders in the notes directory

**Input:** No arguments required

**Output Example:**
```json
{
  "folders": [
    {"name": "personal", "note_count": 5},
    {"name": "work", "note_count": 12},
    {"name": "job-applications", "note_count": 3}
  ],
  "total_categories": 3,
  "notes_path": "/absolute/path/to/notes"
}
```

**Security:** Only lists folders within the configured Roots boundary

**When to Use:** ALWAYS use this before attempting to save a note

---

### 2. `write_note`

**Purpose:** Action tool that saves a markdown note to a specific folder with robust validation

**Input (Pydantic Validated):**
```python
class WriteNoteInput(BaseModel):
    folder_name: str         # 1-255 chars, no path traversal, no invalid chars
    file_name: str           # 1-255 chars, .md extension added if missing
    content: str             # Min 1 char, max 10MB (configurable)
    access_token: str        # OAuth token (optional if OAuth disabled)
    confirmation_id: str     # For file overwrites (optional if elicitation disabled)
```

**Enhanced Security Features:**
- ✅ **OAuth 2.0**: Requires `notes:write` scope when enabled
- ✅ **Rate Limiting**: Enforced per agent
- ✅ **Elicitation**: Requests confirmation for file overwrites
- ✅ **Output Sanitization**: Prevents information leaks
- ✅ **Path Traversal Prevention**: Multiple validation layers
- ✅ **Audit Logging**: Full trail of all write operations

**Validation Rules:**
- ✅ Folder name: No `..`, `/`, `\`, or invalid characters (`<>:"|?*`)
- ✅ No hidden directories (starting with `.`)
- ✅ Must be within Roots boundary
- ✅ Folder must exist
- ✅ File size limit enforced
- ✅ Automatic `.md` extension addition

**Output Example (Success):**
```json
{
  "success": true,
  "message": "Note successfully saved to work/project-ideas.md",
  "path": "/absolute/path/to/notes/work/project-ideas.md",
  "size_bytes": 1024,
  "file_overwritten": false,
  "extracted_facts": {
    "skills_found": 3,
    "preferences_found": 1,
    "indexed_for_search": true
  }
}
```

**Output Example (Elicitation Required):**
```json
{
  "success": false,
  "requires_confirmation": true,
  "confirmation_id": "abc123xyz",
  "message": "File 'work/project-ideas.md' already exists. Please confirm overwrite.",
  "context": {
    "operation": "file_overwrite",
    "existing_file": "/path/to/notes/work/project-ideas.md"
  }
}
```

**Output Example (Validation Error):**
```
ValidationError: Invalid folder name: path traversal detected
```

**When to Use:** ONLY after identifying valid folders with `list_note_folders`

---

### 3. `get_health_status`

**Purpose:** Monitoring tool that returns server health and performance metrics

**Input:** No arguments required

**Output Example:**
```json
{
  "status": "healthy",
  "version": "5.0.0",
  "environment": "production",
  "notes_path": "/path/to/notes",
  "notes_accessible": true,
  "roots_configured": true,
  "root_boundary": "file:///path/to/notes",
  "security_features": {
    "pydantic_validation": true,
    "path_security": true,
    "root_boundaries": true,
    "audit_logging": true,
    "oauth_enabled": false,
    "rate_limiting_enabled": true,
    "elicitation_enabled": true,
    "output_sanitization": true
  },
  "mcp_primitives": {
    "tools": true,
    "resources": true,
    "prompts": true,
    "elicitation": true
  },
  "day2_features": {
    "semantic_search": true,
    "memory_bank": true,
    "vector_index_size": 42,
    "short_term_memory_keys": 3,
    "long_term_memory_keys": 8
  },
  "rate_limiting": {
    "enabled": true,
    "requests_per_window": 100,
    "window_seconds": 60
  },
  "oauth": {
    "enabled": false,
    "active_tokens": 0
  },
  "metrics": {
    "tool_calls": {"write_note": 45, "list_note_folders": 12},
    "tool_errors": {"write_note": 2},
    "tool_latencies": {
      "write_note": {"avg_ms": 12.5, "min_ms": 8, "max_ms": 25}
    }
  }
}
```

**When to Use:** For monitoring, debugging, and health checks

---

### 4. `search_notes`

**Purpose:** Semantic search tool for finding related notes by meaning, not keywords

**Input (Pydantic Validated):**
```python
class SearchNotesInput(BaseModel):
    query: str              # 1-500 chars, semantic search query
    max_results: int = 5    # 1-20, number of results
    folder_name: str        # Optional, search within specific folder
```

**Example Query:** "database performance issues" will find notes about "SQL optimization" even without matching keywords.

**When to Use:** Finding related notes across your knowledge base

---

### 5. `update_memory`

**Purpose:** Update the dual-layer memory bank (short-term or long-term)

**Input (Pydantic Validated):**
```python
class UpdateMemoryInput(BaseModel):
    memory_type: str  # 'short_term' or 'long_term'
    key: str          # Memory key (e.g., 'skills', 'current_project')
    value: str        # Value to store
```

**When to Use:** Storing session context (short-term) or persistent user data (long-term)

---

### 6. `get_memory`

**Purpose:** Retrieve memory from the dual-layer memory bank

**Input (Pydantic Validated):**
```python
class GetMemoryInput(BaseModel):
    memory_type: str  # 'short_term', 'long_term', or 'all'
    key: str          # Optional, specific memory key
```

**When to Use:** Accessing stored context and user profile information

---

### 7. `request_authorization` (OAuth Tools)

**Purpose:** Request OAuth 2.0 authorization code

**Input:** No arguments required

**Output Example:**
```json
{
  "success": true,
  "authorization_code": "auth_abc123xyz",
  "agent_id": "agent_xyz789",
  "scopes": ["notes:read", "notes:write", "memory:read", "memory:write"],
  "expires_in_seconds": 600,
  "instructions": "Use exchange_token with this authorization code to get an access token."
}
```

**When to Use:** First step of OAuth flow when OAuth is enabled

---

### 8. `exchange_token` (OAuth Tools)

**Purpose:** Exchange authorization code for access token

**Input:**
```python
authorization_code: str  # Code from request_authorization
```

**Output Example:**
```json
{
  "success": true,
  "access_token": "token_abc123xyz",
  "token_type": "Bearer",
  "expires_in_seconds": 3600,
  "scopes": ["notes:read", "notes:write", "memory:read", "memory:write"],
  "agent_id": "agent_xyz789",
  "instructions": "Include this access_token in the access_token field when calling protected tools."
}
```

**When to Use:** Second step of OAuth flow to get usable access token

## 🔧 MCP Protocol Implementation

### Message Format

All messages follow JSON-RPC 2.0 specification:

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "list_note_folders",
    "arguments": {}
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [{
      "type": "text",
      "text": "{\"folders\": [...], ...}"
    }]
  }
}
```

### Supported Methods

- `initialize` - Handshake and capability negotiation
- `tools/list` - Get list of available tools with schemas
- `tools/call` - Execute a specific tool

### Error Codes

| Code | Meaning | When It Occurs |
|------|---------|----------------|
| -32700 | Parse error | Invalid JSON received |
| -32600 | Invalid Request | Missing jsonrpc version |
| -32601 | Method not found | Unknown method called |
| -32602 | Invalid params | Missing required arguments |
| -32603 | Internal error | Server-side exception |

## 🧠 The Think-Act-Observe Loop

This system demonstrates Level 2 strategic reasoning:

```
1. OBSERVE: User provides note → Assistant calls list_note_folders
2. THINK: Assistant analyzes folders and content
3. ACT: Assistant calls write_note with valid folder
4. OBSERVE: Assistant receives success/error feedback
5. [Loop continues based on result]
```

### Example Interaction

**User:** "Save this note about my Python project: Building a web scraper using BeautifulSoup"

**Assistant's Internal Process:**

```python
# 1. OBSERVE
→ Call: list_note_folders()
← Result: ["personal", "programming", "job-search"]

# 2. THINK
"The note is about Python programming. The 'programming' folder is most appropriate."

# 3. ACT
→ Call: write_note(
    folder_name="programming",
    file_name="web-scraper-project.md",
    content="# Web Scraper Project\n\nBuilding a web scraper using BeautifulSoup..."
)
← Result: Success

# 4. CONFIRM
"I've saved your note to programming/web-scraper-project.md"
```

## 📁 Directory Structure

```
Personal Assistant/
├── mcp_server.py           # Main MCP server implementation
├── system_prompt.md        # Orchestration prompt for AI
├── README.md              # This file
├── requirements.txt       # Python dependencies (none!)
├── test_mcp_server.py     # Test suite
└── notes/                 # Note storage (created automatically)
    ├── personal/
    ├── work/
    └── job-applications/
```

## 🧪 Testing

### Manual Testing with JSON-RPC

```bash
# Start the server
python mcp_server.py

# Send test request (paste into stdin)
{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}

# Expected response
{"jsonrpc":"2.0","id":1,"result":{"tools":[...]}}
```

### Automated Testing

```bash
python test_mcp_server.py
```

## 🔐 Security Features (v5.0.0)

### Comprehensive Security Implementation

This implementation addresses all major security concerns from the MCP whitepaper:

✅ **OAuth 2.0 Authorization** (Whitepaper Page 39)
- Authorization code grant flow
- Scope-based access control (`notes:read`, `notes:write`, `memory:read`, `memory:write`)
- Token expiration and refresh
- Per-agent authentication
- **Addresses**: "No support for limiting the scope of access" concern

✅ **Elicitation (User Confirmations)** (Whitepaper Page 33)
- Server-initiated confirmation requests
- One-time use confirmation tokens
- Timeout protection (5-minute expiry)
- **Addresses**: File overwrite risks and user control

✅ **Rate Limiting** (Whitepaper Security Section)
- Per-agent rate limiting (default: 100 req/min)
- Sliding window algorithm
- Configurable limits and windows
- **Addresses**: DoS attacks and resource exhaustion

✅ **Output Sanitization** (Whitepaper Page 45)
- Truncates long outputs to prevent info leaks
- Configurable preview lengths
- Applied to all tool responses
- **Addresses**: "Sensitive information Leaks" concern

✅ **Roots Security Boundaries** (MCP Specification)
- Server confined to specific directory
- All paths validated against root boundary
- Prevents access to sensitive system files
- Multi-layer path validation

✅ **Pydantic Input Validation**
- Type safety enforced at schema level
- Field length limits (1-255 characters)
- Custom validators for security checks
- Automatic validation before execution

✅ **Path Traversal Prevention**
- Blocks `..`, `/`, `\` in folder names
- Validates resolved paths stay within root
- Multi-layer validation (Pydantic + runtime)
- **Addresses**: "Tool Shadowing" and path-based attacks

✅ **Invalid Character Filtering**
- Rejects `<>:"|?*` characters
- Prevents shell injection attempts
- Blocks hidden directories (`.git`, `.env`)

✅ **File Size Limits**
- Configurable maximum (default 10MB)
- Pre-write validation
- Prevents disk exhaustion attacks

✅ **Audit Logging**
- JSON-formatted audit trail
- Records all write operations with agent_id
- Includes timestamp, action, details
- Compliance-ready (SOC 2, HIPAA, GDPR)

### Security Configuration

```bash
# Production security configuration
MCP_ENV=production

# OAuth 2.0 (optional but recommended)
ENABLE_OAUTH=true
OAUTH_CLIENT_ID=your_client_id
OAUTH_CLIENT_SECRET=your_client_secret
OAUTH_TOKEN_URL=https://oauth.example.com/token
OAUTH_SCOPES="notes:read notes:write memory:read memory:write"

# Rate limiting
ENABLE_RATE_LIMITING=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW_SECONDS=60

# Elicitation
ENABLE_ELICITATION=true
REQUIRE_CONFIRMATION_OVERWRITE=true

# Audit logging
ENABLE_AUDIT_LOG=true
AUDIT_LOG_PATH=/var/log/mcp-audit.log

# File limits
MAX_FILE_SIZE_MB=10
```

### Security Alignment with MCP Whitepaper

| Whitepaper Concern | Our Implementation | Status |
|-------------------|-------------------|---------|
| Tool Shadowing (p.42) | Path validation, Roots boundaries | ✅ Mitigated |
| Malicious Tool Definitions (p.44) | Pydantic validation, input sanitization | ✅ Mitigated |
| Sensitive Info Leaks (p.45) | Output sanitization, truncation | ✅ Mitigated |
| No Scope Limiting (p.46) | OAuth 2.0 with granular scopes | ✅ Addressed |
| Confused Deputy (p.49-51) | Multi-layer validation, confirmation | ✅ Mitigated |
| Authorization Missing (p.38-39) | OAuth 2.0 code grant flow | ✅ Implemented |
| Rate Limiting | Per-agent rate limiting | ✅ Implemented |

### What's Now Protected (v5.0.0)

✅ **File Overwriting**: Elicitation requests user confirmation  
✅ **Authentication**: OAuth 2.0 with scope-based access  
✅ **Rate Limiting**: Built-in per-agent rate limiting  
✅ **Info Leaks**: Output sanitization prevents excessive disclosure

## 🚧 Evolution & Roadmap

### v5.0.0 Improvements (Current) ✨

✅ **Full MCP Primitives**
- Resources primitive for URI-based note reading
- Prompts primitive for workflow templates
- Elicitation primitive for user confirmations
- All core MCP capabilities implemented

✅ **Enterprise Security**
- OAuth 2.0 authorization with scope control
- Rate limiting per agent/session
- Output sanitization to prevent info leaks
- Enhanced audit logging with agent tracking

✅ **Whitepaper Alignment**
- Addresses all major security concerns from Google's MCP whitepaper
- Implements recommended best practices
- Mitigates identified risks (Tool Shadowing, Info Leaks, etc.)

### Previous Improvements

**v4.0.0 (Day 2 Features):**
- Dual-layer memory bank (short-term/long-term)
- Semantic search with vector embeddings
- Knowledge extraction from notes
- FAISS indexing for similarity search

**v3.0.0 (Production Grade):**
- FastMCP SDK migration (eliminated boilerplate)
- Pydantic validation models
- Explicit Roots security boundaries
- OpenTelemetry tracing and metrics
- Structured logging and audit trails

**v2.0.0 (Observability):**
- Metrics collection
- Error categorization
- Health check endpoint
- Configuration management

**v1.0.0 (Foundation):**
- Basic MCP server with Tools primitive
- Strategic Think-Act-Observe loop
- File-based note storage

### Current Limitations
- No concurrent access handling (single-threaded)
- No note versioning/history
- Cannot create new folders (by design, requires manual creation)
- OAuth requires external provider in production
- In-memory rate limiting (resets on restart)

### Planned Features (Future)
- Folder creation tool
- Note versioning and history
- Concurrent request handling
- Persistent rate limit storage (Redis)
- External OAuth provider integration
- Multi-user/multi-workspace support
- Calendar integration
- Circuit breakers for resilience
- Webhook notifications
- Backup and restore functionality

## 📚 Learning Resources

- [Model Context Protocol Specification](https://modelcontextprotocol.io/)
- [JSON-RPC 2.0 Specification](https://www.jsonrpc.org/specification)
- [Strategic AI Reasoning Patterns](https://arxiv.org/abs/2201.11903)

## 🤝 Contributing

This is a learning project demonstrating MCP implementation. Feel free to:
- Extend with additional tools
- Improve error handling
- Add tests
- Enhance documentation

## 📄 License

MIT License - Feel free to use this for learning and projects

---

**Built with the Model Context Protocol for strategic AI reasoning**

