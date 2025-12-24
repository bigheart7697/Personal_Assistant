# Strategic Note-Taking MCP Server

A **Production-Grade Level 2 Strategic Problem-Solver** AI assistant powered by the Model Context Protocol (MCP). This system demonstrates how to move beyond simple pattern prediction to strategic tool usage with the **Think-Act-Observe** loop.

**Version 3.0.0** - Built with FastMCP SDK, Pydantic validation, and explicit Roots security boundaries.

## 🎯 Project Overview

This project implements a production-grade MCP server that enables AI assistants to:
- **Discover** the current state of a notes directory dynamically
- **Reason** about the best categorization for new notes
- **Execute** note-saving actions with proper validation
- **Monitor** performance with OpenTelemetry tracing and metrics

### Key Design Principles

1. **No Hardcoding**: The model discovers folder structure at runtime
2. **Granular Tools**: Single-purpose tools for specific operations
3. **Context Engineering**: Comprehensive docstrings guide model behavior
4. **Error Handling**: Clear feedback for strategic decision-making
5. **Zero Boilerplate**: FastMCP SDK handles all protocol details
6. **Robust Validation**: Pydantic models ensure type safety
7. **Security First**: Explicit Roots boundaries prevent unauthorized access

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
│          MCP Server (v3.0.0)                         │
│  ┌─────────────────────────────────────────────┐   │
│  │  FastMCP SDK Layer                          │   │
│  │  • Protocol handling (zero boilerplate)     │   │
│  │  • Request routing                           │   │
│  │  • Pydantic validation                       │   │
│  └─────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────┐   │
│  │  Tools:                                      │   │
│  │  • list_note_folders (Discovery)            │   │
│  │  • write_note (Action + Validation)         │   │
│  │  • get_health_status (Monitoring)           │   │
│  └─────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────┐   │
│  │  Security & Observability:                   │   │
│  │  • Roots boundaries (file://notes/)         │   │
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
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG/INFO/WARNING/ERROR) |
| `ENABLE_TRACING` | `true` | Enable OpenTelemetry tracing |
| `ENABLE_AUDIT_LOG` | `true` | Enable audit logging |
| `AUDIT_LOG_PATH` | `./audit.log` | Path to audit log file |
| `MAX_FILE_SIZE_MB` | `10` | Maximum file size in megabytes |

The server communicates via stdin/stdout using JSON-RPC 2.0 protocol.

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
    folder_name: str  # 1-255 chars, no path traversal, no invalid chars
    file_name: str    # 1-255 chars, .md extension added if missing
    content: str      # Min 1 char, max 10MB (configurable)
```

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
  "size_bytes": 1024
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
  "version": "3.0.0",
  "environment": "production",
  "notes_path": "/path/to/notes",
  "notes_accessible": true,
  "roots_configured": true,
  "root_boundary": "/path/to/notes",
  "security_features": {
    "pydantic_validation": true,
    "path_security": true,
    "root_boundaries": true,
    "audit_logging": true
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

## 🔐 Security Features (v3.0.0)

### Implemented Security Measures

✅ **Roots Security Boundaries**
- Server confined to specific directory via `mcp.add_root()`
- All paths validated against root boundary
- Prevents access to sensitive system files (`/etc/passwd`, `C:\Windows`, etc.)

✅ **Pydantic Input Validation**
- Type safety enforced at schema level
- Field length limits (1-255 characters)
- Custom validators for security checks
- Automatic validation before execution

✅ **Path Traversal Prevention**
- Blocks `..`, `/`, `\` in folder names
- Validates resolved paths stay within root
- Multi-layer validation (Pydantic + runtime)

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
- Records all write operations
- Includes timestamp, action, agent_id, details
- Compliance-ready (SOC 2, HIPAA, GDPR)

### Security Configuration

```bash
# Enable all security features
MCP_ENV=production
ENABLE_AUDIT_LOG=true
AUDIT_LOG_PATH=/var/log/mcp-audit.log
MAX_FILE_SIZE_MB=10
```

### What's NOT Protected

⚠️ **File Overwriting**: Server will overwrite existing files without confirmation
⚠️ **Authentication**: No user authentication (assumes trusted AI agent)
⚠️ **Rate Limiting**: No built-in rate limiting (add upstream if needed)

## 🚧 Limitations & Future Enhancements

### Current Limitations
- No authentication/authorization (assumes trusted agent)
- No concurrent access handling (single-threaded)
- No note versioning/history
- Cannot create new folders (by design, requires manual creation)
- File overwriting without confirmation

### v3.0.0 Improvements

✅ **Eliminated Boilerplate**
- Migrated from manual JSON-RPC parsing to FastMCP SDK
- ~200 lines of protocol code replaced with decorators

✅ **Robust Validation**
- Pydantic models replace manual `if` checks
- Self-documenting schemas
- Automatic type coercion and validation

✅ **Explicit Security Boundaries**
- Roots pattern implemented via `mcp.add_root()`
- Path validation at multiple layers
- Clear security boundaries documented

✅ **Production Observability**
- OpenTelemetry distributed tracing
- Metrics collection (calls, errors, latencies)
- Structured logging to stderr
- Audit logging for compliance

### Planned Features (Future)
- Folder creation tool
- Note search and retrieval
- Job application tracking
- Task management
- Calendar integration
- Rate limiting
- Circuit breakers

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

