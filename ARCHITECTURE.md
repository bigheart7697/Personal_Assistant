# Architecture & Design Decisions

## Overview

This document explains the architectural decisions behind the Strategic Note-Taking MCP Server and how it implements Level 2 reasoning.

**Version 2.0.0 - Production Grade** ✨

This implementation has been enhanced from Day 1 foundations to production-ready status following Google's "Building for Production" guidelines from the Agent Workshop training.

## Production Features (v2.0.0)

### 1. Observability & Monitoring

**OpenTelemetry Distributed Tracing**
- Full instrumentation of all tool calls
- Span attributes for debugging (folder names, file sizes, latencies)
- Error tracking with exception recording
- Configurable exporters (Console for dev, OTLP for production)

**Structured Logging**
- Context-rich log entries with metadata
- Separate handlers for application logs (stderr) and JSON-RPC (stdout)
- Log levels configurable via environment variables
- No interference with protocol communication

**Metrics Collection**
- In-memory metrics collector tracking:
  - Tool call counts per tool
  - Error counts by tool and category
  - Latency statistics (min, max, avg)
  - Error categorization for analysis

### 2. Security & Governance

**Enhanced Path Validation**
- Path traversal attack prevention (`../` detection)
- Invalid character blocking (`<>:"|?*`)
- Hidden directory protection (`.` prefix)
- Absolute path resolution verification
- Multi-layered validation before file operations

**Audit Logging**
- Separate audit log file for compliance
- Captures: timestamp, action, agent identity, file details
- JSON-formatted entries for log analysis tools
- Configurable via environment variables

**File Size Limits**
- Configurable maximum file size (default 10MB)
- Pre-write validation to prevent disk exhaustion
- Clear error messages when limits exceeded

### 3. Configuration Management

**Environment-Based Configuration**
```bash
# Available environment variables:
MCP_ENV=development|staging|production
NOTES_PATH=./notes
LOG_LEVEL=DEBUG|INFO|WARNING|ERROR
ENABLE_TRACING=true|false
ENABLE_AUDIT_LOG=true|false
AUDIT_LOG_PATH=./audit.log
MAX_FILE_SIZE_MB=10
```

**Config Object Benefits**
- Single source of truth
- Easy testing with mock configs
- Environment-specific behavior
- No hardcoded values

### 4. Error Categorization

**Systematic Error Classification**
- `VALIDATION_ERROR`: Input validation failures
- `IO_ERROR`: File system operations
- `PROTOCOL_ERROR`: JSON-RPC violations
- `SECURITY_ERROR`: Security validation failures  
- `INTERNAL_ERROR`: Unexpected server errors

**Benefits**
- Better error tracking and debugging
- Metrics by error type
- Informed retry strategies
- Root cause analysis

### 5. Health Check Endpoint

**New Tool: `get_health_status`**
```json
{
  "status": "healthy|degraded",
  "version": "2.0.0",
  "environment": "production",
  "notes_accessible": true,
  "metrics": {
    "tool_calls": {...},
    "tool_errors": {...},
    "tool_latencies": {...}
  }
}
```

**Use Cases**
- Monitoring system integration
- Load balancer health checks
- Debugging and diagnostics
- Performance analysis

## Design Philosophy

### 1. Level 2 Strategic Reasoning

**What it means:**
- **Level 1**: Pattern prediction (like autocomplete)
- **Level 2**: Strategic problem-solving with environment awareness
- **Level 3**: Multi-step planning (future enhancement)

**How we achieve it:**
- Tools provide **observation** capabilities
- Comprehensive schemas guide **thinking**
- Clear error messages enable **adaptation**

### 2. The Think-Act-Observe Loop

```
┌─────────────────────────────────────┐
│  1. OBSERVE                         │
│  Call: list_note_folders()          │
│  Result: ["personal", "work", ...]  │
└─────────┬───────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│  2. THINK                           │
│  - Analyze user input               │
│  - Match to available categories    │
│  - Choose best fit or suggest new   │
└─────────┬───────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│  3. ACT                             │
│  Call: write_note(folder, file, content)
│  With: Validated folder name        │
└─────────┬───────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│  4. OBSERVE (feedback)              │
│  Result: Success or Error           │
│  Adapt: If error, try different approach
└─────────────────────────────────────┘
```

## Component Architecture

### MCP Server (`mcp_server.py`)

**Responsibilities:**
1. JSON-RPC 2.0 protocol handling
2. Tool registration and schema management
3. File system operations
4. Error handling and validation
5. **NEW:** Observability and tracing
6. **NEW:** Security validation
7. **NEW:** Metrics collection
8. **NEW:** Audit logging

**Key Classes:**

```python
class ServerConfig:
    """Environment-based configuration management"""
    - environment: str          # dev/staging/production
    - notes_path: str          # Base directory for notes
    - log_level: str           # Logging verbosity
    - enable_tracing: bool     # Toggle OpenTelemetry
    - enable_audit_log: bool   # Toggle audit logging
    - max_file_size_mb: int    # File size limits

class ErrorCategory(Enum):
    """Error classification for metrics and analysis"""
    VALIDATION_ERROR
    IO_ERROR
    PROTOCOL_ERROR
    SECURITY_ERROR
    INTERNAL_ERROR

class Metrics:
    """In-memory metrics collector"""
    - record_tool_call(tool_name)
    - record_tool_error(tool_name, category)
    - record_tool_latency(tool_name, ms)
    - get_stats() -> Dict

class MCPServer:
    # Configuration & Setup
    - __init__(config)           # Initialize with ServerConfig
    - _setup_logging()           # Structured logging setup
    - _setup_tracing()           # OpenTelemetry configuration
    - _setup_audit_logging()     # Audit log file handler
    
    # Security
    - _validate_path_security()  # Enhanced path validation
    - _audit_log()               # Record auditable actions
    
    # Protocol Handlers
    - handle_request(request)    # Main JSON-RPC router
    - handle_initialize()        # MCP handshake
    - handle_tools_list()        # Tool discovery
    - handle_tools_call()        # Tool execution (instrumented)
    
    # Tools (All instrumented with tracing & metrics)
    - list_note_folders()        # Discovery tool
    - write_note()               # Action tool (with security)
    - get_health_status()        # NEW: Health check tool
    
    # Utilities
    - create_error_response()    # JSON-RPC error formatting
    - create_success_response()  # JSON-RPC success formatting
    - run()                      # Main server loop
```

### Transport Layer

**Protocol:** JSON-RPC 2.0  
**Transport:** stdio (stdin/stdout)  
**Why stdio?**
- Fast local communication
- Simple process management
- No network configuration needed
- Standard for MCP servers

**Message Flow:**

```
AI Assistant                    MCP Server
     │                               │
     ├──── JSON Request ────────────▶│
     │     (via stdin)               │
     │                               │
     │                          [Process]
     │                               │
     │◀──── JSON Response ───────────┤
     │     (via stdout)              │
     │                               │
```

### Tool Design

#### Tool 1: `list_note_folders`

**Type:** Discovery/Observation  
**Granularity:** Single purpose  
**Input:** None (parameterless)  
**Output:** Structured JSON with metadata

**Why no parameters?**
- Simplifies usage
- Clear single purpose
- No decision-making required from AI

**Output structure:**
```json
{
  "folders": [
    {"name": "personal", "note_count": 5}
  ],
  "total_categories": 1,
  "notes_path": "/absolute/path"
}
```

**Design decisions:**
- Include note counts (helps AI understand usage)
- Provide absolute path (debugging aid)
- Return structured data (easier to parse)

#### Tool 2: `write_note`

**Type:** Action/Execution  
**Granularity:** Single purpose (write only)  
**Input:** Three required parameters  
**Output:** Success message or detailed error

**Why three parameters?**
- `folder_name`: Must be validated against discovered folders
- `file_name`: Enables descriptive naming
- `content`: Full text (no truncation)

**Validation chain:**
1. Check parameters exist
2. Validate folder_name against filesystem
3. Ensure file_name has .md extension
4. Write atomically to prevent corruption

**Error messages:**
- Specific: "Folder 'X' not found"
- Actionable: "Use list_note_folders to see options"
- Helpful: Lists available folders

### Schema Engineering

**Purpose:** The schema IS the instruction manual for the AI

**Key principles:**

1. **Comprehensive descriptions**
```python
"description": (
    "Lists all available note category folders. "
    "Use this tool to 'Scan the Scene' and discover "
    "what categories exist before attempting to save a note."
)
```

2. **Clear expectations**
```python
"Use this only AFTER identifying the correct target folder"
```

3. **Contextual guidance**
```python
"This prevents errors from assuming folder names"
```

## Error Handling Strategy

### JSON-RPC Error Codes

| Code | Name | Usage |
|------|------|-------|
| -32700 | Parse error | Malformed JSON |
| -32600 | Invalid Request | Missing jsonrpc version |
| -32601 | Method not found | Unknown method or tool |
| -32602 | Invalid params | Missing required arguments |
| -32603 | Internal error | Server exception |

### Business Logic Errors

Returned as successful responses with `isError: true`:

```json
{
  "content": [{
    "type": "text",
    "text": "Folder 'projects' not found. Use list_note_folders..."
  }],
  "isError": true
}
```

**Why not JSON-RPC errors?**
- Business logic failures aren't protocol errors
- Enables richer error messages
- AI can parse and respond appropriately

## Security Considerations

### Path Traversal Prevention

```python
target_folder = self.notes_base_path / folder_name
```

Using `pathlib` ensures:
- Paths stay within notes directory
- No `../` attacks possible
- Cross-platform path handling

### Input Validation

1. **folder_name**: Checked against actual directories
2. **file_name**: Sanitized and .md extension enforced
3. **content**: Accepted as-is (user's data)

### Production Enhancements (✅ Completed in v2.0.0)

- [x] **Filename sanitization** - Enhanced path validation with security checks
- [x] **File size limits** - Configurable via MAX_FILE_SIZE_MB
- [x] **Audit logging** - Full audit trail with file handler
- [x] **OpenTelemetry tracing** - Distributed tracing for observability
- [x] **Structured logging** - Context-rich logs with metadata
- [x] **Metrics collection** - Performance and usage tracking
- [x] **Error categorization** - Systematic error classification
- [x] **Health checks** - Monitoring endpoint for system status
- [x] **Configuration management** - Environment-based config

### Future Enhancements (Day 3+)

- [ ] Rate limiting per agent/user
- [ ] Distributed tracing to OTLP collector
- [ ] Prometheus metrics exporter
- [ ] Circuit breaker pattern for resilience

## Scalability Considerations

### Current Scope (Day 1)

- Local filesystem only
- Single-threaded
- Synchronous I/O
- No caching

**Why?**
- Simplicity for learning
- Sufficient for personal use
- Easy to understand and debug

### Future Scaling (Day 2+)

- [ ] Database backend (SQLite/PostgreSQL)
- [ ] Full-text search
- [ ] Concurrent request handling
- [ ] Cloud sync
- [ ] Multi-user support

## Testing Strategy

### Unit Tests (`test_mcp_server.py`)

**Coverage:**
- Protocol compliance (JSON-RPC 2.0)
- Tool functionality (all three tools)
- Error handling (all error paths)
- Edge cases (missing params, invalid folders)
- **NEW:** Security validation (path traversal, invalid chars)
- **NEW:** Health check endpoint
- **NEW:** Metrics collection

**Test isolation:**
- Temporary directories for each test
- Mock ServerConfig with disabled tracing/audit logs
- Clean setup/teardown
- No shared state

**Test structure:**
```python
1. setup() - Create test environment with config
2. test_*() - Run test case
3. teardown() - Clean up temp files
```

**New Tests in v2.0.0:**
- `test_health_check()` - Health endpoint functionality
- `test_path_traversal_security()` - Security validation
- `test_invalid_characters_security()` - Input sanitization
- `test_metrics_collection()` - Metrics tracking

**Test Results:**
- 28 tests, all passing
- Coverage includes all production features
- No external dependencies required

## Extension Points

### Adding New Tools

```python
# 1. Add to self.tools registry
self.tools["new_tool"] = {
    "description": "...",
    "inputSchema": {...}
}

# 2. Implement handler method
def new_tool(self, param1, param2):
    # Implementation
    return result

# 3. Add to handle_tools_call router
elif tool_name == "new_tool":
    result = self.new_tool(args["param1"], args["param2"])
    return self.create_success_response(request_id, result)
```

### Day 2+ Planned Tools

1. **search_notes** - Full-text search across all notes
2. **read_note** - Retrieve note content
3. **list_notes_in_folder** - List files in a category
4. **create_folder** - Add new categories
5. **update_note** - Edit existing notes
6. **delete_note** - Remove notes with confirmation

## Performance Characteristics

### Time Complexity

- `list_note_folders`: O(n) where n = number of folders
- `write_note`: O(1) for file write

### Space Complexity

- Memory: O(1) - streaming I/O
- Disk: O(m) where m = note content size

### Bottlenecks (Current)

1. Synchronous I/O blocks on large files
2. No indexing for future search operations
3. Full directory scan on every list operation

**Acceptable because:**
- Personal use case (< 1000 notes expected)
- Fast local SSD access
- Rare operations (< 10/minute)

## Design Patterns Used

1. **Strategy Pattern**: Tool handlers can be swapped
2. **Template Method**: JSON-RPC handling standardized
3. **Factory Pattern**: Response creation methods
4. **Command Pattern**: Tool calls encapsulate actions

## Production Deployment

### Environment Setup

**Development:**
```bash
export MCP_ENV=development
export LOG_LEVEL=DEBUG
export ENABLE_TRACING=true
python mcp_server.py
```

**Production:**
```bash
export MCP_ENV=production
export NOTES_PATH=/var/lib/notes
export LOG_LEVEL=INFO
export ENABLE_TRACING=true
export ENABLE_AUDIT_LOG=true
export AUDIT_LOG_PATH=/var/log/mcp-audit.log
export MAX_FILE_SIZE_MB=10
python mcp_server.py
```

### Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\Activate.ps1

# Activate (Unix/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run server
python mcp_server.py
```

### Monitoring Integration

The health check endpoint returns JSON-formatted metrics suitable for:
- Prometheus (with a simple exporter wrapper)
- Datadog (custom metrics)
- CloudWatch (via AWS SDK)
- Grafana dashboards

**Example Health Check:**
```bash
# In a separate terminal, send JSON-RPC request
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"get_health_status","arguments":{}}}' | python mcp_server.py
```

### Log Aggregation

**Application Logs** (stderr):
- Structured format with timestamp, level, message, context
- Easily parsed by ELK stack, Splunk, or CloudWatch

**Audit Logs** (file):
- JSON-formatted entries
- Immutable record of all file operations
- Suitable for compliance requirements

## MCP Specification Compliance

### Required Features
✅ stdio transport  
✅ JSON-RPC 2.0 protocol  
✅ initialize method  
✅ tools/list method  
✅ tools/call method  
✅ Tool schemas with descriptions  

### Optional Features (Future)
⬜ resources (file reading)  
⬜ prompts (predefined prompts)  
⬜ completion (autocomplete)  
⬜ sampling (LLM requests)  

## Lessons & Best Practices

### What Worked Well

1. **Comprehensive schemas** - AI understands tool usage immediately
2. **Error messages** - Guide AI to correct behavior
3. **Granular tools** - Single-purpose is easier to reason about
4. **OpenTelemetry integration** - Deep visibility without code clutter
5. **Configuration abstraction** - Easy to test and deploy to different environments
6. **Layered security** - Multiple validation stages catch different attack vectors
7. **Metrics-first design** - Built-in observability from the start

### What Could Improve (Future Work)

1. **Async I/O** - Would handle concurrent requests better
2. **Database Backend** - SQLite for better note management
3. **Caching** - Cache folder listings for performance
4. **Multi-tenancy** - Support multiple users/workspaces

### Key Insights

**Insight 1: Documentation is the UI**
> For AI tools, the schema description is how the model "sees" the tool. Invest time here.

**Insight 2: Errors are feedback**
> Good error messages enable AI to self-correct without human intervention.

**Insight 3: Discovery prevents hardcoding**
> The `list_note_folders` tool is critical - it prevents the AI from making assumptions.

**Insight 4: Start simple, then extend**
> Two tools are enough for Day 1. Build on solid foundations.

**Insight 5: Observability is not optional in production** (v2.0.0)
> Without tracing and metrics, you're flying blind. OpenTelemetry provides the "why" behind failures.

**Insight 6: Security is layers, not a single check** (v2.0.0)
> Path validation, character filtering, size limits, and audit logging work together to prevent attacks.

**Insight 7: Configuration drives flexibility** (v2.0.0)
> Environment variables enable the same code to run in dev, staging, and production with different behaviors.

## References

### Core Specifications
- [MCP Specification](https://modelcontextprotocol.io/)
- [JSON-RPC 2.0](https://www.jsonrpc.org/specification)
- [Python pathlib](https://docs.python.org/3/library/pathlib.html)
- [ReAct: Reasoning and Acting](https://arxiv.org/abs/2210.03629)

### Production Features (v2.0.0)
- [OpenTelemetry Python](https://opentelemetry.io/docs/instrumentation/python/)
- [OpenTelemetry Specification](https://opentelemetry.io/docs/specs/otel/)
- [Python Logging](https://docs.python.org/3/library/logging.html)
- [OWASP Path Traversal](https://owasp.org/www-community/attacks/Path_Traversal)

### Training Materials
- Google "Introduction to Agents" Workshop (Day 1-4)
- "Agent Ops: A Structured Approach to the Unpredictable"
- "Prototype to Production" Guidelines

---

**Architecture version: 2.0.0 (Production Grade)**  
**Previous: 1.0.0 (Day 1 Foundation)**

