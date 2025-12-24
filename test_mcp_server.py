#!/usr/bin/env python3
"""
Test suite for MCP Server (FastMCP-based implementation)
Validates tool behavior, Pydantic validation, and security features

Updated for v3.0.0:
- Tests work with FastMCP SDK
- Validates Pydantic input validation
- Tests Roots security boundaries
- Tests health check tool
"""

import json
import tempfile
import shutil
import os
from pathlib import Path
from pydantic import ValidationError

# Import the necessary components from mcp_server
import mcp_server
from mcp_server import (
    WriteNoteInput,
    SearchNotesInput,
    UpdateMemoryInput,
    GetMemoryInput,
    list_note_folders_impl,
    write_note_impl,
    get_health_status_impl,
    search_notes_impl,
    update_memory_impl,
    get_memory_impl,
    config,
    metrics,
    notes_base_path,
    validate_path_within_root,
    MemoryBank,
    KnowledgeExtractor,
    SEMANTIC_SEARCH_AVAILABLE
)


class TestMCPServer:
    """Test cases for MCP Server functionality"""
    
    def __init__(self):
        self.test_dir = None
        self.original_notes_path = None
        self.test_results = []
    
    def setup(self):
        """Create temporary test environment"""
        # Create temporary directory
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Store original notes path and update config
        self.original_notes_path = str(mcp_server.notes_base_path)
        
        # Update the module-level notes_base_path
        mcp_server.notes_base_path = self.test_dir
        
        # Reset metrics for clean tests
        mcp_server.metrics = mcp_server.Metrics()
        
        # Create test folder structure
        (self.test_dir / "personal").mkdir()
        (self.test_dir / "work").mkdir()
        (self.test_dir / "personal" / "test-note.md").write_text("Test content")
    
    def teardown(self):
        """Clean up test environment"""
        # Restore original notes path
        if self.original_notes_path:
            mcp_server.notes_base_path = Path(self.original_notes_path)
        
        # Clean up test directory
        if self.test_dir and self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def assert_equal(self, actual, expected, test_name):
        """Simple assertion helper"""
        if actual == expected:
            self.test_results.append(f"[PASS] {test_name}")
            return True
        else:
            self.test_results.append(f"[FAIL] {test_name}")
            self.test_results.append(f"   Expected: {expected}")
            self.test_results.append(f"   Actual: {actual}")
            return False
    
    def assert_true(self, condition, test_name):
        """Assert that condition is True"""
        return self.assert_equal(condition, True, test_name)
    
    def assert_contains(self, text, substring, test_name):
        """Assert that text contains substring"""
        contains = substring.lower() in text.lower()
        if contains:
            self.test_results.append(f"[PASS] {test_name}")
            return True
        else:
            self.test_results.append(f"[FAIL] {test_name}")
            self.test_results.append(f"   Expected '{substring}' in: {text}")
            return False
    
    def test_list_note_folders(self):
        """Test list_note_folders discovery tool"""
        result = list_note_folders_impl()
        data = json.loads(result)
        
        self.assert_equal(
            data.get("total_categories"),
            2,
            "Discovers 2 test folders"
        )
        
        folder_names = [f["name"] for f in data.get("folders", [])]
        self.assert_equal(
            set(folder_names),
            {"personal", "work"},
            "Lists correct folder names"
        )
        
        # Check note count
        personal_folder = [f for f in data["folders"] if f["name"] == "personal"][0]
        self.assert_equal(
            personal_folder.get("note_count"),
            1,
            "Counts notes correctly"
        )
    
    def test_write_note_success(self):
        """Test successful note writing"""
        input_data = WriteNoteInput(
            folder_name="work",
            file_name="test-note.md",
            content="# Test Note\n\nThis is a test."
        )
        
        result = write_note_impl(input_data)
        data = json.loads(result)
        
        self.assert_equal(
            data.get("success"),
            True,
            "write_note returns success"
        )
        
        # Verify file was created
        note_path = self.test_dir / "work" / "test-note.md"
        self.assert_true(
            note_path.exists(),
            "Note file is created on disk"
        )
        
        # Verify content
        actual_content = note_path.read_text()
        expected_content = "# Test Note\n\nThis is a test."
        self.assert_equal(
            actual_content,
            expected_content,
            "Note content matches"
        )
    
    def test_write_note_missing_folder(self):
        """Test error handling for non-existent folder"""
        input_data = WriteNoteInput(
            folder_name="nonexistent",
            file_name="test.md",
            content="Content"
        )
        
        try:
            write_note_impl(input_data)
            self.test_results.append("[FAIL] Should raise error for missing folder")
        except ValueError as e:
            error_msg = str(e)
            self.assert_contains(
                error_msg,
                "not found",
                "Returns error for missing folder"
            )
            self.assert_contains(
                error_msg,
                "available folders",
                "Error message lists available folders"
            )
    
    def test_pydantic_validation_empty_folder(self):
        """Test Pydantic validation for empty folder name"""
        try:
            WriteNoteInput(
                folder_name="",
                file_name="test.md",
                content="Content"
            )
            self.test_results.append("[FAIL] Should reject empty folder name")
        except ValidationError as e:
            self.assert_true(
                True,
                "Pydantic rejects empty folder name"
            )
    
    def test_pydantic_validation_empty_file(self):
        """Test Pydantic validation for empty file name"""
        try:
            WriteNoteInput(
                folder_name="work",
                file_name="",
                content="Content"
            )
            self.test_results.append("[FAIL] Should reject empty file name")
        except ValidationError as e:
            self.assert_true(
                True,
                "Pydantic rejects empty file name"
            )
    
    def test_pydantic_validation_empty_content(self):
        """Test Pydantic validation for empty content"""
        try:
            WriteNoteInput(
                folder_name="work",
                file_name="test.md",
                content=""
            )
            self.test_results.append("[FAIL] Should reject empty content")
        except ValidationError as e:
            self.assert_true(
                True,
                "Pydantic rejects empty content"
            )
    
    def test_file_extension_handling(self):
        """Test automatic .md extension addition"""
        input_data = WriteNoteInput(
            folder_name="personal",
            file_name="no-extension",
            content="Test"
        )
        
        write_note_impl(input_data)
        
        # Check that file was created with .md extension
        note_path = self.test_dir / "personal" / "no-extension.md"
        self.assert_true(
            note_path.exists(),
            "Automatically adds .md extension"
        )
    
    def test_health_check(self):
        """Test health check endpoint"""
        result = get_health_status_impl()
        data = json.loads(result)
        
        self.assert_true(
            "status" in data,
            "Health check returns status"
        )
        
        self.assert_true(
            "metrics" in data,
            "Health check includes metrics"
        )
        
        self.assert_equal(
            data.get("version"),
            "3.0.0",
            "Health check shows correct version"
        )
        
        self.assert_true(
            data.get("roots_configured"),
            "Health check confirms Roots are configured"
        )
        
        self.assert_true(
            "security_features" in data,
            "Health check includes security features"
        )
    
    def test_path_traversal_security(self):
        """Test Pydantic validation against path traversal"""
        try:
            WriteNoteInput(
                folder_name="../etc",
                file_name="passwd",
                content="malicious"
            )
            self.test_results.append("[FAIL] Should reject path traversal")
        except ValidationError as e:
            error_msg = str(e)
            self.assert_contains(
                error_msg,
                "traversal",
                "Validation error mentions path traversal"
            )
    
    def test_hidden_directory_security(self):
        """Test Pydantic validation against hidden directories"""
        try:
            WriteNoteInput(
                folder_name=".git",
                file_name="config",
                content="malicious"
            )
            self.test_results.append("[FAIL] Should reject hidden directories")
        except ValidationError as e:
            error_msg = str(e)
            self.assert_contains(
                error_msg,
                "hidden",
                "Validation error mentions hidden directories"
            )
    
    def test_invalid_characters_security(self):
        """Test Pydantic validation against invalid characters"""
        try:
            WriteNoteInput(
                folder_name="folder<>name",
                file_name="test.md",
                content="test"
            )
            self.test_results.append("[FAIL] Should reject invalid characters")
        except ValidationError as e:
            error_msg = str(e)
            self.assert_contains(
                error_msg,
                "invalid characters",
                "Validation error mentions invalid characters"
            )
    
    def test_roots_boundary_validation(self):
        """Test Roots security boundary validation"""
        # Test valid path within root
        valid_path = self.test_dir / "work"
        self.assert_true(
            validate_path_within_root(valid_path, self.test_dir),
            "Validates path within root boundary"
        )
        
        # Test invalid path outside root
        if os.name == 'nt':  # Windows
            invalid_path = Path("C:/Windows/System32")
        else:  # Unix/Linux/Mac
            invalid_path = Path("/etc")
        
        self.assert_equal(
            validate_path_within_root(invalid_path, self.test_dir),
            False,
            "Rejects path outside root boundary"
        )
    
    def test_metrics_collection(self):
        """Test that metrics are being collected"""
        # Make a few calls
        list_note_folders_impl()
        list_note_folders_impl()
        
        # Check metrics
        stats = mcp_server.metrics.get_stats()
        
        self.assert_true(
            "list_note_folders" in stats.get("tool_calls", {}),
            "Metrics track tool calls"
        )
        
        self.assert_true(
            stats["tool_calls"]["list_note_folders"] >= 2,
            "Tool call count is incremented"
        )
    
    def test_file_size_limit(self):
        """Test file size limit validation"""
        # Create content larger than limit (default 10MB)
        large_content = "x" * (11 * 1024 * 1024)  # 11MB
        
        input_data = WriteNoteInput(
            folder_name="work",
            file_name="large.md",
            content=large_content
        )
        
        try:
            write_note_impl(input_data)
            self.test_results.append("[FAIL] Should reject files exceeding size limit")
        except ValueError as e:
            error_msg = str(e)
            self.assert_contains(
                error_msg,
                "exceeds maximum",
                "Error message mentions size limit"
            )
    
    def test_pydantic_field_length_limits(self):
        """Test Pydantic field length validation"""
        # Test folder name too long
        try:
            WriteNoteInput(
                folder_name="x" * 300,  # Exceeds 255 char limit
                file_name="test.md",
                content="Content"
            )
            self.test_results.append("[FAIL] Should reject folder name exceeding length")
        except ValidationError:
            self.assert_true(
                True,
                "Pydantic enforces folder name max length"
            )
    
    def test_write_note_with_unicode(self):
        """Test writing notes with unicode content"""
        input_data = WriteNoteInput(
            folder_name="personal",
            file_name="unicode-test.md",
            content="# Unicode Test\n\n你好世界 🌍 Émojis work!"
        )
        
        result = write_note_impl(input_data)
        data = json.loads(result)
        
        self.assert_true(
            data.get("success"),
            "Handles unicode content"
        )
        
        # Verify content
        note_path = self.test_dir / "personal" / "unicode-test.md"
        actual_content = note_path.read_text(encoding='utf-8')
        self.assert_contains(
            actual_content,
            "你好世界",
            "Unicode characters preserved"
        )
    
    def test_concurrent_folder_listing(self):
        """Test that folder listing handles concurrent structure"""
        # Create additional folders
        (self.test_dir / "projects").mkdir()
        (self.test_dir / "archive").mkdir()
        
        result = list_note_folders_impl()
        data = json.loads(result)
        
        self.assert_equal(
            data.get("total_categories"),
            4,
            "Lists all folders after additions"
        )
    
    # ========== Day 2: Memory Bank Tests ==========
    
    def test_memory_bank_short_term(self):
        """Test short-term memory operations"""
        # Create temporary memory bank
        memory_dir = self.test_dir / "memory"
        memory_dir.mkdir()
        test_memory = MemoryBank(memory_dir)
        
        # Update short-term memory
        test_memory.update_short_term("current_job", "Software Engineer at Google")
        
        # Retrieve short-term memory
        result = test_memory.get_short_term("current_job")
        
        self.assert_equal(
            result.get("value"),
            "Software Engineer at Google",
            "Short-term memory storage and retrieval"
        )
    
    def test_memory_bank_long_term(self):
        """Test long-term memory operations"""
        # Create temporary memory bank
        memory_dir = self.test_dir / "memory"
        memory_dir.mkdir()
        test_memory = MemoryBank(memory_dir)
        
        # Update long-term memory
        test_memory.update_long_term("skills", ["Python", "JavaScript", "AWS"])
        
        # Retrieve long-term memory
        result = test_memory.get_long_term("skills")
        
        self.assert_equal(
            result.get("value"),
            ["Python", "JavaScript", "AWS"],
            "Long-term memory storage and retrieval"
        )
    
    def test_memory_persistence(self):
        """Test that memory persists across MemoryBank instances"""
        # Create temporary memory bank
        memory_dir = self.test_dir / "memory"
        memory_dir.mkdir()
        
        # First instance
        memory1 = MemoryBank(memory_dir)
        memory1.update_long_term("test_key", "test_value")
        
        # Second instance (simulates restart)
        memory2 = MemoryBank(memory_dir)
        result = memory2.get_long_term("test_key")
        
        self.assert_equal(
            result.get("value"),
            "test_value",
            "Memory persistence across instances"
        )
    
    def test_update_memory_tool(self):
        """Test update_memory tool integration"""
        # Create test folder
        test_folder = self.test_dir / "test"
        test_folder.mkdir()
        
        # Update short-term memory
        input_data = UpdateMemoryInput(
            memory_type="short_term",
            key="current_focus",
            value="Job applications"
        )
        result = update_memory_impl(input_data)
        data = json.loads(result)
        
        self.assert_true(
            data.get("success") is True and data.get("key") == "current_focus",
            "Update memory tool execution"
        )
    
    def test_get_memory_tool(self):
        """Test get_memory tool integration"""
        # Create test folder
        test_folder = self.test_dir / "test"
        test_folder.mkdir()
        
        # Update memory first
        update_input = UpdateMemoryInput(
            memory_type="long_term",
            key="career_goal",
            value="Senior Software Engineer"
        )
        update_memory_impl(update_input)
        
        # Get memory
        get_input = GetMemoryInput(
            memory_type="long_term",
            key="career_goal"
        )
        result = get_memory_impl(get_input)
        data = json.loads(result)
        
        self.assert_true(
            data.get("success") is True,
            "Get memory tool execution"
        )
    
    def test_memory_validation(self):
        """Test Pydantic validation for memory tools"""
        try:
            # Invalid memory type
            invalid_input = UpdateMemoryInput(
                memory_type="invalid_type",
                key="test",
                value="test"
            )
            self.assert_true(False, "Should reject invalid memory type")
        except ValidationError:
            self.assert_true(True, "Memory validation - invalid type")
    
    # ========== Day 2: Knowledge Extraction Tests ==========
    
    def test_knowledge_extractor_skills(self):
        """Test skill extraction from notes"""
        content = """
        I've been working with Python and Django for web development.
        Also learning React and PostgreSQL for the frontend and database.
        AWS deployment is part of my skillset too.
        """
        
        extractor = KnowledgeExtractor()
        facts = extractor.extract_key_facts(content)
        skills = facts.get("skills", [])
        
        self.assert_true(
            len(skills) > 0 and "Python" in skills,
            "Knowledge extraction - skills"
        )
    
    def test_knowledge_extractor_preferences(self):
        """Test preference extraction from notes"""
        content = """
        I prefer Python over Java for backend development.
        I really enjoy working with modern frameworks.
        I dislike legacy codebases without tests.
        """
        
        extractor = KnowledgeExtractor()
        facts = extractor.extract_key_facts(content)
        preferences = facts.get("preferences", [])
        
        self.assert_true(
            len(preferences) > 0,
            "Knowledge extraction - preferences"
        )
    
    def test_write_note_with_extraction(self):
        """Test that write_note automatically extracts facts"""
        # Create test folder
        test_folder = self.test_dir / "test"
        test_folder.mkdir()
        
        # Write note with extractable content
        content = "I prefer Python over Java. I have skills in AWS and Docker."
        input_data = WriteNoteInput(
            folder_name="test",
            file_name="test-extraction.md",
            content=content
        )
        
        result = write_note_impl(input_data)
        data = json.loads(result)
        
        extracted_facts = data.get("extracted_facts", {})
        
        self.assert_true(
            "skills_found" in extracted_facts,
            "Write note with automatic extraction"
        )
    
    # ========== Day 2: Semantic Search Tests ==========
    
    def test_search_notes_availability(self):
        """Test semantic search availability check"""
        if not SEMANTIC_SEARCH_AVAILABLE:
            self.assert_true(
                True,
                "Semantic search - dependencies not installed (expected)"
            )
        else:
            # Create test folder
            test_folder = self.test_dir / "test"
            test_folder.mkdir()
            
            input_data = SearchNotesInput(
                query="test query",
                max_results=5
            )
            
            result = search_notes_impl(input_data)
            data = json.loads(result)
            
            self.assert_true(
                "success" in data,
                "Semantic search functionality"
            )
    
    def test_search_notes_validation(self):
        """Test search_notes input validation"""
        try:
            # Query too long
            invalid_input = SearchNotesInput(
                query="x" * 501,  # Exceeds 500 char limit
                max_results=5
            )
            self.assert_true(False, "Should reject query over 500 chars")
        except ValidationError:
            self.assert_true(True, "Search validation - query length limit")
    
    def test_health_check_day2_features(self):
        """Test health check includes Day 2 feature status"""
        result = get_health_status_impl()
        data = json.loads(result)
        
        self.assert_true(
            "day2_features" in data and "semantic_search" in data["day2_features"],
            "Health check includes Day 2 features"
        )
    
    def run_all_tests(self):
        """Run all test cases"""
        print("Running MCP Server Test Suite (FastMCP v4.0.0 - Day 2)\n")
        print("=" * 60)
        
        test_methods = [
            # Day 1 tests
            self.test_list_note_folders,
            self.test_write_note_success,
            self.test_write_note_missing_folder,
            self.test_pydantic_validation_empty_folder,
            self.test_pydantic_validation_empty_file,
            self.test_pydantic_validation_empty_content,
            self.test_file_extension_handling,
            self.test_health_check,
            self.test_path_traversal_security,
            self.test_hidden_directory_security,
            self.test_invalid_characters_security,
            self.test_roots_boundary_validation,
            self.test_metrics_collection,
            self.test_file_size_limit,
            self.test_pydantic_field_length_limits,
            self.test_write_note_with_unicode,
            self.test_concurrent_folder_listing,
            # Day 2 tests - Memory Bank
            self.test_memory_bank_short_term,
            self.test_memory_bank_long_term,
            self.test_memory_persistence,
            self.test_update_memory_tool,
            self.test_get_memory_tool,
            self.test_memory_validation,
            # Day 2 tests - Knowledge Extraction
            self.test_knowledge_extractor_skills,
            self.test_knowledge_extractor_preferences,
            self.test_write_note_with_extraction,
            # Day 2 tests - Semantic Search
            self.test_search_notes_availability,
            self.test_search_notes_validation,
            self.test_health_check_day2_features
        ]
        
        for test_method in test_methods:
            try:
                self.setup()
                test_method()
                self.teardown()
            except Exception as e:
                self.test_results.append(f"[EXCEPTION] in {test_method.__name__}: {str(e)}")
                import traceback
                self.test_results.append(f"   {traceback.format_exc()}")
        
        # Print results
        print("\n" + "=" * 60)
        print("Test Results:\n")
        for result in self.test_results:
            print(result)
        
        # Summary
        passed = sum(1 for r in self.test_results if r.startswith("[PASS]"))
        failed = sum(1 for r in self.test_results if r.startswith("[FAIL]"))
        exceptions = sum(1 for r in self.test_results if r.startswith("[EXCEPTION]"))
        
        print("\n" + "=" * 60)
        print(f"Summary: {passed} passed, {failed} failed, {exceptions} exceptions")
        print("=" * 60)
        
        return failed == 0 and exceptions == 0


def main():
    """Run test suite"""
    tester = TestMCPServer()
    success = tester.run_all_tests()
    
    if success:
        print("\nAll tests passed!")
        exit(0)
    else:
        print("\nSome tests failed")
        exit(1)


if __name__ == "__main__":
    main()
