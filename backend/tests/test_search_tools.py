"""
Tests for CourseSearchTool and CourseOutlineTool to verify search functionality
"""
import pytest
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager


class TestCourseSearchTool:
    """Test CourseSearchTool execute method with various parameters"""

    def test_search_tool_definition(self, search_tool):
        """Test that tool definition is correctly formatted"""
        tool_def = search_tool.get_tool_definition()

        assert tool_def["name"] == "search_course_content"
        assert "description" in tool_def
        assert "input_schema" in tool_def
        assert tool_def["input_schema"]["required"] == ["query"]

        # Check properties exist
        props = tool_def["input_schema"]["properties"]
        assert "query" in props
        assert "course_name" in props
        assert "lesson_number" in props

    def test_execute_with_query_only(self, search_tool):
        """Test search with only query parameter (no filters)"""
        result = search_tool.execute(query="What is MCP?")

        assert isinstance(result, str)
        assert len(result) > 0
        # Should find content about MCP
        assert "MCP" in result or "Model Context Protocol" in result

    def test_execute_with_course_filter(self, search_tool):
        """Test search with course_name filter"""
        result = search_tool.execute(
            query="protocol",
            course_name="Introduction to MCP"
        )

        assert isinstance(result, str)
        assert "Introduction to MCP" in result
        # Should not include content from other courses
        assert "Building with Anthropic Claude" not in result

    def test_execute_with_partial_course_name(self, search_tool):
        """Test that partial course names work via semantic matching"""
        result = search_tool.execute(
            query="What are tools?",
            course_name="MCP"  # Partial match for "Introduction to MCP"
        )

        assert isinstance(result, str)
        assert len(result) > 0
        # Should resolve to the full course name
        assert "Introduction to MCP" in result

    def test_execute_with_lesson_filter(self, search_tool):
        """Test search with lesson_number filter"""
        result = search_tool.execute(
            query="tools and resources",
            course_name="Introduction to MCP",
            lesson_number=1
        )

        assert isinstance(result, str)
        # Should include lesson number in results
        assert "Lesson 1" in result

    def test_execute_with_nonexistent_course(self, search_tool):
        """Test search with non-existent course name"""
        result = search_tool.execute(
            query="test",
            course_name="Nonexistent Course"
        )

        assert isinstance(result, str)
        # Should return error message about course not found
        assert "No" in result and "found" in result.lower()

    def test_execute_tracks_sources(self, search_tool):
        """Test that execute method tracks sources correctly"""
        # Reset sources first
        search_tool.last_sources = []

        result = search_tool.execute(
            query="What is MCP?",
            course_name="Introduction to MCP"
        )

        # Check that sources were tracked
        assert len(search_tool.last_sources) > 0
        # Sources should contain course title
        assert any("Introduction to MCP" in source for source in search_tool.last_sources)

    def test_execute_empty_results(self, search_tool):
        """Test behavior when no results are found"""
        result = search_tool.execute(
            query="quantum physics advanced equations",  # Unlikely to match course content
            course_name="Introduction to MCP",
            lesson_number=0
        )

        assert isinstance(result, str)
        # Should indicate no results found
        assert "No relevant content found" in result


class TestCourseOutlineTool:
    """Test CourseOutlineTool for getting course structure"""

    def test_outline_tool_definition(self, outline_tool):
        """Test that outline tool definition is correctly formatted"""
        tool_def = outline_tool.get_tool_definition()

        assert tool_def["name"] == "get_course_outline"
        assert "description" in tool_def
        assert "input_schema" in tool_def
        assert tool_def["input_schema"]["required"] == ["course_name"]

    def test_get_outline_full_course_name(self, outline_tool):
        """Test getting outline with full course name"""
        result = outline_tool.execute(course_name="Introduction to MCP")

        assert isinstance(result, str)
        assert "Introduction to MCP" in result
        assert "John Doe" in result  # Instructor
        assert "Lesson 0: Getting Started" in result
        assert "Lesson 1: Basic Concepts" in result
        assert "Lesson 2: Advanced Topics" in result

    def test_get_outline_partial_course_name(self, outline_tool):
        """Test getting outline with partial course name"""
        result = outline_tool.execute(course_name="MCP")

        assert isinstance(result, str)
        assert "Introduction to MCP" in result
        assert "Total Lessons: 3" in result

    def test_get_outline_nonexistent_course(self, outline_tool):
        """Test getting outline for non-existent course"""
        result = outline_tool.execute(course_name="Nonexistent Course XYZ")

        assert isinstance(result, str)
        assert "No course found" in result

    def test_get_outline_includes_all_metadata(self, outline_tool):
        """Test that outline includes all course metadata"""
        result = outline_tool.execute(course_name="Building with Anthropic Claude")

        assert "Building with Anthropic Claude" in result
        assert "Jane Smith" in result  # Instructor
        assert "https://example.com/claude-course" in result  # Course link
        assert "Total Lessons: 2" in result


class TestToolManager:
    """Test ToolManager for managing and executing tools"""

    def test_register_tool(self, search_tool):
        """Test registering a tool"""
        manager = ToolManager()
        manager.register_tool(search_tool)

        assert "search_course_content" in manager.tools

    def test_get_tool_definitions(self, tool_manager):
        """Test getting all tool definitions"""
        definitions = tool_manager.get_tool_definitions()

        assert len(definitions) == 2  # search and outline tools
        tool_names = [d["name"] for d in definitions]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names

    def test_execute_tool_by_name(self, tool_manager):
        """Test executing a tool by name"""
        result = tool_manager.execute_tool(
            "search_course_content",
            query="What is MCP?"
        )

        assert isinstance(result, str)
        assert len(result) > 0

    def test_execute_nonexistent_tool(self, tool_manager):
        """Test executing a non-existent tool"""
        result = tool_manager.execute_tool(
            "nonexistent_tool",
            query="test"
        )

        assert "not found" in result.lower()

    def test_get_last_sources(self, tool_manager):
        """Test retrieving sources from last search"""
        # Execute a search
        tool_manager.execute_tool(
            "search_course_content",
            query="MCP protocol",
            course_name="Introduction to MCP"
        )

        sources = tool_manager.get_last_sources()
        assert len(sources) > 0
        assert any("Introduction to MCP" in source for source in sources)

    def test_reset_sources(self, tool_manager):
        """Test resetting sources"""
        # Execute a search to generate sources
        tool_manager.execute_tool(
            "search_course_content",
            query="MCP"
        )

        # Verify sources exist
        assert len(tool_manager.get_last_sources()) > 0

        # Reset sources
        tool_manager.reset_sources()

        # Verify sources are cleared
        assert len(tool_manager.get_last_sources()) == 0


class TestSearchToolIntegration:
    """Integration tests for search tools with vector store"""

    def test_multiple_course_search(self, search_tool):
        """Test that search without course filter searches all courses"""
        result = search_tool.execute(query="AI assistant")

        # Could match content from either course
        assert len(result) > 0

    def test_search_respects_max_results(self, vector_store):
        """Test that search respects max_results limit"""
        # Create tool with max_results=2
        tool = CourseSearchTool(vector_store)

        result = tool.execute(query="MCP")

        # Count number of result blocks (separated by double newlines)
        result_blocks = [b for b in result.split("\n\n") if b.strip()]
        # Should not exceed max_results (5 by default in fixture)
        assert len(result_blocks) <= 5
