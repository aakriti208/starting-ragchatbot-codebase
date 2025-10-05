"""
Shared test fixtures and configuration for RAG system tests
"""

import shutil
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import Mock, patch

import pytest
from models import Course, CourseChunk, Lesson
from search_tools import CourseOutlineTool, CourseSearchTool, ToolManager
from vector_store import VectorStore
from config import Config
from rag_system import RAGSystem


@pytest.fixture
def temp_chroma_dir():
    """Create a temporary directory for ChromaDB and clean up after test"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_course():
    """Create a sample course for testing"""
    return Course(
        title="Introduction to MCP",
        course_link="https://example.com/mcp-course",
        instructor="John Doe",
        lessons=[
            Lesson(
                lesson_number=0,
                title="Getting Started",
                lesson_link="https://example.com/lesson0",
            ),
            Lesson(
                lesson_number=1,
                title="Basic Concepts",
                lesson_link="https://example.com/lesson1",
            ),
            Lesson(
                lesson_number=2,
                title="Advanced Topics",
                lesson_link="https://example.com/lesson2",
            ),
        ],
    )


@pytest.fixture
def sample_course_chunks(sample_course):
    """Create sample course chunks for testing"""
    return [
        CourseChunk(
            content="MCP stands for Model Context Protocol. It's a standardized way to connect AI models to external data sources.",
            course_title=sample_course.title,
            lesson_number=0,
            chunk_index=0,
        ),
        CourseChunk(
            content="The Model Context Protocol enables AI assistants to securely access data from various sources like databases, APIs, and file systems.",
            course_title=sample_course.title,
            lesson_number=0,
            chunk_index=1,
        ),
        CourseChunk(
            content="MCP servers provide tools and resources that AI models can use. A tool is a function that the model can call to perform actions.",
            course_title=sample_course.title,
            lesson_number=1,
            chunk_index=2,
        ),
        CourseChunk(
            content="Resources in MCP represent data that the model can read. They are similar to files or database records.",
            course_title=sample_course.title,
            lesson_number=1,
            chunk_index=3,
        ),
        CourseChunk(
            content="Advanced MCP features include streaming, pagination, and error handling for robust integrations.",
            course_title=sample_course.title,
            lesson_number=2,
            chunk_index=4,
        ),
    ]


@pytest.fixture
def second_sample_course():
    """Create a second sample course for testing multiple courses"""
    return Course(
        title="Building with Anthropic Claude",
        course_link="https://example.com/claude-course",
        instructor="Jane Smith",
        lessons=[
            Lesson(
                lesson_number=0,
                title="Introduction to Claude",
                lesson_link="https://example.com/claude-lesson0",
            ),
            Lesson(
                lesson_number=1,
                title="Prompt Engineering",
                lesson_link="https://example.com/claude-lesson1",
            ),
        ],
    )


@pytest.fixture
def second_course_chunks(second_sample_course):
    """Create chunks for second course"""
    return [
        CourseChunk(
            content="Claude is Anthropic's AI assistant. It's designed to be helpful, harmless, and honest.",
            course_title=second_sample_course.title,
            lesson_number=0,
            chunk_index=0,
        ),
        CourseChunk(
            content="Effective prompt engineering with Claude involves clear instructions, examples, and structured thinking.",
            course_title=second_sample_course.title,
            lesson_number=1,
            chunk_index=1,
        ),
    ]


@pytest.fixture
def vector_store(
    temp_chroma_dir,
    sample_course,
    sample_course_chunks,
    second_sample_course,
    second_course_chunks,
):
    """Create a vector store with sample data"""
    store = VectorStore(
        chroma_path=temp_chroma_dir, embedding_model="all-MiniLM-L6-v2", max_results=5
    )

    # Add first course
    store.add_course_metadata(sample_course)
    store.add_course_content(sample_course_chunks)

    # Add second course
    store.add_course_metadata(second_sample_course)
    store.add_course_content(second_course_chunks)

    return store


@pytest.fixture
def search_tool(vector_store):
    """Create a CourseSearchTool with populated vector store"""
    return CourseSearchTool(vector_store)


@pytest.fixture
def outline_tool(vector_store):
    """Create a CourseOutlineTool with populated vector store"""
    return CourseOutlineTool(vector_store)


@pytest.fixture
def tool_manager(search_tool, outline_tool):
    """Create a ToolManager with registered tools"""
    manager = ToolManager()
    manager.register_tool(search_tool)
    manager.register_tool(outline_tool)
    return manager


@pytest.fixture
def mock_anthropic_response():
    """Mock response from Anthropic API"""

    class MockContent:
        def __init__(self, text=None, tool_name=None, tool_input=None, tool_id=None):
            self.text = text
            self.type = "text" if text else "tool_use"
            self.name = tool_name
            self.input = tool_input or {}
            self.id = tool_id or "tool_123"

    class MockResponse:
        def __init__(self, text=None, stop_reason="end_turn", tool_use=None):
            if tool_use:
                self.content = [
                    MockContent(
                        tool_name=tool_use["name"],
                        tool_input=tool_use["input"],
                        tool_id=tool_use.get("id", "tool_123"),
                    )
                ]
                self.stop_reason = "tool_use"
            else:
                self.content = [MockContent(text=text)]
                self.stop_reason = stop_reason

    return MockResponse


@pytest.fixture
def test_config():
    """Create a test configuration with temporary directories"""
    config = Config()
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.MAX_RESULTS = 5
    config.MAX_HISTORY = 10
    config.CHROMA_PATH = tempfile.mkdtemp()
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.LLM_PROVIDER = "ollama"
    config.OLLAMA_BASE_URL = "http://localhost:11434"
    config.OLLAMA_MODEL = "llama3.2"

    yield config

    # Cleanup
    shutil.rmtree(config.CHROMA_PATH, ignore_errors=True)


@pytest.fixture
def mock_rag_system(test_config, sample_course, sample_course_chunks):
    """Create a mocked RAG system for API testing"""
    with patch('openai.OpenAI') as mock_openai:
        # Mock the OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response from RAG system"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        system = RAGSystem(test_config)

        # Add test data
        system.vector_store.add_course_metadata(sample_course)
        system.vector_store.add_course_content(sample_course_chunks)

        yield system
