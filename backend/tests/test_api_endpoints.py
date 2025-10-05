"""
API endpoint tests for the FastAPI application

These tests use a test app factory to avoid static file mounting issues.
Tests cover /api/query, /api/courses endpoints with various scenarios.
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import tempfile
import shutil

from config import Config
from rag_system import RAGSystem


@pytest.fixture
def test_app(mock_rag_system):
    """
    Create a test FastAPI app without static file mounting

    This avoids the static file mount issue by creating a minimal
    app with only the API endpoints for testing.
    """
    app = FastAPI(title="Course Materials RAG System (Test)")

    # Store RAG system in app state
    app.state.rag_system = mock_rag_system

    # Import models
    from pydantic import BaseModel
    from typing import List, Optional
    from fastapi import HTTPException

    class QueryRequest(BaseModel):
        """Request model for course queries"""
        query: str
        session_id: Optional[str] = None

    class QueryResponse(BaseModel):
        """Response model for course queries"""
        answer: str
        sources: List[str]
        session_id: str

    class CourseStats(BaseModel):
        """Response model for course statistics"""
        total_courses: int
        course_titles: List[str]

    # Define API endpoints inline
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        """Process a query and return response with sources"""
        try:
            rag_system = app.state.rag_system
            # Create session if not provided
            session_id = request.session_id
            if not session_id:
                session_id = rag_system.session_manager.create_session()

            # Process query using RAG system
            answer, sources = rag_system.query(request.query, session_id)

            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        """Get course analytics and statistics"""
        try:
            rag_system = app.state.rag_system
            analytics = rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


@pytest.fixture
def client(test_app):
    """Create a test client for the FastAPI app"""
    return TestClient(test_app)


@pytest.mark.api
class TestQueryEndpoint:
    """Test the /api/query endpoint"""

    def test_query_without_session_id(self, client):
        """Test querying without providing a session ID"""
        response = client.post(
            "/api/query",
            json={"query": "What is MCP?"}
        )

        assert response.status_code == 200
        data = response.json()

        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert len(data["session_id"]) > 0

    def test_query_with_session_id(self, client):
        """Test querying with an existing session ID"""
        # First query to get session ID
        response1 = client.post(
            "/api/query",
            json={"query": "What is MCP?"}
        )
        session_id = response1.json()["session_id"]

        # Second query with same session
        response2 = client.post(
            "/api/query",
            json={
                "query": "Tell me more",
                "session_id": session_id
            }
        )

        assert response2.status_code == 200
        data = response2.json()

        assert data["session_id"] == session_id
        assert isinstance(data["answer"], str)

    def test_query_empty_string(self, client):
        """Test querying with an empty string"""
        response = client.post(
            "/api/query",
            json={"query": ""}
        )

        # Should still return 200 but might have specific answer
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data

    def test_query_invalid_request_missing_query(self, client):
        """Test request without required query field"""
        response = client.post(
            "/api/query",
            json={"session_id": "test-123"}
        )

        # Should return 422 Unprocessable Entity for validation error
        assert response.status_code == 422

    def test_query_invalid_json(self, client):
        """Test request with invalid JSON"""
        response = client.post(
            "/api/query",
            data="not valid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    def test_query_response_structure(self, client):
        """Test that response has correct structure"""
        response = client.post(
            "/api/query",
            json={"query": "What is MCP?"}
        )

        assert response.status_code == 200
        data = response.json()

        # Verify all required fields
        assert set(data.keys()) == {"answer", "sources", "session_id"}

        # Verify types
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)

        # Verify sources structure if present
        for source in data["sources"]:
            assert isinstance(source, str)

    def test_query_conversation_history(self, client):
        """Test that conversation history is maintained"""
        # First query
        response1 = client.post(
            "/api/query",
            json={"query": "What is MCP?"}
        )
        session_id = response1.json()["session_id"]

        # Multiple follow-up queries
        queries = ["Tell me more", "What are the benefits?", "How do I use it?"]
        for query in queries:
            response = client.post(
                "/api/query",
                json={
                    "query": query,
                    "session_id": session_id
                }
            )
            assert response.status_code == 200
            assert response.json()["session_id"] == session_id


@pytest.mark.api
class TestCoursesEndpoint:
    """Test the /api/courses endpoint"""

    def test_get_course_stats(self, client):
        """Test getting course statistics"""
        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()

        assert "total_courses" in data
        assert "course_titles" in data
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)

    def test_get_course_stats_structure(self, client):
        """Test course stats response structure"""
        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()

        # Verify exact fields
        assert set(data.keys()) == {"total_courses", "course_titles"}

        # Verify course_titles contains strings
        for title in data["course_titles"]:
            assert isinstance(title, str)

    def test_get_course_stats_with_courses(self, client):
        """Test that added courses appear in stats"""
        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()

        # Should have at least one course from fixtures
        assert data["total_courses"] >= 1
        assert len(data["course_titles"]) >= 1

        # Check that sample course is present
        assert "Introduction to MCP" in data["course_titles"]

    def test_get_course_stats_count_matches_titles(self, client):
        """Test that total_courses matches length of course_titles"""
        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()

        assert data["total_courses"] == len(data["course_titles"])


@pytest.mark.api
class TestAPIErrorHandling:
    """Test error handling in API endpoints"""

    def test_query_with_rag_system_error(self, client, test_app):
        """Test handling when RAG system raises an error"""
        # Mock query to raise exception
        original_query = test_app.state.rag_system.query
        test_app.state.rag_system.query = Mock(side_effect=Exception("Database connection failed"))

        response = client.post(
            "/api/query",
            json={"query": "What is MCP?"}
        )

        assert response.status_code == 500
        assert "Database connection failed" in response.json()["detail"]

        # Restore
        test_app.state.rag_system.query = original_query

    def test_courses_with_rag_system_error(self, client, test_app):
        """Test handling when getting course analytics fails"""
        # Mock get_course_analytics to raise exception
        original_analytics = test_app.state.rag_system.get_course_analytics
        test_app.state.rag_system.get_course_analytics = Mock(
            side_effect=Exception("Vector store error")
        )

        response = client.get("/api/courses")

        assert response.status_code == 500
        assert "Vector store error" in response.json()["detail"]

        # Restore
        test_app.state.rag_system.get_course_analytics = original_analytics


@pytest.mark.api
class TestAPIIntegration:
    """Integration tests combining multiple endpoints"""

    def test_query_then_check_stats(self, client):
        """Test querying and then checking course stats"""
        # Make a query
        query_response = client.post(
            "/api/query",
            json={"query": "What is MCP?"}
        )
        assert query_response.status_code == 200

        # Check stats
        stats_response = client.get("/api/courses")
        assert stats_response.status_code == 200

        # Stats should show courses
        stats = stats_response.json()
        assert stats["total_courses"] >= 1

    def test_multiple_concurrent_sessions(self, client):
        """Test handling multiple concurrent sessions"""
        # Create multiple sessions
        sessions = []
        for i in range(3):
            response = client.post(
                "/api/query",
                json={"query": f"Question {i}"}
            )
            assert response.status_code == 200
            sessions.append(response.json()["session_id"])

        # Verify all sessions are unique
        assert len(set(sessions)) == 3

        # Continue conversations in each session
        for session_id in sessions:
            response = client.post(
                "/api/query",
                json={
                    "query": "Follow-up question",
                    "session_id": session_id
                }
            )
            assert response.status_code == 200
            assert response.json()["session_id"] == session_id
