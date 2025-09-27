# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start Commands

```bash
# Setup and installation
uv sync                    # Install all dependencies
cp .env.example .env       # Copy environment template (add ANTHROPIC_API_KEY)

# Run the application
./run.sh                   # Quick start script
# OR manually:
cd backend && uv run uvicorn app:app --reload --port 8000

# Development commands (always use uv)
uv run python main.py      # Run any Python script
uv add package_name        # Add new dependencies
uv remove package_name     # Remove dependencies

# Access points
# Web Interface: http://localhost:8000
# API Documentation: http://localhost:8000/docs
```

## Architecture Overview

This is a **Retrieval-Augmented Generation (RAG) system** for course materials using a modular, tool-based architecture:

### Core Pipeline Flow
1. **Document Processing** → **Vector Storage** → **Tool-Based Search** → **AI Generation** → **Response**
2. **Session Management** maintains conversation context across queries

### Key Components & Interactions

**RAGSystem** (`rag_system.py`) - Main orchestrator that coordinates:
- **DocumentProcessor**: Parses structured course files with specific format requirements
- **VectorStore**: Dual ChromaDB collections (course metadata + content chunks)
- **AIGenerator**: Claude integration with function calling capabilities
- **ToolManager**: Extensible search tool framework
- **SessionManager**: Conversation history with configurable max length

**Vector Storage Design:**
- **Course Catalog Collection**: Stores course metadata for semantic course name matching
- **Course Content Collection**: Stores text chunks with lesson context for retrieval
- Uses sentence-transformer embeddings with persistent ChromaDB storage

**Tool-Based Search Pattern:**
- AI uses `search_course_content` tool via Claude's function calling
- Tool can filter by course name and returns formatted content with sources
- Sources tracked separately from search results for frontend display

## Document Format Requirements

Course documents in `/docs/` must follow this structure:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [instructor]

Lesson 0: [title]
Lesson Link: [url]
[lesson content...]

Lesson 1: [next title]
[content...]
```

**Processing Behavior:**
- Text chunked at sentence boundaries with configurable overlap (default: 800 chars, 100 overlap)
- Chunks prefixed with lesson context: `"Course [title] Lesson X content: [text]"`
- Deduplication by course title prevents reprocessing existing courses

## Configuration & Environment

**Required Environment Variables:**
- `ANTHROPIC_API_KEY`: Claude API access (separate from Claude Code Pro subscription)

**Key Config Settings** (`config.py`):
- `CHUNK_SIZE = 800` / `CHUNK_OVERLAP = 100`: Text processing parameters
- `MAX_RESULTS = 5`: Vector search result limit
- `MAX_HISTORY = 10`: Session conversation length
- `ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"`: AI model version

## API & Frontend Integration

**Backend** (`app.py`): FastAPI with CORS enabled
- `POST /api/query`: Main chat endpoint with session management
- `GET /api/courses`: Course statistics and catalog
- Startup automatically loads documents from `/docs/` folder

**Frontend** (`frontend/`): Vanilla JavaScript with:
- Real-time chat interface with markdown rendering
- Session persistence and loading indicators
- Course statistics display with collapsible sources

## Key Development Patterns

**Error Handling:** Comprehensive API error catching in `ai_generator.py` with user-friendly messages for common issues (API limits, authentication, credits)

**Search Strategy:**
- Course name resolution using vector similarity before content search
- Smart keyword detection to trigger course-specific vs general knowledge responses
- One search per query maximum to optimize performance

**Session Management:**
- Auto-generated session IDs for new conversations
- Conversation history formatted as context for AI prompts
- History trimmed to configured maximum length

**File Processing:**
- `add_course_folder()` supports selective file processing via `selected_files` parameter
- Graceful handling of malformed documents with fallback to filename as course title
- Support for PDF, DOCX, and TXT file formats