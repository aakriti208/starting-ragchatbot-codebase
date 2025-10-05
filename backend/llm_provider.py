from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """Generate response from the LLM"""
        pass


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider implementation"""

    def __init__(self, api_key: str, model: str):
        import anthropic
        from anthropic import APIError, AuthenticationError, RateLimitError

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}
        self.APIError = APIError
        self.AuthenticationError = AuthenticationError
        self.RateLimitError = RateLimitError

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:

        system_prompt = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- **One search per query maximum**
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

        system_content = (
            f"{system_prompt}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else system_prompt
        )

        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content,
        }

        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        try:
            response = self.client.messages.create(**api_params)

            if response.stop_reason == "tool_use" and tool_manager:
                return self._handle_tool_execution(response, api_params, tool_manager)

            return response.content[0].text

        except self.AuthenticationError:
            return "Error: Invalid Anthropic API key. Please check your API key configuration."
        except self.RateLimitError:
            return "Error: API rate limit exceeded. Please try again in a moment."
        except self.APIError as e:
            if "credit balance" in str(e).lower():
                return "Error: Insufficient Anthropic API credits. Please add credits to your Anthropic account to continue using the service."
            return f"Error: API request failed - {str(e)}"
        except Exception as e:
            return f"Error: An unexpected error occurred - {str(e)}"

    def _handle_tool_execution(
        self, initial_response, base_params: Dict[str, Any], tool_manager
    ):
        messages = base_params["messages"].copy()
        messages.append({"role": "assistant", "content": initial_response.content})

        tool_results = []
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(
                    content_block.name, **content_block.input
                )

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result,
                    }
                )

        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"],
        }

        try:
            final_response = self.client.messages.create(**final_params)
            return final_response.content[0].text
        except self.AuthenticationError:
            return "Error: Invalid Anthropic API key. Please check your API key configuration."
        except self.RateLimitError:
            return "Error: API rate limit exceeded. Please try again in a moment."
        except self.APIError as e:
            if "credit balance" in str(e).lower():
                return "Error: Insufficient Anthropic API credits. Please add credits to your Anthropic account to continue using the service."
            return f"Error: API request failed - {str(e)}"
        except Exception as e:
            return f"Error: An unexpected error occurred - {str(e)}"


class OllamaProvider(LLMProvider):
    """Local Ollama provider implementation using OpenAI-compatible API"""

    def __init__(
        self, base_url: str = "http://localhost:11434", model: str = "llama3.2"
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package required for Ollama provider. Install with: uv add openai"
            )

        self.client = OpenAI(
            base_url=f"{base_url}/v1",
            api_key="ollama",  # Ollama doesn't require real API key
        )
        self.model = model

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:

        # Check if we should search course content based on query keywords
        should_search = self._should_search_courses(query)
        search_results = ""

        # Perform search if needed and tools are available
        if should_search and tool_manager:
            try:
                # Extract parameters from the query
                search_params = self._extract_search_parameters(query)

                # Execute search with extracted parameters
                search_results = tool_manager.execute_tool(
                    "search_course_content", **search_params
                )
                if search_results and search_results.strip():
                    search_context = (
                        f"\n\nRelevant course content found:\n{search_results}"
                    )
                else:
                    search_context = (
                        "\n\nNo relevant course content found for this query."
                    )
            except Exception as e:
                search_context = f"\n\nError searching course content: {str(e)}"
        else:
            search_context = ""

        system_prompt = f""" You are an AI assistant specialized in course materials and educational content.

Search Tool Usage:
- When users ask about specific course content, programming concepts, or technical topics that might be covered in educational materials, search the course database first
- Synthesize search results into accurate, fact-based responses
- If search yields no results, provide general knowledge answers

Response Protocol:
- **Course-specific questions**: Use search results when available, supplement with general knowledge
- **General knowledge questions**: Answer using existing knowledge
- **No meta-commentary**: Provide direct answers without mentioning search process

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.{search_context}
"""

        messages = [{"role": "system", "content": system_prompt}]

        if conversation_history:
            messages.append(
                {
                    "role": "assistant",
                    "content": f"Previous conversation:\n{conversation_history}",
                }
            )

        messages.append({"role": "user", "content": query})

        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=0, max_tokens=800
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"Error: Failed to connect to Ollama - {str(e)}. Make sure Ollama is running and the model '{self.model}' is installed."

    def _should_search_courses(self, query: str) -> bool:
        """Determine if the query likely relates to course content"""
        import re

        query_lower = query.lower()

        # Strong indicators - always search
        strong_patterns = [
            r"\bcourse\b",
            r"\blesson\b",
            r"\btutorial\b",
            r"\bmodule\b",
            r"\bMCP\b",
            r"\bClaude\b",
            r"\bAnthropic\b",
        ]

        if any(re.search(pattern, query, re.IGNORECASE) for pattern in strong_patterns):
            return True

        # Educational question patterns
        educational_patterns = [
            "how to",
            "what is",
            "what are",
            "explain",
            "learn",
            "teach",
            "show me",
            "tell me about",
            "describe",
        ]

        # Technical keywords
        tech_keywords = [
            "api",
            "programming",
            "code",
            "software",
            "development",
            "python",
            "javascript",
            "database",
            "ml",
            "ai",
        ]

        # Search if educational pattern + tech keyword
        has_educational = any(
            pattern in query_lower for pattern in educational_patterns
        )
        has_tech = any(keyword in query_lower for keyword in tech_keywords)

        return has_educational and has_tech

    def _extract_search_parameters(self, query: str) -> Dict[str, Any]:
        """Extract search parameters from the query using regex patterns"""
        import re

        params = {"query": query}

        # Extract lesson number using regex
        # Patterns: "lesson 1", "lesson 2", "in lesson 3", etc.
        lesson_patterns = [
            r"\blesson\s+(\d+)\b",
            r"\bin\s+lesson\s+(\d+)\b",
            r"\bof\s+lesson\s+(\d+)\b",
        ]

        for pattern in lesson_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                params["lesson_number"] = int(match.group(1))
                break

        # Extract course name using pattern matching
        # Patterns: "in the X course", "from X course", "about X course", etc.
        course_patterns = [
            r"in\s+(?:the\s+)?([^?]+?)\s+course",
            r"from\s+(?:the\s+)?([^?]+?)\s+course",
            r"about\s+(?:the\s+)?([^?]+?)\s+course",
            r"of\s+(?:the\s+)?([^?]+?)\s+course",
        ]

        for pattern in course_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                course_name = match.group(1).strip()
                # Clean up common words
                course_name = re.sub(
                    r"\s+(is|are|was|were)\s*$", "", course_name, flags=re.IGNORECASE
                )
                params["course_name"] = course_name
                break

        # If no explicit course pattern, check for course names directly
        if "course_name" not in params:
            # Check for common course identifiers
            course_identifiers = [
                r"\bMCP\b",
                r"\bIntroduction\s+to\s+MCP\b",
                r"\bAnthropic\b",
                r"\bClaude\b",
                r"\bBuilding\s+with\s+\w+\b",
            ]

            for pattern in course_identifiers:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    params["course_name"] = match.group(0)
                    break

        return params


class LocalAIProvider(LLMProvider):
    """LocalAI provider implementation"""

    def __init__(
        self, base_url: str = "http://localhost:8080", model: str = "gpt-3.5-turbo"
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package required for LocalAI provider. Install with: uv add openai"
            )

        self.client = OpenAI(
            base_url=f"{base_url}/v1",
            api_key="local",  # LocalAI doesn't require real API key
        )
        self.model = model

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:

        # Check if we should search course content based on query keywords
        should_search = self._should_search_courses(query)
        search_results = ""

        # Perform search if needed and tools are available
        if should_search and tool_manager:
            try:
                # Extract parameters from the query
                search_params = self._extract_search_parameters(query)

                # Execute search with extracted parameters
                search_results = tool_manager.execute_tool(
                    "search_course_content", **search_params
                )
                if search_results and search_results.strip():
                    search_context = (
                        f"\n\nRelevant course content found:\n{search_results}"
                    )
                else:
                    search_context = (
                        "\n\nNo relevant course content found for this query."
                    )
            except Exception as e:
                search_context = f"\n\nError searching course content: {str(e)}"
        else:
            search_context = ""

        system_prompt = f""" You are an AI assistant specialized in course materials and educational content.

Search Tool Usage:
- When users ask about specific course content, programming concepts, or technical topics that might be covered in educational materials, search the course database first
- Synthesize search results into accurate, fact-based responses
- If search yields no results, provide general knowledge answers

Response Protocol:
- **Course-specific questions**: Use search results when available, supplement with general knowledge
- **General knowledge questions**: Answer using existing knowledge
- **No meta-commentary**: Provide direct answers without mentioning search process

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.{search_context}
"""

        messages = [{"role": "system", "content": system_prompt}]

        if conversation_history:
            messages.append(
                {
                    "role": "assistant",
                    "content": f"Previous conversation:\n{conversation_history}",
                }
            )

        messages.append({"role": "user", "content": query})

        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=0, max_tokens=800
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"Error: Failed to connect to LocalAI - {str(e)}. Make sure LocalAI is running at {self.client.base_url}."

    def _should_search_courses(self, query: str) -> bool:
        """Determine if the query likely relates to course content"""
        import re

        query_lower = query.lower()

        # Strong indicators - always search
        strong_patterns = [
            r"\bcourse\b",
            r"\blesson\b",
            r"\btutorial\b",
            r"\bmodule\b",
            r"\bMCP\b",
            r"\bClaude\b",
            r"\bAnthropic\b",
        ]

        if any(re.search(pattern, query, re.IGNORECASE) for pattern in strong_patterns):
            return True

        # Educational question patterns
        educational_patterns = [
            "how to",
            "what is",
            "what are",
            "explain",
            "learn",
            "teach",
            "show me",
            "tell me about",
            "describe",
        ]

        # Technical keywords
        tech_keywords = [
            "api",
            "programming",
            "code",
            "software",
            "development",
            "python",
            "javascript",
            "database",
            "ml",
            "ai",
        ]

        # Search if educational pattern + tech keyword
        has_educational = any(
            pattern in query_lower for pattern in educational_patterns
        )
        has_tech = any(keyword in query_lower for keyword in tech_keywords)

        return has_educational and has_tech

    def _extract_search_parameters(self, query: str) -> Dict[str, Any]:
        """Extract search parameters from the query using regex patterns"""
        import re

        params = {"query": query}

        # Extract lesson number using regex
        # Patterns: "lesson 1", "lesson 2", "in lesson 3", etc.
        lesson_patterns = [
            r"\blesson\s+(\d+)\b",
            r"\bin\s+lesson\s+(\d+)\b",
            r"\bof\s+lesson\s+(\d+)\b",
        ]

        for pattern in lesson_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                params["lesson_number"] = int(match.group(1))
                break

        # Extract course name using pattern matching
        # Patterns: "in the X course", "from X course", "about X course", etc.
        course_patterns = [
            r"in\s+(?:the\s+)?([^?]+?)\s+course",
            r"from\s+(?:the\s+)?([^?]+?)\s+course",
            r"about\s+(?:the\s+)?([^?]+?)\s+course",
            r"of\s+(?:the\s+)?([^?]+?)\s+course",
        ]

        for pattern in course_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                course_name = match.group(1).strip()
                # Clean up common words
                course_name = re.sub(
                    r"\s+(is|are|was|were)\s*$", "", course_name, flags=re.IGNORECASE
                )
                params["course_name"] = course_name
                break

        # If no explicit course pattern, check for course names directly
        if "course_name" not in params:
            # Check for common course identifiers
            course_identifiers = [
                r"\bMCP\b",
                r"\bIntroduction\s+to\s+MCP\b",
                r"\bAnthropic\b",
                r"\bClaude\b",
                r"\bBuilding\s+with\s+\w+\b",
            ]

            for pattern in course_identifiers:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    params["course_name"] = match.group(0)
                    break

        return params
