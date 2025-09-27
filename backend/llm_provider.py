from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """Generate response from the LLM"""
        pass

class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider implementation"""

    def __init__(self, api_key: str, model: str):
        import anthropic
        from anthropic import APIError, AuthenticationError, RateLimitError

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
        self.APIError = APIError
        self.AuthenticationError = AuthenticationError
        self.RateLimitError = RateLimitError

    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:

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
            "system": system_content
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

    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        messages = base_params["messages"].copy()
        messages.append({"role": "assistant", "content": initial_response.content})

        tool_results = []
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(
                    content_block.name,
                    **content_block.input
                )

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": tool_result
                })

        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"]
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

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2"):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package required for Ollama provider. Install with: uv add openai")

        self.client = OpenAI(
            base_url=f"{base_url}/v1",
            api_key="ollama"  # Ollama doesn't require real API key
        )
        self.model = model

    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:

        system_prompt = """ You are an AI assistant specialized in course materials and educational content.

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

        messages = [{"role": "system", "content": system_prompt}]

        if conversation_history:
            messages.append({"role": "assistant", "content": f"Previous conversation:\n{conversation_history}"})

        messages.append({"role": "user", "content": query})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
                max_tokens=800
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"Error: Failed to connect to Ollama - {str(e)}. Make sure Ollama is running and the model '{self.model}' is installed."

class LocalAIProvider(LLMProvider):
    """LocalAI provider implementation"""

    def __init__(self, base_url: str = "http://localhost:8080", model: str = "gpt-3.5-turbo"):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package required for LocalAI provider. Install with: uv add openai")

        self.client = OpenAI(
            base_url=f"{base_url}/v1",
            api_key="local"  # LocalAI doesn't require real API key
        )
        self.model = model

    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:

        system_prompt = """ You are an AI assistant specialized in course materials and educational content.

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

        messages = [{"role": "system", "content": system_prompt}]

        if conversation_history:
            messages.append({"role": "assistant", "content": f"Previous conversation:\n{conversation_history}"})

        messages.append({"role": "user", "content": query})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
                max_tokens=800
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"Error: Failed to connect to LocalAI - {str(e)}. Make sure LocalAI is running at {self.client.base_url}."