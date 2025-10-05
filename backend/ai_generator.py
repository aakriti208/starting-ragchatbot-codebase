from typing import Any, Dict, List, Optional

import anthropic
from anthropic import APIError, AuthenticationError, RateLimitError


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive search tools for course information.

Available Tools:
1. **Course Content Search**: For questions about specific course content or detailed educational materials
2. **Course Outline**: For questions about course structure, lessons, or course overview

Tool Usage Guidelines:
- **Course outline queries**: Use the course outline tool for questions about:
  - Course structure or lesson overview
  - "What lessons are in [course]?"
  - "Show me the outline of [course]"
  - Course title, instructor, or course link information
- **Course content queries**: Use the content search tool for:
  - Specific content within lessons
  - Detailed educational materials
  - Questions about concepts, examples, or explanations
- **One tool call per query maximum**
- Synthesize tool results into accurate, fact-based responses
- If tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without tool usage
- **Course-specific questions**: Use appropriate tool first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results" or "using the tool"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content,
        }

        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        # Get response from Claude with error handling
        try:
            response = self.client.messages.create(**api_params)

            # Handle tool execution if needed
            if response.stop_reason == "tool_use" and tool_manager:
                return self._handle_tool_execution(response, api_params, tool_manager)

            # Return direct response
            return response.content[0].text

        except AuthenticationError:
            return "Error: Invalid Anthropic API key. Please check your API key configuration."
        except RateLimitError:
            return "Error: API rate limit exceeded. Please try again in a moment."
        except APIError as e:
            if "credit balance" in str(e).lower():
                return "Error: Insufficient Anthropic API credits. Please add credits to your Anthropic account to continue using the service."
            return f"Error: API request failed - {str(e)}"
        except Exception as e:
            return f"Error: An unexpected error occurred - {str(e)}"

    def _handle_tool_execution(
        self, initial_response, base_params: Dict[str, Any], tool_manager
    ):
        """
        Handle execution of tool calls and get follow-up response.

        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()

        # Add AI's tool use response
        messages.append({"role": "assistant", "content": initial_response.content})

        # Execute all tool calls and collect results
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

        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"],
        }

        # Get final response with error handling
        try:
            final_response = self.client.messages.create(**final_params)
            return final_response.content[0].text
        except AuthenticationError:
            return "Error: Invalid Anthropic API key. Please check your API key configuration."
        except RateLimitError:
            return "Error: API rate limit exceeded. Please try again in a moment."
        except APIError as e:
            if "credit balance" in str(e).lower():
                return "Error: Insufficient Anthropic API credits. Please add credits to your Anthropic account to continue using the service."
            return f"Error: API request failed - {str(e)}"
        except Exception as e:
            return f"Error: An unexpected error occurred - {str(e)}"
