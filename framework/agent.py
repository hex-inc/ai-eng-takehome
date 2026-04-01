"""Agent framework for autonomous task execution with tool calling.

This module implements an agent that uses the OpenRouter API for LLM inference,
supporting streaming responses, tool calling, and reasoning token display.
"""

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

from framework.llm import OpenRouterClient, OpenRouterConfig, TokenUsage

# Prefix that indicates the agent should stop (answer was submitted)
ANSWER_SUBMITTED_PREFIX = "ANSWER_SUBMITTED:"

type ToolFunction = Callable[..., str]

GUIDES_DIR = Path(__file__).parent.parent / "evaluation" / "data" / "guides"


def _load_guides() -> str:
    """Load all guide markdown files and concatenate them."""
    if not GUIDES_DIR.exists():
        return ""
    parts = []
    for guide_path in sorted(GUIDES_DIR.glob("*.md")):
        parts.append(f"### {guide_path.stem}\n{guide_path.read_text()}")
    return "\n\n---\n\n".join(parts)


class EventType(Enum):
    GENERATION_START = auto()
    THINKING_START = auto()
    THINKING_CHUNK = auto()
    THINKING_END = auto()
    RESPONSE_CHUNK = auto()
    GENERATION_END = auto()
    TOOL_CALL_START = auto()
    TOOL_CALL_PARSED = auto()
    TOOL_EXECUTION_START = auto()
    TOOL_EXECUTION_END = auto()
    ITERATION_START = auto()
    ITERATION_END = auto()
    AGENT_COMPLETE = auto()
    AGENT_ERROR = auto()


@dataclass
class AgentEvent:
    type: EventType
    data: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.type.name}] {self.data}"


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    function: ToolFunction


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    error: str | None = None


@dataclass
class Message:
    role: str
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


@dataclass
class ContextCompressionSettings:
    enabled: bool = False
    keep_recent: int = 3
    max_chars: int = 500


@dataclass
class Conversation:
    messages: list[Message] = field(default_factory=list)

    def to_api_format(
        self,
        compression: ContextCompressionSettings | None = None,
    ) -> list[dict[str, Any]]:
        messages_to_convert = self.messages
        if compression and compression.enabled:
            messages_to_convert = _compress_messages(
                self.messages,
                keep_recent=compression.keep_recent,
                max_chars=compression.max_chars,
            )
        result: list[dict[str, Any]] = []
        for message in messages_to_convert:
            msg: dict[str, Any] = {"role": message.role}
            if message.content is not None:
                msg["content"] = message.content
            if message.tool_calls is not None:
                msg["tool_calls"] = message.tool_calls
            if message.tool_call_id is not None:
                msg["tool_call_id"] = message.tool_call_id
            result.append(msg)
        return result


def _truncate_tool_result(content: str, max_chars: int) -> str:
    if len(content) <= max_chars:
        return content
    first_line = content.split("\n")[0]
    if len(first_line) <= max_chars - 20:
        return f"[Truncated] {first_line}"
    return f"[Truncated] {content[:max_chars - 15]}..."


def _compress_messages(
    messages: list[Message],
    keep_recent: int,
    max_chars: int,
) -> list[Message]:
    tool_indices = [i for i, m in enumerate(messages) if m.role == "tool"]
    recent_tool_indices = set(tool_indices[-keep_recent:]) if tool_indices else set()
    result: list[Message] = []
    for i, msg in enumerate(messages):
        if msg.role == "tool" and i not in recent_tool_indices and msg.content:
            result.append(
                Message(
                    role=msg.role,
                    content=_truncate_tool_result(msg.content, max_chars),
                    tool_call_id=msg.tool_call_id,
                )
            )
        else:
            result.append(msg)
    return result


class Agent:
    """A simple ReAct agent built on the OpenRouter API."""

    def __init__(self, config: OpenRouterConfig, tools: dict[str, Tool]):
        self.config = config
        self.tools = tools
        self.client = OpenRouterClient(config)
        self.conversation = Conversation()
        self._compression = ContextCompressionSettings(
            enabled=config.compress_context,
            keep_recent=config.compress_keep_recent,
            max_chars=config.compress_max_chars,
        )
        self.reset_conversation()

    def _get_tool_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self.tools.values()
        ]

    def _execute_tool(self, tool_call: ToolCall) -> str:
        if tool_call.error:
            return f"Error parsing arguments for tool '{tool_call.name}': {tool_call.error}"
        if tool_call.name not in self.tools:
            return f"Error: Unknown tool '{tool_call.name}'"
        try:
            return self.tools[tool_call.name].function(**tool_call.arguments)
        except Exception as e:
            return f"Error executing {tool_call.name}: {e}"

    def _generate_response(self, conversation: Conversation) -> Iterator[AgentEvent]:
        yield AgentEvent(type=EventType.GENERATION_START)
        messages = conversation.to_api_format(compression=self._compression)
        tools = self._get_tool_definitions() if self.tools else None
        full_content = ""
        tool_calls: list[dict[str, Any]] = []
        in_thinking = False
        finish_reason: str | None = None
        usage: TokenUsage | None = None

        for chunk in self.client.chat_completion_stream(messages, tools):
            if chunk.reasoning_details:
                for detail in chunk.reasoning_details:
                    if detail.get("type") == "reasoning.text":
                        text = detail.get("text", "")
                        if text:
                            if not in_thinking:
                                in_thinking = True
                                yield AgentEvent(type=EventType.THINKING_START)
                            yield AgentEvent(type=EventType.THINKING_CHUNK, data={"chunk": text})
            if chunk.content:
                if in_thinking:
                    in_thinking = False
                    yield AgentEvent(type=EventType.THINKING_END)
                full_content += chunk.content
                yield AgentEvent(type=EventType.RESPONSE_CHUNK, data={"chunk": chunk.content})
            if chunk.tool_calls:
                tool_calls = chunk.tool_calls
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason
            if chunk.usage:
                usage = chunk.usage

        if in_thinking:
            yield AgentEvent(type=EventType.THINKING_END)

        event_data: dict[str, Any] = {
            "full_response": full_content,
            "tool_calls": tool_calls,
            "finish_reason": finish_reason,
        }
        if usage:
            event_data["usage"] = usage
        yield AgentEvent(type=EventType.GENERATION_END, data=event_data)

    def _get_system_message(self) -> str:
        """Build system prompt: full schema map + all business rule guides."""
        from framework.database import list_schemas, list_tables

        schemas = list_schemas()
        schema_lines = []
        for schema in schemas:
            tables = list_tables(schema) or []
            table_str = ", ".join(tables[:10])
            suffix = f" (+{len(tables)-10} more)" if len(tables) > 10 else ""
            schema_lines.append(f"  - {schema}: {table_str}{suffix}")
        schema_block = "\n".join(schema_lines)

        guides = _load_guides()
        guides_block = (
            f"# Business Rules\n\nApply these domain rules when writing SQL:\n\n{guides}"
            if guides else ""
        )

        return f"""You are an autonomous SQL agent querying a DuckDB database.
Complete every task by calling submit_answer with a valid SQL query.
Never ask for clarification — make your best judgment and proceed.

# Available Schemas

Use schema-qualified table names in all queries (e.g. schema.table).

{schema_block}

Use list_tables(schemaName) to see all tables in a schema.
Use describe_table(schemaName, tableName) to see column names and types.
Use run_sql(query) to test a query before submitting.

# Workflow

1. Identify the relevant schema from the list above.
2. Use describe_table to confirm column names before writing SQL.
3. Use run_sql to verify your query returns sensible results.
4. Call submit_answer with your final SQL query.

CRITICAL: You MUST call submit_answer to complete every task.
Do not return plain text answers.

{guides_block}"""

    def run(self, prompt: str) -> Iterator[AgentEvent]:
        """Run the agent on a user prompt."""
        self.conversation.messages.append(Message(role="user", content=prompt))
        total_usage = TokenUsage()
        empty_response_count = 0

        for iteration in range(self.config.max_iterations):
            yield AgentEvent(type=EventType.ITERATION_START, data={"iteration": iteration + 1})

            full_response = ""
            tool_calls_data: list[dict[str, Any]] = []

            for event in self._generate_response(self.conversation):
                yield event
                if event.type == EventType.GENERATION_END:
                    full_response = event.data.get("full_response", "")
                    tool_calls_data = event.data.get("tool_calls", [])
                    if "usage" in event.data and event.data["usage"]:
                        total_usage = total_usage + event.data["usage"]

            tool_calls = _parse_tool_calls_from_api(tool_calls_data)

            if not tool_calls:
                is_empty = not full_response or not full_response.strip()
                if is_empty:
                    empty_response_count += 1
                    if empty_response_count >= 3:
                        yield AgentEvent(
                            type=EventType.AGENT_ERROR,
                            data={"error": "Too many empty generations without tool calls", "usage": total_usage},
                        )
                        return
                    self.conversation.messages.append(Message(role="assistant", content=""))
                    self.conversation.messages.append(
                        Message(
                            role="user",
                            content="You must use the submit_answer tool to complete this task. Call it now with a SQL query.",
                        )
                    )
                    continue

                # Non-empty text with no tool call — nudge toward submit_answer
                self.conversation.messages.append(
                    Message(role="assistant", content=full_response)
                )
                self.conversation.messages.append(
                    Message(
                        role="user",
                        content="You must call submit_answer with a SQL query to complete this task.",
                    )
                )
                continue

            empty_response_count = 0
            yield AgentEvent(type=EventType.TOOL_CALL_START, data={"count": len(tool_calls)})
            self.conversation.messages.append(
                Message(
                    role="assistant",
                    content=full_response if full_response else None,
                    tool_calls=tool_calls_data,
                )
            )

            for tool_call in tool_calls:
                yield AgentEvent(
                    type=EventType.TOOL_CALL_PARSED,
                    data={"name": tool_call.name, "arguments": tool_call.arguments},
                )
                yield AgentEvent(type=EventType.TOOL_EXECUTION_START, data={"name": tool_call.name})
                tool_result = self._execute_tool(tool_call)
                yield AgentEvent(
                    type=EventType.TOOL_EXECUTION_END,
                    data={"name": tool_call.name, "result": tool_result},
                )
                self.conversation.messages.append(
                    Message(role="tool", content=tool_result, tool_call_id=tool_call.id)
                )
                if tool_result.startswith(ANSWER_SUBMITTED_PREFIX):
                    yield AgentEvent(
                        type=EventType.AGENT_COMPLETE,
                        data={"reason": "answer_submitted", "tool": tool_call.name, "usage": total_usage},
                    )
                    return

            yield AgentEvent(type=EventType.ITERATION_END, data={"iteration": iteration + 1})

        yield AgentEvent(
            type=EventType.AGENT_ERROR,
            data={"error": "Max iterations reached", "usage": total_usage},
        )

    def reset_conversation(self) -> None:
        self.conversation = Conversation()
        self.conversation.messages.append(
            Message(role="system", content=self._get_system_message())
        )


def _parse_tool_calls_from_api(tool_calls_data: list[dict[str, Any]]) -> list[ToolCall]:
    tool_calls: list[ToolCall] = []
    for tc in tool_calls_data:
        tc_id = tc.get("id", "")
        function = tc.get("function", {})
        name = function.get("name", "")
        arguments_str = function.get("arguments", "{}")
        try:
            arguments = json.loads(arguments_str)
            error = None
        except json.JSONDecodeError as e:
            arguments = {}
            error = f"Invalid JSON arguments: {e}"
        tool_calls.append(ToolCall(id=tc_id, name=name, arguments=arguments, error=error))
    return tool_calls