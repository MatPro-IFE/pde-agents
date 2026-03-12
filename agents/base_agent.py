"""
Base agent class shared by all three agents.

Each agent is a LangGraph StateGraph with:
  - A reasoning node (LLM call + tool-call parser)
  - A tool execution node
  - A finish node
  - Edges that loop until completion or max_iterations

State is typed via TypedDict and persists across the graph nodes.

Tool-calling compatibility notes:
  - llama3.3:70b   → native Ollama tool_calls (structured)
  - qwen2.5-coder  → outputs tool call as JSON text in content
  The _parse_content_tool_call fallback handles the qwen case transparently.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Annotated, Any, Optional, Sequence, TypedDict

from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage,
)
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ─── DB logging (optional — fails gracefully if DB is unavailable) ─────────────

try:
    from database.operations import (
        log_agent_step as _db_log_step,
        backfill_task_run_id as _db_backfill,
    )
    _DB_LOG_AVAILABLE = True
except ImportError:
    _DB_LOG_AVAILABLE = False

# Per-process counter keyed by task_id so step_index is monotone
_STEP_COUNTER: dict[str, int] = {}


def _safe_log(
    task_id: str,
    run_id: str | None,
    agent_name: str,
    step_type: str,
    content: dict,
    elapsed_ms: int | None = None,
) -> None:
    """Write one agent log step to the DB; silently swallows all errors."""
    if not _DB_LOG_AVAILABLE or not task_id:
        return
    try:
        idx = _STEP_COUNTER.get(task_id, 0)
        _STEP_COUNTER[task_id] = idx + 1
        _db_log_step(
            task_id=task_id,
            agent_name=agent_name,
            step_index=idx,
            step_type=step_type,
            content=content,
            run_id=run_id,
            elapsed_ms=elapsed_ms,
        )
    except Exception:
        pass


def _extract_run_id(messages: list) -> str | None:
    """Scan ToolMessages for a 'run_id' field in their JSON content."""
    for msg in reversed(messages):
        if not isinstance(msg, ToolMessage):
            continue
        try:
            data = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
            if isinstance(data, dict) and data.get("run_id"):
                return str(data["run_id"])
        except Exception:
            pass
    return None


# ─── Shared State ─────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """State that flows through all nodes in the agent graph."""
    messages: Annotated[list[BaseMessage], add_messages]
    task: str                          # original task description
    agent_name: str                    # which agent is running
    iteration: int                     # iteration counter
    max_iterations: int                # safety limit
    status: str                        # "running" | "done" | "error"
    final_answer: Optional[str]        # structured result when done
    tool_calls_log: list[dict]         # audit log of tool invocations
    context: dict[str, Any]            # shared context (run_id, study_id, task_id, etc.)
    task_id: str                       # UUID for this agent invocation (for DB logging)


# ─── Base Agent ───────────────────────────────────────────────────────────────

class BaseAgent:
    """
    Base class for ReAct-style agents built on LangGraph.

    Subclasses provide:
      - system_prompt   : the agent's persona and instructions
      - tools           : list of LangChain tools available to this agent
      - model_name      : Ollama model to use
    """

    system_prompt: str = "You are a helpful AI assistant."
    tools: list[BaseTool] = []
    model_name: str = "qwen2.5-coder:14b"
    agent_name: str = "base"
    max_iterations: int = 20
    temperature: float = 0.1

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_iterations: Optional[int] = None,
        temperature: Optional[float] = None,
        ollama_base_url: Optional[str] = None,
    ):
        self.model_name = model_name or self.__class__.model_name
        self.max_iterations = max_iterations or self.__class__.max_iterations
        self.temperature = temperature or self.__class__.temperature
        self.ollama_url = ollama_base_url or OLLAMA_BASE_URL

        self.llm = ChatOllama(
            model=self.model_name,
            base_url=self.ollama_url,
            temperature=self.temperature,
            num_ctx=8192,
        ).bind_tools(self.tools)

        self.tool_node = ToolNode(self.tools)
        self.graph = self._build_graph()

    # ── Graph construction ────────────────────────────────────────────────────

    def _build_graph(self) -> StateGraph:
        g = StateGraph(AgentState)

        g.add_node("reason", self._reason_node)
        g.add_node("act",    self.tool_node)
        g.add_node("finish", self._finish_node)

        g.set_entry_point("reason")

        g.add_conditional_edges(
            "reason",
            self._router,
            {
                "act":    "act",
                "finish": "finish",
            },
        )
        g.add_edge("act", "reason")
        g.add_edge("finish", END)

        return g.compile()

    # ── Tool-call content parser ──────────────────────────────────────────────

    def _parse_content_tool_call(self, response: AIMessage) -> AIMessage:
        """
        Fallback for models (e.g. qwen2.5-coder via Ollama) that output tool
        calls as JSON text in `content` rather than in `tool_calls`.

        Handles three patterns:
          1. Content is a bare JSON object: {"name": "tool", "arguments": {...}}
          2. JSON embedded in markdown fences: ```json\n{"name":...}\n```
          3. JSON embedded anywhere in mixed text (e.g. "I apologize... {\"name\":...}")
        """
        import re

        if response.tool_calls:
            return response  # already structured, nothing to do

        content = (response.content or "").strip()
        if not content:
            return response

        known_tools = {t.name for t in self.tools}
        parsed = None

        # ── Pass 1: whole content is JSON ─────────────────────────────────────
        if content.startswith("{"):
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                pass

        # ── Pass 2: markdown code fence  ```json ... ``` ──────────────────────
        if parsed is None:
            m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass

        # ── Pass 3: first { ... } block anywhere in the text ─────────────────
        if parsed is None:
            m = re.search(r"(\{.*\})", content, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass

        if parsed is None or not isinstance(parsed, dict):
            return response

        name = parsed.get("name") or parsed.get("tool") or parsed.get("function")
        args = parsed.get("arguments") or parsed.get("args") or parsed.get("parameters", {})

        if not name or not isinstance(args, dict):
            return response

        if name not in known_tools:
            return response

        tool_call = {
            "name": name,
            "args": args,
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "tool_call",
        }
        return AIMessage(content="", tool_calls=[tool_call])

    # ── Node implementations ──────────────────────────────────────────────────

    def _reason_node(self, state: AgentState) -> dict:
        """Call the LLM to produce the next action or final answer."""
        messages = list(state["messages"])
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=self.system_prompt)] + messages

        task_id  = state.get("task_id", "")
        context  = dict(state.get("context", {}))
        iteration = state["iteration"]

        # ── Log tool results that the act-node just wrote ──────────────────
        # After iteration 0 the last messages (before calling the LLM) are
        # ToolMessages from the most recent tool execution.
        if iteration > 0:
            tool_msgs: list[ToolMessage] = []
            for msg in reversed(messages):
                if isinstance(msg, ToolMessage):
                    tool_msgs.append(msg)
                else:
                    break
            for tmsg in reversed(tool_msgs):
                try:
                    result_data = (json.loads(tmsg.content)
                                   if isinstance(tmsg.content, str) else tmsg.content)
                except Exception:
                    result_data = {"raw": str(tmsg.content)[:2000]}

                # Back-fill run_id into context as soon as we see it
                if not context.get("run_id") and isinstance(result_data, dict):
                    found = result_data.get("run_id") or _extract_run_id(messages)
                    if found:
                        context["run_id"] = found
                        # Retroactively tag all earlier log rows for this task
                        if _DB_LOG_AVAILABLE:
                            try:
                                _db_backfill(task_id, found)
                            except Exception:
                                pass

                _safe_log(task_id, context.get("run_id"), self.agent_name,
                          "tool_result",
                          {"tool_call_id": getattr(tmsg, "tool_call_id", ""),
                           "result": result_data})

        # ── LLM call ──────────────────────────────────────────────────────
        t0 = time.perf_counter()
        response = self.llm.invoke(messages)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        # Normalize: parse JSON-text tool calls from models like qwen2.5-coder
        response = self._parse_content_tool_call(response)

        # ── Log reasoning step ────────────────────────────────────────────
        _safe_log(task_id, context.get("run_id"), self.agent_name,
                  "reasoning",
                  {"iteration": iteration, "model": self.model_name,
                   "content": (response.content or "")[:3000]},
                  elapsed_ms=elapsed_ms)

        # ── Log each tool call the LLM wants to make ──────────────────────
        for tc in response.tool_calls:
            _safe_log(task_id, context.get("run_id"), self.agent_name,
                      "tool_call",
                      {"tool": tc["name"], "args": tc.get("args", {})})

        log_entry = {
            "iteration": iteration,
            "model": self.model_name,
            "elapsed_ms": elapsed_ms,
            "has_tool_calls": bool(response.tool_calls),
        }

        return {
            "messages": [response],
            "iteration": iteration + 1,
            "tool_calls_log": state.get("tool_calls_log", []) + [log_entry],
            "context": context,
        }

    def _finish_node(self, state: AgentState) -> dict:
        """Extract the final answer from the last AI message."""
        last = state["messages"][-1]
        content = last.content if hasattr(last, "content") else str(last)
        # Strip ChatML / tokenizer artifacts produced by some qwen models
        for marker in ("<|im_start|>", "<|im_end|>", "<|endoftext|>"):
            content = content.split(marker)[0].strip()

        task_id = state.get("task_id", "")
        context = state.get("context", {})
        _safe_log(task_id, context.get("run_id"), self.agent_name,
                  "final_answer", {"answer": content[:4000]})

        # Clean up step counter for this task to avoid unbounded growth
        _STEP_COUNTER.pop(task_id, None)

        return {
            "status": "done",
            "final_answer": content,
        }

    def _router(self, state: AgentState) -> str:
        """Route to 'act' if tool calls present, else 'finish'."""
        last = state["messages"][-1]
        if state["iteration"] >= state["max_iterations"]:
            return "finish"
        if isinstance(last, AIMessage) and last.tool_calls:
            return "act"
        return "finish"

    # ── Public interface ──────────────────────────────────────────────────────

    def run(self, task: str, context: Optional[dict] = None) -> dict:
        """
        Execute the agent on a given task.

        Args:
            task:    Natural language description of the task.
            context: Optional shared context dict (run_id, study_id, etc.)

        Returns:
            dict with 'answer', 'iterations', 'tool_calls_log', 'task_id'
        """
        task_id = uuid.uuid4().hex
        ctx = dict(context or {})
        ctx["task_id"] = task_id

        initial_state: AgentState = {
            "messages": [HumanMessage(content=task)],
            "task": task,
            "agent_name": self.agent_name,
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "status": "running",
            "final_answer": None,
            "tool_calls_log": [],
            "context": ctx,
            "task_id": task_id,
        }

        final_state = self.graph.invoke(initial_state)

        return {
            "agent": self.agent_name,
            "task": task,
            "task_id": task_id,
            "run_id": final_state.get("context", {}).get("run_id"),
            "answer": final_state.get("final_answer", ""),
            "status": final_state.get("status", "unknown"),
            "iterations": final_state.get("iteration", 0),
            "tool_calls_log": final_state.get("tool_calls_log", []),
        }

    def stream(self, task: str, context: Optional[dict] = None):
        """Stream intermediate states for real-time monitoring."""
        task_id = uuid.uuid4().hex
        ctx = dict(context or {})
        ctx["task_id"] = task_id
        initial_state: AgentState = {
            "messages": [HumanMessage(content=task)],
            "task": task,
            "agent_name": self.agent_name,
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "status": "running",
            "final_answer": None,
            "tool_calls_log": [],
            "context": ctx,
            "task_id": task_id,
        }
        for event in self.graph.stream(initial_state, stream_mode="updates"):
            yield event
