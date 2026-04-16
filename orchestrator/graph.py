"""
Multi-Agent Orchestrator using LangGraph

Implements a supervisor pattern:
  - Orchestrator (LLM) decides which agent to call next
  - Agents report back with results
  - Orchestrator synthesizes final answer

Flow for a typical PDE investigation:
  1. Orchestrator → Simulation Agent: "Run heat equation with these params"
  2. Simulation Agent: runs FEM, returns result
  3. Orchestrator → Database Agent: "Store this result"
  4. Database Agent: stores, returns confirmation
  5. Orchestrator → Analytics Agent: "Analyze this run"
  6. Analytics Agent: analyzes, returns insights + suggestions
  7. Orchestrator: if suggestion accepted → back to step 1 (next iteration)
  8. Orchestrator: returns final synthesis to user
"""

from __future__ import annotations

import json
import os
from typing import Annotated, Any, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from agents.simulation_agent import SimulationAgent
from agents.analytics_agent import AnalyticsAgent
from agents.database_agent import DatabaseAgent

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
ORCHESTRATOR_MODEL = os.getenv("ORCHESTRATOR_MODEL", "llama4:scout")

MEMBERS = ["simulation_agent", "analytics_agent", "database_agent", "FINISH"]


# ─── Orchestrator State ───────────────────────────────────────────────────────

class OrchestratorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    task: str
    next_agent: str
    iteration: int
    max_iterations: int
    agent_results: dict[str, Any]
    run_ids: list[str]
    study_id: Optional[str]
    final_report: Optional[str]


# ─── Supervisor Prompt ────────────────────────────────────────────────────────

SUPERVISOR_PROMPT = f"""You are the Orchestrator of a multi-agent PDE simulation ecosystem.

You coordinate three specialist agents to solve PDE problems using FEM:

## Agents:
1. **simulation_agent** - Sets up, runs, and debugs FEM simulations (FEniCSx)
   - Use when: setting up new runs, running parametric sweeps, debugging failures

2. **analytics_agent** - Analyzes results, compares runs, suggests improvements
   - Use when: analyzing completed runs, comparing parametric study results,
               generating insights, deciding next simulation parameters

3. **database_agent** - Stores results, runs queries, catalogs studies
   - Use when: storing results, querying history, exporting data

## Decision rules:
- ALWAYS start with simulation_agent for new simulation tasks
- ALWAYS call database_agent after any successful simulation to store results
- Call analytics_agent after storing results to get insights
- If analytics_agent suggests a new run, call simulation_agent again
- Limit the loop to at most {8} simulation-analyze iterations
- Call FINISH when the task is complete or no more progress can be made

## Output format:
Respond with JSON:
  {{"next": "simulation_agent" | "analytics_agent" | "database_agent" | "FINISH",
    "instructions": "Specific task for the chosen agent",
    "reasoning": "Why you chose this agent"}}
"""


# ─── Orchestrator ─────────────────────────────────────────────────────────────

class MultiAgentOrchestrator:
    """
    Coordinates the three agents in a supervisor pattern.

    The orchestrator LLM decides which agent to invoke next,
    passes it targeted instructions, collects the result,
    and iterates until the goal is achieved.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_agent_calls: int = 20,
        auto_store: bool = True,
        auto_analyze: bool = True,
    ):
        self.model_name = model_name or ORCHESTRATOR_MODEL
        self.max_agent_calls = max_agent_calls
        self.auto_store = auto_store
        self.auto_analyze = auto_analyze

        self.llm = ChatOllama(
            model=self.model_name,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,
            num_ctx=8192,
            format="json",
        )

        # Instantiate agents
        self.simulation_agent = SimulationAgent()
        self.analytics_agent  = AnalyticsAgent()
        self.database_agent   = DatabaseAgent()

        self.graph = self._build_graph()

    # ── Graph construction ────────────────────────────────────────────────────

    def _build_graph(self) -> Any:
        g = StateGraph(OrchestratorState)

        g.add_node("supervisor",        self._supervisor_node)
        g.add_node("simulation_agent",  self._simulation_node)
        g.add_node("analytics_agent",   self._analytics_node)
        g.add_node("database_agent",    self._database_node)
        g.add_node("synthesize",        self._synthesize_node)

        g.set_entry_point("supervisor")

        g.add_conditional_edges(
            "supervisor",
            self._route,
            {
                "simulation_agent": "simulation_agent",
                "analytics_agent":  "analytics_agent",
                "database_agent":   "database_agent",
                "FINISH":           "synthesize",
            },
        )

        for agent_node in ("simulation_agent", "analytics_agent", "database_agent"):
            g.add_edge(agent_node, "supervisor")

        g.add_edge("synthesize", END)

        return g.compile()

    # ── Node implementations ──────────────────────────────────────────────────

    def _supervisor_node(self, state: OrchestratorState) -> dict:
        """LLM decides which agent to call next."""
        # Build context summary for the LLM
        context_parts = [f"Original task: {state['task']}"]
        if state["agent_results"]:
            context_parts.append("Previous agent results:")
            for agent, result in state["agent_results"].items():
                summary = str(result.get("answer", ""))[:500]
                context_parts.append(f"  {agent}: {summary}")
        if state["run_ids"]:
            context_parts.append(f"Completed run_ids: {state['run_ids']}")
        context_parts.append(f"Iteration: {state['iteration']}/{state['max_iterations']}")

        messages = [
            SystemMessage(content=SUPERVISOR_PROMPT),
            HumanMessage(content="\n".join(context_parts)),
        ]

        response = self.llm.invoke(messages)
        try:
            decision = json.loads(response.content)
        except json.JSONDecodeError:
            # If JSON parse fails, finish
            decision = {"next": "FINISH", "instructions": "Parse error, stopping.",
                        "reasoning": "Could not parse supervisor response."}

        return {
            "messages": [AIMessage(content=json.dumps(decision))],
            "next_agent": decision.get("next", "FINISH"),
            "iteration": state["iteration"] + 1,
        }

    def _route(self, state: OrchestratorState) -> str:
        """Route to the next agent or FINISH."""
        if state["iteration"] >= state["max_iterations"]:
            return "FINISH"
        return state.get("next_agent", "FINISH")

    def _get_latest_instructions(self, state: OrchestratorState) -> str:
        """Extract the instructions from the supervisor's last decision."""
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage):
                try:
                    d = json.loads(msg.content)
                    return d.get("instructions", state["task"])
                except (json.JSONDecodeError, AttributeError):
                    pass
        return state["task"]

    def _simulation_node(self, state: OrchestratorState) -> dict:
        instructions = self._get_latest_instructions(state)
        result = self.simulation_agent.run(
            instructions,
            context={"run_ids": state["run_ids"]},
        )

        # Extract run_id from result if present
        new_run_ids = list(state.get("run_ids", []))
        try:
            answer = result.get("answer", "")
            if isinstance(answer, str) and "run_id" in answer:
                # Try parsing as JSON
                d = json.loads(answer)
                rid = d.get("run_id")
                if rid and rid not in new_run_ids:
                    new_run_ids.append(rid)
        except (json.JSONDecodeError, Exception):
            pass

        results = dict(state.get("agent_results", {}))
        results["simulation_agent"] = result

        return {
            "messages": [HumanMessage(content=f"simulation_agent result: {result['answer'][:500]}")],
            "agent_results": results,
            "run_ids": new_run_ids,
        }

    def _analytics_node(self, state: OrchestratorState) -> dict:
        instructions = self._get_latest_instructions(state)
        # Add run context to instructions
        if state["run_ids"]:
            instructions += f"\n\nRun IDs to analyze: {json.dumps(state['run_ids'])}"

        result = self.analytics_agent.run(
            instructions,
            context={"run_ids": state["run_ids"], "study_id": state.get("study_id")},
        )

        results = dict(state.get("agent_results", {}))
        results["analytics_agent"] = result

        return {
            "messages": [HumanMessage(content=f"analytics_agent result: {result['answer'][:500]}")],
            "agent_results": results,
        }

    def _database_node(self, state: OrchestratorState) -> dict:
        instructions = self._get_latest_instructions(state)
        if state["run_ids"]:
            instructions += f"\n\nRun IDs to process: {json.dumps(state['run_ids'])}"

        result = self.database_agent.run(
            instructions,
            context={"run_ids": state["run_ids"], "study_id": state.get("study_id")},
        )

        results = dict(state.get("agent_results", {}))
        results["database_agent"] = result

        return {
            "messages": [HumanMessage(content=f"database_agent result: {result['answer'][:500]}")],
            "agent_results": results,
        }

    def _synthesize_node(self, state: OrchestratorState) -> dict:
        """Produce a final synthesized report from all agent results."""
        synthesis_llm = ChatOllama(
            model=self.model_name,
            base_url=OLLAMA_BASE_URL,
            temperature=0.2,
            num_ctx=8192,
        )

        parts = [
            f"Task: {state['task']}",
            f"Total iterations: {state['iteration']}",
            f"Completed runs: {state['run_ids']}",
            "",
            "Agent results:",
        ]
        for agent, result in state.get("agent_results", {}).items():
            parts.append(f"\n## {agent}:\n{result.get('answer', 'No result')}")

        synthesis_prompt = (
            "Summarize the PDE simulation investigation. Include:\n"
            "1. What was done (simulations run, parameters used)\n"
            "2. Key findings (temperature ranges, convergence, performance)\n"
            "3. Insights from analysis\n"
            "4. Recommended next steps\n\n"
            + "\n".join(parts)
        )

        response = synthesis_llm.invoke([HumanMessage(content=synthesis_prompt)])
        report = response.content

        return {
            "messages": [AIMessage(content=report)],
            "final_report": report,
        }

    # ── Public interface ──────────────────────────────────────────────────────

    def run(self, task: str, max_iterations: int = 20) -> dict:
        """
        Execute the full multi-agent workflow for a given task.

        Args:
            task:           Natural language task description.
            max_iterations: Maximum number of agent calls.

        Returns:
            dict with 'final_report', 'run_ids', 'agent_results'
        """
        initial_state: OrchestratorState = {
            "messages": [HumanMessage(content=task)],
            "task": task,
            "next_agent": "",
            "iteration": 0,
            "max_iterations": max_iterations,
            "agent_results": {},
            "run_ids": [],
            "study_id": None,
            "final_report": None,
        }

        final_state = self.graph.invoke(initial_state)

        return {
            "task": task,
            "final_report": final_state.get("final_report", ""),
            "run_ids": final_state.get("run_ids", []),
            "agent_results": final_state.get("agent_results", {}),
            "total_iterations": final_state.get("iteration", 0),
        }

    def stream(self, task: str, max_iterations: int = 20):
        """Stream intermediate steps for real-time monitoring."""
        initial_state: OrchestratorState = {
            "messages": [HumanMessage(content=task)],
            "task": task,
            "next_agent": "",
            "iteration": 0,
            "max_iterations": max_iterations,
            "agent_results": {},
            "run_ids": [],
            "study_id": None,
            "final_report": None,
        }
        for event in self.graph.stream(initial_state, stream_mode="updates"):
            yield event
