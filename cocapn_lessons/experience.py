"""experience.py — Capture agent interactions and outcomes.

An Experience is a rich record of an agent's interaction with a task,
including context, actions taken, observations, and the final outcome.
Experiences feed into the LessonExtractor to produce Lessons.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
from enum import Enum


class Outcome(Enum):
    """Result of an agent interaction."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    ERROR = "error"
    ABORTED = "aborted"


class InteractionType(Enum):
    """Kind of interaction the agent performed."""
    CLI_COMMAND = "cli_command"
    API_CALL = "api_call"
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    WEB_REQUEST = "web_request"
    SUBAGENT_SPAWN = "subagent_spawn"
    REASONING = "reasoning"
    EXPLORATION = "exploration"
    TOOL_USE = "tool_use"


@dataclass
class Interaction:
    """A single step in an agent's interaction sequence."""
    type: InteractionType
    description: str
    target: str = ""
    result: str = ""
    duration_sec: float = 0.0
    tokens_used: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return not any(
            w in self.result.lower()
            for w in ("error", "fail", "timeout", "reject", "denied")
        )


@dataclass
class Experience:
    """A complete agent experience: context + interactions + outcome.

    Captures everything about an agent's attempt at a task, enabling
    downstream analysis and lesson extraction.

    Attributes:
        task: What the agent was asked to do.
        agent: Agent identifier.
        context: Environmental context (tools available, constraints, etc.).
        interactions: Ordered sequence of steps the agent took.
        outcome: Final result.
        observations: Agent's own observations/reflections.
    """
    task: str
    agent: str = "unknown"
    context: Dict[str, Any] = field(default_factory=dict)
    interactions: List[Interaction] = field(default_factory=list)
    outcome: Outcome = Outcome.SUCCESS
    observations: List[str] = field(default_factory=list)
    total_tokens: int = 0
    total_duration_sec: float = 0.0
    tags: List[str] = field(default_factory=list)
    at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def add_interaction(self, **kwargs) -> Interaction:
        """Append an interaction and return it."""
        i = Interaction(**kwargs)
        self.interactions.append(i)
        self.total_tokens += i.tokens_used
        self.total_duration_sec += i.duration_sec
        return i

    def add_observation(self, text: str):
        """Record an observation."""
        self.observations.append(text)

    @property
    def succeeded(self) -> bool:
        return self.outcome == Outcome.SUCCESS

    @property
    def failed(self) -> bool:
        return self.outcome in (Outcome.FAILURE, Outcome.TIMEOUT, Outcome.ERROR)

    @property
    def num_interactions(self) -> int:
        return len(self.interactions)

    @property
    def failed_interactions(self) -> List[Interaction]:
        return [i for i in self.interactions if not i.succeeded]

    @property
    def interaction_summary(self) -> Dict[str, int]:
        """Count of interactions by type."""
        from collections import Counter

        counts: Counter = Counter()
        for i in self.interactions:
            counts[i.type.value] += 1
        return dict(counts)

    def summary(self) -> str:
        """Human-readable summary of the experience."""
        lines = [
            f"Task: {self.task}",
            f"Agent: {self.agent}",
            f"Outcome: {self.outcome.value}",
            f"Interactions: {self.num_interactions}",
            f"Tokens: {self.total_tokens}",
            f"Duration: {self.total_duration_sec:.1f}s",
        ]
        if self.tags:
            lines.append(f"Tags: {', '.join(self.tags)}")
        if self.failed_interactions:
            lines.append(f"Failed steps: {len(self.failed_interactions)}")
        if self.observations:
            lines.append("Observations:")
            for obs in self.observations[:5]:
                lines.append(f"  - {obs}")
        return "\n".join(lines)
