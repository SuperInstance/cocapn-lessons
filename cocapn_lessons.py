"""cocapn_lessons — Trial-based learning for distributed agent fleets.

Every failed execution becomes a `Trial` — a structured negative example
that future agents learn from. The `Lesson` class compiles trials into
retrievable knowledge. The math: common-error failure rate drops as O(1/n)
where n = agents who have attempted the task.

Usage:
    lesson = Lesson("README detection via gh CLI")
    lesson.record_trial(success=False, agent="ccc-direct", error="gh repo view fails")
    lesson.record_trial(success=True, agent="kimi-auditor", technique="gh api repos/.../readme")
    advice = lesson.advice_for_new_agent()  # "Use gh api, not gh repo view"
"""
import json
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from datetime import datetime, timezone
from collections import Counter


@dataclass
class Trial:
    """A single attempt at a task, successful or not."""
    task: str
    agent: str = "unknown"
    success: bool = False
    error: str = ""
    technique: str = ""           # what the agent tried
    tokens_used: int = 0
    duration_sec: float = 0.0
    tags: List[str] = field(default_factory=list)
    at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def fingerprint(self) -> str:
        """Hash of task+error for deduplication."""
        return hashlib.sha256(f"{self.task}:{self.error}".encode()).hexdigest()[:16]


@dataclass
class Lesson:
    """Compiled knowledge from multiple trials of the same task."""
    task: str
    description: str = ""
    trials: List[Trial] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def record_trial(self, **kwargs) -> Trial:
        """Record a new trial and return it."""
        t = Trial(task=self.task, **kwargs)
        self.trials.append(t)
        return t

    @property
    def total_attempts(self) -> int:
        return len(self.trials)

    @property
    def success_rate(self) -> float:
        if not self.trials:
            return 0.0
        return sum(1 for t in self.trials if t.success) / len(self.trials)

    @property
    def unique_agents(self) -> int:
        return len(set(t.agent for t in self.trials))

    def failure_modes(self, top_n: int = 3) -> List[tuple]:
        """Most common errors. Returns [(error, count, success_rate)]."""
        errors = Counter()
        error_success = Counter()
        for t in self.trials:
            if not t.success and t.error:
                errors[t.error] += 1
            else:
                errors["(success)"] += 1
            if t.success:
                error_success[t.error or "(success)"] += 1
        return [(err, cnt, error_success[err]/cnt) for err, cnt in errors.most_common(top_n)]

    def advice_for_new_agent(self) -> str:
        """Generate advice based on trial history."""
        if not self.trials:
            return "No trials yet. Be the first."
        # Find techniques that succeeded
        winning = [t.technique for t in self.trials if t.success and t.technique]
        failing = [t.technique for t in self.trials if not t.success and t.technique]
        common_errors = self.failure_modes(top_n=2)

        lines = [
            f"Lesson: {self.task}",
            f"Success rate: {self.success_rate:.0%} ({sum(1 for t in self.trials if t.success)}/{len(self.trials)})",
            f"Agents attempted: {self.unique_agents}",
        ]
        if common_errors and common_errors[0][0] != "(success)":
            lines.append(f"Common failure: '{common_errors[0][0]}' ({common_errors[0][1]}x)")
        if winning:
            lines.append(f"Winning technique: {winning[-1]}")
        if failing:
            lines.append(f"Avoid: {failing[0]}")
        return "\n".join(lines)

    def predict_failure_rate(self, n_agents: int) -> float:
        """Mathematical model: failure rate ~ 1/n for common errors."""
        if self.success_rate == 1.0:
            return 0.0
        if self.success_rate == 0.0 and self.total_attempts < 3:
            return 0.8  # high uncertainty
        # Simplified O(1/n) model: base_rate / (1 + n/unique_agents)
        base = 1 - self.success_rate
        return base / (1 + n_agents / max(self.unique_agents, 1))

    def save(self, path: str = None) -> str:
        path = path or f"lesson_{self.task.lower().replace(' ', '_').replace('/', '_')[:40]}.json"
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=lambda o: asdict(o) if hasattr(o, '__dataclass_fields__') else str)
        return path

    @classmethod
    def load(cls, path: str) -> "Lesson":
        with open(path) as f:
            data = json.load(f)
        data["trials"] = [Trial(**t) for t in data.get("trials", [])]
        return cls(**data)


class LessonLibrary:
    """Indexed collection of lessons."""
    def __init__(self):
        self.lessons: Dict[str, Lesson] = {}

    def get_or_create(self, task: str) -> Lesson:
        if task not in self.lessons:
            self.lessons[task] = Lesson(task=task)
        return self.lessons[task]

    def search(self, tag: str = None, min_success_rate: float = None) -> List[Lesson]:
        results = list(self.lessons.values())
        if tag:
            results = [l for l in results if any(tag in t.tags for t in l.trials)]
        if min_success_rate is not None:
            results = [l for l in results if l.success_rate >= min_success_rate]
        return results

    def fleet_stats(self) -> dict:
        total_trials = sum(len(l.trials) for l in self.lessons.values())
        total_lessons = len(self.lessons)
        avg_success = sum(l.success_rate for l in self.lessons.values()) / max(total_lessons, 1)
        return {
            "lessons": total_lessons,
            "trials": total_trials,
            "avg_success_rate": round(avg_success, 2),
            "unique_agents": len(set(t.agent for l in self.lessons.values() for t in l.trials)),
        }

    def save(self, path: str = "lesson_library.json"):
        with open(path, "w") as f:
            json.dump({k: asdict(v) for k, v in self.lessons.items()}, f, indent=2, default=lambda o: asdict(o) if hasattr(o, '__dataclass_fields__') else str)


if __name__ == "__main__":
    # Demo: replicate the README-detection lesson from CCC's audit
    lib = LessonLibrary()
    lesson = lib.get_or_create("README detection via gh CLI")
    lesson.description = "Detect if a GitHub repo has a README using CLI tools."

    # Trials from CCC's 2026-05-03 audit
    lesson.record_trial(agent="ccc-direct", success=False, error="gh repo view --readme fails for 95% of repos", technique="gh repo view --readme", tokens_used=8000, tags=["github", "readme", "cli"])
    lesson.record_trial(agent="kimi-auditor", success=True, error="", technique="gh api repos/{owner}/{repo}/readme", tokens_used=3000, tags=["github", "readme", "api"])
    lesson.record_trial(agent="ccc-scout-2", success=True, error="", technique="gh api repos/{owner}/{repo}/readme", tokens_used=2500, tags=["github", "readme", "api"])
    lesson.record_trial(agent="readme-builder", success=False, error="README already exists — false positive", technique="gh repo view --readme", tokens_used=4000, tags=["github", "readme", "cli"])

    print(lesson.advice_for_new_agent())
    print()
    print(f"Predicted failure rate after 10 agents: {lesson.predict_failure_rate(10):.0%}")
    print()
    print(f"Fleet stats: {lib.fleet_stats()}")
    print()
    path = lesson.save()
    print(f"Saved to {path}")
