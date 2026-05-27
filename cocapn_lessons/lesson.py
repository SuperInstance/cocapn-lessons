"""lesson.py — Core Lesson, Trial, and LessonLibrary classes.

A Trial is a single attempt (success or failure). A Lesson compiles
multiple trials into retrievable knowledge with O(1/n) failure rate.
"""
import json
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from datetime import datetime, timezone
from collections import Counter
from enum import Enum


class FailureMode(Enum):
    """Classification of how a trial failed."""
    TIMEOUT = "timeout"
    ERROR = "error"
    REJECTION = "rejection"
    WRONG_RESULT = "wrong_result"
    CRASH = "crash"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    EXTERNAL_DEPENDENCY = "external_dependency"
    UNKNOWN = "unknown"


class Category(Enum):
    """Topic category for a lesson."""
    TOOLING = "tooling"
    NAVIGATION = "navigation"
    API = "api"
    STRATEGY = "strategy"
    RESOURCE_MANAGEMENT = "resource_management"
    COMMUNICATION = "communication"
    INFRASTRUCTURE = "infrastructure"
    GENERAL = "general"


@dataclass
class Trial:
    """A single attempt at a task, successful or not."""
    task: str
    agent: str = "unknown"
    success: bool = False
    error: str = ""
    technique: str = ""
    tokens_used: int = 0
    duration_sec: float = 0.0
    tags: List[str] = field(default_factory=list)
    failure_mode: Optional[FailureMode] = None
    root_cause: str = ""
    at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def fingerprint(self) -> str:
        """Hash of task+error for deduplication."""
        return hashlib.sha256(f"{self.task}:{self.error}".encode()).hexdigest()[:16]


@dataclass
class Lesson:
    """Compiled knowledge from multiple trials of the same task.

    Attributes:
        task: Human-readable task description.
        category: Topic classification.
        confidence: 0.0–1.0 based on trial count and consistency.
        applicability: Domains/tags where this lesson applies.
    """
    task: str
    description: str = ""
    category: Category = Category.GENERAL
    trials: List[Trial] = field(default_factory=list)
    applicability: List[str] = field(default_factory=list)
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

    @property
    def confidence(self) -> float:
        """Confidence in the lesson, 0.0–1.0.

        Based on number of trials and consistency of results.
        More trials → higher confidence. Consistent results → higher confidence.
        """
        n = len(self.trials)
        if n == 0:
            return 0.0
        # Sample size factor: log scale, saturates around 10 trials
        sample_factor = min(1.0, (n / 10.0) ** 0.5)
        # Consistency factor: how decisive the results are
        sr = self.success_rate
        consistency = 1.0 - 2.0 * min(sr, 1.0 - sr)
        return round(sample_factor * consistency, 3)

    def failure_modes(self, top_n: int = 3) -> List[tuple]:
        """Most common errors. Returns [(error, count, success_rate)]."""
        errors: Counter = Counter()
        error_success: Counter = Counter()
        for t in self.trials:
            if not t.success and t.error:
                errors[t.error] += 1
            else:
                errors["(success)"] += 1
            if t.success:
                error_success[t.error or "(success)"] += 1
        return [
            (err, cnt, error_success[err] / cnt)
            for err, cnt in errors.most_common(top_n)
        ]

    def advice_for_new_agent(self) -> str:
        """Generate advice based on trial history."""
        if not self.trials:
            return "No trials yet. Be the first."
        winning = [t.technique for t in self.trials if t.success and t.technique]
        failing = [t.technique for t in self.trials if not t.success and t.technique]
        common_errors = self.failure_modes(top_n=2)

        lines = [
            f"Lesson: {self.task}",
            f"Category: {self.category.value}",
            f"Confidence: {self.confidence:.0%}",
            f"Success rate: {self.success_rate:.0%} "
            f"({sum(1 for t in self.trials if t.success)}/{len(self.trials)})",
            f"Agents attempted: {self.unique_agents}",
        ]
        if common_errors and common_errors[0][0] != "(success)":
            lines.append(
                f"Common failure: '{common_errors[0][0]}' ({common_errors[0][1]}x)"
            )
        if winning:
            lines.append(f"Winning technique: {winning[-1]}")
        if failing:
            lines.append(f"Avoid: {failing[0]}")
        if self.applicability:
            lines.append(f"Applies to: {', '.join(self.applicability)}")
        return "\n".join(lines)

    def predict_failure_rate(self, n_agents: int) -> float:
        """Mathematical model: failure rate ~ O(1/n) for common errors."""
        if self.success_rate == 1.0:
            return 0.0
        if self.success_rate == 0.0 and self.total_attempts < 3:
            return 0.8
        base = 1 - self.success_rate
        return base / (1 + n_agents / max(self.unique_agents, 1))

    def winning_technique(self) -> Optional[str]:
        """Return the technique from the most recent successful trial."""
        for t in reversed(self.trials):
            if t.success and t.technique:
                return t.technique
        return None

    def save(self, path: str = None) -> str:
        path = (
            path
            or f"lesson_{self.task.lower().replace(' ', '_').replace('/', '_')[:40]}.json"
        )
        def _serialize(o):
            if isinstance(o, (FailureMode, Category)):
                return o.value
            if hasattr(o, '__dataclass_fields__'):
                return asdict(o)
            return str(o)

        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=_serialize)
        return path

    @classmethod
    def load(cls, path: str) -> "Lesson":
        with open(path) as f:
            data = json.load(f)
        trials_data = data.pop("trials", [])
        trials = []
        for t in trials_data:
            if "failure_mode" in t and t["failure_mode"] and not isinstance(t["failure_mode"], FailureMode):
                try:
                    t["failure_mode"] = FailureMode(t["failure_mode"])
                except ValueError:
                    t["failure_mode"] = FailureMode.UNKNOWN
            trials.append(Trial(**t))
        if "category" in data and not isinstance(data["category"], Category):
            try:
                data["category"] = Category(data["category"])
            except (ValueError, TypeError):
                data["category"] = Category.GENERAL
        return cls(trials=trials, **data)


class LessonLibrary:
    """Indexed collection of lessons with search and fleet statistics."""

    def __init__(self):
        self.lessons: Dict[str, Lesson] = {}

    def get_or_create(self, task: str, **kwargs) -> Lesson:
        if task not in self.lessons:
            self.lessons[task] = Lesson(task=task, **kwargs)
        return self.lessons[task]

    def search(
        self,
        tag: str = None,
        category: Category = None,
        min_confidence: float = None,
        min_success_rate: float = None,
    ) -> List[Lesson]:
        results = list(self.lessons.values())
        if tag:
            results = [
                l for l in results if any(tag in t.tags for t in l.trials)
            ]
        if category is not None:
            results = [l for l in results if l.category == category]
        if min_confidence is not None:
            results = [l for l in results if l.confidence >= min_confidence]
        if min_success_rate is not None:
            results = [l for l in results if l.success_rate >= min_success_rate]
        return results

    def fleet_stats(self) -> dict:
        total_trials = sum(len(l.trials) for l in self.lessons.values())
        total_lessons = len(self.lessons)
        avg_success = (
            sum(l.success_rate for l in self.lessons.values()) / max(total_lessons, 1)
        )
        return {
            "lessons": total_lessons,
            "trials": total_trials,
            "avg_success_rate": round(avg_success, 2),
            "unique_agents": len(
                set(
                    t.agent
                    for l in self.lessons.values()
                    for t in l.trials
                )
            ),
        }

    def save(self, path: str = "lesson_library.json"):
        with open(path, "w") as f:
            json.dump(
                {k: asdict(v) for k, v in self.lessons.items()},
                f,
                indent=2,
                default=lambda o: o.value if isinstance(o, (FailureMode, Category)) else str(o),
            )

    @classmethod
    def load(cls, path: str) -> "LessonLibrary":
        lib = cls()
        with open(path) as f:
            data = json.load(f)
        for task, lesson_data in data.items():
            lib.lessons[task] = Lesson.load_from_dict(lesson_data)
        return lib
