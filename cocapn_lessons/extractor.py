"""extractor.py — Extract lessons from agent experiences.

The LessonExtractor analyzes experiences (both successful and failed)
to find patterns, identify root causes, and produce Lessons that
future agents can learn from.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import Counter

from cocapn_lessons.experience import Experience, Outcome, Interaction
from cocapn_lessons.lesson import (
    Lesson,
    Trial,
    FailureMode,
    Category,
    LessonLibrary,
)


def _classify_failure_mode(experience: Experience) -> FailureMode:
    """Infer the failure mode from an experience."""
    if experience.outcome == Outcome.TIMEOUT:
        return FailureMode.TIMEOUT
    if experience.outcome == Outcome.ERROR:
        return FailureMode.CRASH
    if experience.outcome == Outcome.ABORTED:
        return FailureMode.REJECTION

    # Check interaction patterns
    failed = experience.failed_interactions
    if not failed:
        return FailureMode.UNKNOWN

    for i in failed:
        result_lower = i.result.lower()
        if "rate" in result_lower and "limit" in result_lower:
            return FailureMode.RESOURCE_EXHAUSTED
        if "timeout" in result_lower:
            return FailureMode.TIMEOUT
        if "denied" in result_lower or "reject" in result_lower:
            return FailureMode.REJECTION
        if "dependency" in result_lower or "unavailable" in result_lower:
            return FailureMode.EXTERNAL_DEPENDENCY

    return FailureMode.ERROR


def _infer_category(experience: Experience) -> Category:
    """Guess the lesson category from the experience."""
    from cocapn_lessons.experience import InteractionType

    type_counts: Counter = Counter()
    for i in experience.interactions:
        type_counts[i.type] += 1

    if not type_counts:
        return Category.GENERAL

    top = type_counts.most_common(1)[0][0]
    mapping = {
        InteractionType.CLI_COMMAND: Category.TOOLING,
        InteractionType.API_CALL: Category.API,
        InteractionType.WEB_REQUEST: Category.API,
        InteractionType.FILE_READ: Category.TOOLING,
        InteractionType.FILE_WRITE: Category.TOOLING,
        InteractionType.EXPLORATION: Category.NAVIGATION,
        InteractionType.SUBAGENT_SPAWN: Category.RESOURCE_MANAGEMENT,
        InteractionType.REASONING: Category.STRATEGY,
        InteractionType.TOOL_USE: Category.TOOLING,
    }
    return mapping.get(top, Category.GENERAL)


def _infer_root_cause(experience: Experience) -> str:
    """Produce a root-cause string from failed interactions."""
    failed = experience.failed_interactions
    if not failed:
        return ""
    last = failed[-1]
    parts = [f"{last.type.value}: {last.description}"]
    if last.result:
        parts.append(f"Result: {last.result}")
    if last.target:
        parts.append(f"Target: {last.target}")
    return " | ".join(parts)


def _find_technique(experience: Experience) -> str:
    """Extract the technique used (description of the approach)."""
    if not experience.interactions:
        return ""
    steps = [i.description for i in experience.interactions[:5]]
    return "; ".join(steps)


@dataclass
class ExtractedLesson:
    """A lesson candidate extracted from one or more experiences."""
    task: str
    pattern: str  # what was observed
    root_cause: str
    technique: str
    category: Category
    failure_mode: FailureMode
    source_experiences: List[str] = field(default_factory=list)  # agent names
    confidence: float = 0.0
    tags: List[str] = field(default_factory=list)


class LessonExtractor:
    """Analyze experiences and extract lessons.

    Workflow:
        1. Feed experiences via ``add_experience()`` or ``add_experiences()``.
        2. Call ``extract()`` to produce Lesson objects grouped by task.
        3. Optionally ``apply_to_library()`` to merge into a LessonLibrary.
    """

    def __init__(self):
        self.experiences: List[Experience] = []

    def add_experience(self, experience: Experience):
        self.experiences.append(experience)

    def add_experiences(self, experiences: List[Experience]):
        self.experiences.extend(experiences)

    def extract(self) -> List[ExtractedLesson]:
        """Extract lesson candidates from all collected experiences.

        Groups experiences by task, then identifies patterns where:
        - Multiple failures share the same root cause
        - A successful technique differs from a failing one
        - A single failure reveals a clear anti-pattern
        """
        by_task = self._group_by_task()
        lessons: List[ExtractedLesson] = []

        for task, exps in by_task.items():
            failures = [e for e in exps if e.failed]
            successes = [e for e in exps if e.succeeded]

            # Pattern 1: Repeated failures → anti-pattern lesson
            if len(failures) >= 2:
                root_causes = [_infer_root_cause(e) for e in failures]
                most_common_rc = Counter(root_causes).most_common(1)[0]
                if most_common_rc[0]:  # non-empty root cause
                    el = ExtractedLesson(
                        task=task,
                        pattern=f"Repeated failure across {len(failures)} attempts",
                        root_cause=most_common_rc[0],
                        technique=_find_technique(failures[0]),
                        category=_infer_category(failures[0]),
                        failure_mode=_classify_failure_mode(failures[0]),
                        source_experiences=[e.agent for e in failures],
                        confidence=min(1.0, len(failures) / 5.0),
                        tags=list(
                            set(t for e in failures for t in e.tags)
                        ),
                    )
                    lessons.append(el)

            # Pattern 2: Failure followed by success → technique lesson
            if failures and successes:
                fail_technique = _find_technique(failures[0])
                success_technique = _find_technique(successes[0])
                if fail_technique != success_technique:
                    el = ExtractedLesson(
                        task=task,
                        pattern="Failed technique replaced by successful one",
                        root_cause=_infer_root_cause(failures[0]),
                        technique=success_technique,
                        category=_infer_category(successes[0]),
                        failure_mode=_classify_failure_mode(failures[0]),
                        source_experiences=[e.agent for e in failures + successes],
                        confidence=min(1.0, (len(failures) + len(successes)) / 4.0),
                        tags=list(
                            set(
                                t
                                for e in failures + successes
                                for t in e.tags
                            )
                        ),
                    )
                    lessons.append(el)

            # Pattern 3: Single revealing failure
            if len(failures) == 1 and not successes:
                e = failures[0]
                rc = _infer_root_cause(e)
                if rc:
                    el = ExtractedLesson(
                        task=task,
                        pattern="Single failure with identifiable root cause",
                        root_cause=rc,
                        technique=_find_technique(e),
                        category=_infer_category(e),
                        failure_mode=_classify_failure_mode(e),
                        source_experiences=[e.agent],
                        confidence=0.3,
                        tags=e.tags,
                    )
                    lessons.append(el)

            # Pattern 4: Consistent success → best practice
            if len(successes) >= 3 and not failures:
                el = ExtractedLesson(
                    task=task,
                    pattern=f"Consistent success across {len(successes)} attempts",
                    root_cause="",
                    technique=_find_technique(successes[0]),
                    category=_infer_category(successes[0]),
                    failure_mode=FailureMode.UNKNOWN,
                    source_experiences=[e.agent for e in successes],
                    confidence=min(1.0, len(successes) / 5.0),
                    tags=list(set(t for e in successes for t in e.tags)),
                )
                lessons.append(el)

        return lessons

    def apply_to_library(self, library: LessonLibrary) -> List[Lesson]:
        """Extract lessons and merge them into a LessonLibrary.

        Returns the Lesson objects created or updated.
        """
        extracted = self.extract()
        result: List[Lesson] = []

        for el in extracted:
            lesson = library.get_or_create(
                el.task, category=el.category, applicability=el.tags
            )
            # Record a trial from the extracted knowledge
            lesson.record_trial(
                agent="extractor",
                success=bool(el.technique and "fail" not in el.pattern.lower()[:10]),
                error=el.root_cause,
                technique=el.technique,
                tags=el.tags,
                failure_mode=el.failure_mode,
            )
            # Merge source experience trials if not already present
            for exp in self.experiences:
                if exp.task == el.task and exp.agent not in [
                    t.agent for t in lesson.trials
                ]:
                    lesson.record_trial(
                        agent=exp.agent,
                        success=exp.succeeded,
                        error=_infer_root_cause(exp) if exp.failed else "",
                        technique=_find_technique(exp),
                        tokens_used=exp.total_tokens,
                        duration_sec=exp.total_duration_sec,
                        tags=exp.tags,
                        failure_mode=_classify_failure_mode(exp) if exp.failed else None,
                    )
            result.append(lesson)
        return result

    def _group_by_task(self) -> Dict[str, List[Experience]]:
        grouped: Dict[str, List[Experience]] = {}
        for e in self.experiences:
            grouped.setdefault(e.task, []).append(e)
        return grouped
