"""transfer.py — Apply lessons from one context to another.

KnowledgeTransfer enables cross-context learning: lessons learned
in one domain or fleet can be mapped and applied to new situations,
enabling meta-learning across agent populations.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from cocapn_lessons.lesson import Lesson, Trial, LessonLibrary, Category


@dataclass
class ContextMapping:
    """Maps concepts from a source context to a target context.

    For example, mapping "gh CLI" → "git CLI" or
    "MUD room" → "Dungeon cell".
    """
    source_task: str
    target_task: str
    concept_map: Dict[str, str] = field(default_factory=dict)
    similarity: float = 1.0  # 0.0–1.0 how similar the contexts are

    def translate_text(self, text: str) -> str:
        """Apply concept mapping to translate text."""
        result = text
        for src, tgt in self.concept_map.items():
            result = result.replace(src, tgt)
        return result


@dataclass
class TransferredLesson:
    """A lesson adapted from one context to another."""
    original_task: str
    adapted_task: str
    original_lesson: Lesson
    adapted_lesson: Lesson
    mapping: ContextMapping
    confidence_modifier: float  # multiplied with original confidence

    @property
    def effective_confidence(self) -> float:
        return self.original_lesson.confidence * self.confidence_modifier


class KnowledgeTransfer:
    """Apply lessons learned in one context to new situations.

    Workflow:
        1. Define ContextMappings between source and target contexts.
        2. Call ``transfer_lesson()`` or ``transfer_library()`` to adapt lessons.
        3. The adapted lessons have translated tasks, techniques, and errors.

    Usage:
        kt = KnowledgeTransfer()
        mapping = ContextMapping(
            source_task="README detection via gh CLI",
            target_task="README detection via git CLI",
            concept_map={"gh repo view": "git show", "gh api": "git cat-file"},
            similarity=0.7,
        )
        result = kt.transfer_lesson(source_lesson, mapping)
    """

    def __init__(self):
        self.mappings: List[ContextMapping] = []
        self.transferred: List[TransferredLesson] = []

    def add_mapping(self, mapping: ContextMapping):
        self.mappings.append(mapping)

    def transfer_lesson(
        self,
        lesson: Lesson,
        mapping: ContextMapping,
        confidence_floor: float = 0.1,
    ) -> TransferredLesson:
        """Adapt a lesson to a new context using a mapping.

        Translates task name, error messages, techniques, and root causes.
        Adjusts confidence based on context similarity.
        """
        # Create adapted lesson
        adapted = Lesson(
            task=mapping.translate_text(lesson.task),
            description=mapping.translate_text(lesson.description),
            category=lesson.category,
            applicability=[
                mapping.translate_text(a) for a in lesson.applicability
            ],
        )

        # Translate trials
        for trial in lesson.trials:
            adapted.record_trial(
                agent=trial.agent,
                success=trial.success,
                error=mapping.translate_text(trial.error) if trial.error else "",
                technique=mapping.translate_text(trial.technique) if trial.technique else "",
                tokens_used=trial.tokens_used,
                duration_sec=trial.duration_sec,
                tags=[mapping.translate_text(t) for t in trial.tags],
                failure_mode=trial.failure_mode,
                root_cause=mapping.translate_text(trial.root_cause) if trial.root_cause else "",
            )

        # Confidence is reduced by context distance
        confidence_modifier = max(mapping.similarity, confidence_floor)
        result = TransferredLesson(
            original_task=lesson.task,
            adapted_task=adapted.task,
            original_lesson=lesson,
            adapted_lesson=adapted,
            mapping=mapping,
            confidence_modifier=confidence_modifier,
        )
        self.transferred.append(result)
        return result

    def transfer_library(
        self,
        source: LessonLibrary,
        mappings: Optional[List[ContextMapping]] = None,
    ) -> LessonLibrary:
        """Transfer all matching lessons from source library.

        Uses provided mappings or previously added mappings.
        Returns a new library with adapted lessons.
        """
        mappings = mappings or self.mappings
        target = LessonLibrary()

        mapping_by_source = {m.source_task: m for m in mappings}

        for task, lesson in source.lessons.items():
            mapping = mapping_by_source.get(task)
            if mapping:
                transferred = self.transfer_lesson(lesson, mapping)
                target.lessons[mapping.target_task] = transferred.adapted_lesson
            else:
                # Try auto-mapping by category similarity
                target.lessons[task] = lesson

        return target

    def find_applicable(
        self,
        target_task: str,
        library: LessonLibrary,
        min_confidence: float = 0.3,
        max_results: int = 5,
    ) -> List[TransferredLesson]:
        """Find lessons from the library that could apply to a target task.

        Scores by keyword overlap between lesson tasks and the target task.
        """
        target_words = set(target_task.lower().split())
        candidates: List[tuple] = []  # (score, lesson, mapping)

        for task, lesson in library.lessons.items():
            if not lesson.trials:
                continue
            task_words = set(task.lower().split())
            overlap = len(target_words & task_words)
            if overlap == 0:
                continue

            similarity = overlap / max(len(target_words | task_words), 1)
            confidence = lesson.confidence * similarity
            if confidence < min_confidence:
                continue

            mapping = ContextMapping(
                source_task=task,
                target_task=target_task,
                similarity=similarity,
            )
            candidates.append((confidence, lesson, mapping))

        candidates.sort(key=lambda x: -x[0])
        results = []
        for confidence, lesson, mapping in candidates[:max_results]:
            results.append(self.transfer_lesson(lesson, mapping))
        return results

    def transfer_summary(self) -> str:
        """Human-readable summary of all transfers performed."""
        if not self.transferred:
            return "No transfers yet."
        lines = [f"KnowledgeTransfer: {len(self.transferred)} transfers"]
        for t in self.transferred:
            lines.append(
                f"  {t.original_task} → {t.adapted_task} "
                f"(confidence: {t.effective_confidence:.0%})"
            )
        return "\n".join(lines)
