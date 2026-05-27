"""curriculum.py — Organize lessons by topic and difficulty.

A Curriculum groups Lessons into topics with difficulty levels,
enabling mastery-based progression: agents must demonstrate competence
at lower levels before advancing.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from enum import Enum

from cocapn_lessons.lesson import Lesson, Category, LessonLibrary


class CurriculumLevel(Enum):
    """Difficulty level for a lesson within a curriculum."""
    BEGINNER = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    EXPERT = 4


@dataclass
class Topic:
    """A curriculum topic containing related lessons at a difficulty level."""
    name: str
    level: CurriculumLevel
    category: Category
    lesson_tasks: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)  # topic names
    description: str = ""

    @property
    def num_lessons(self) -> int:
        return len(self.lesson_tasks)


@dataclass
class Curriculum:
    """Organized collection of lessons grouped by topic and difficulty.

    Supports mastery-based progression: agents must pass lower-level
    topics before accessing higher-level ones.

    Usage:
        curriculum = Curriculum("Agent Onboarding")
        curriculum.add_topic("Basic CLI", CurriculumLevel.BEGINNER, Category.TOOLING)
        curriculum.assign_lesson("Basic CLI", "README detection")
        ready = curriculum.available_topics(passed={"Basic CLI"})
    """
    name: str
    description: str = ""
    topics: Dict[str, Topic] = field(default_factory=dict)

    def add_topic(
        self,
        name: str,
        level: CurriculumLevel,
        category: Category = Category.GENERAL,
        prerequisites: List[str] = None,
        description: str = "",
    ) -> Topic:
        """Create and register a new topic."""
        topic = Topic(
            name=name,
            level=level,
            category=category,
            prerequisites=prerequisites or [],
            description=description,
        )
        self.topics[name] = topic
        return topic

    def assign_lesson(self, topic_name: str, lesson_task: str):
        """Assign a lesson to a topic."""
        if topic_name not in self.topics:
            raise KeyError(f"Topic '{topic_name}' not found")
        self.topics[topic_name].lesson_tasks.append(lesson_task)

    def remove_topic(self, name: str):
        """Remove a topic and unassign its lessons."""
        self.topics.pop(name, None)

    def get_topic(self, name: str) -> Optional[Topic]:
        return self.topics.get(name)

    def topics_at_level(self, level: CurriculumLevel) -> List[Topic]:
        """Get all topics at a given difficulty level."""
        return [t for t in self.topics.values() if t.level == level]

    def available_topics(self, passed: Set[str] = None) -> List[Topic]:
        """Topics whose prerequisites are all met.

        Args:
            passed: Set of topic names the agent has mastered.
        """
        passed = passed or set()
        available = []
        for t in self.topics.values():
            if all(p in passed for p in t.prerequisites):
                available.append(t)
        return available

    def locked_topics(self, passed: Set[str] = None) -> List[Topic]:
        """Topics whose prerequisites are NOT all met."""
        passed = passed or set()
        return [
            t
            for t in self.topics.values()
            if not all(p in passed for p in t.prerequisites)
        ]

    def learning_path(self) -> List[Topic]:
        """Return topics in a sensible learning order (low to high difficulty)."""
        return sorted(
            self.topics.values(),
            key=lambda t: (t.level.value, t.name),
        )

    def lessons_for_topic(
        self, topic_name: str, library: LessonLibrary
    ) -> List[Lesson]:
        """Resolve topic's lesson_tasks into actual Lesson objects from a library."""
        topic = self.topics.get(topic_name)
        if not topic:
            return []
        lessons = []
        for task in topic.lesson_tasks:
            if task in library.lessons:
                lessons.append(library.lessons[task])
        return lessons

    def mastery_check(
        self,
        topic_name: str,
        library: LessonLibrary,
        min_success_rate: float = 0.8,
        min_confidence: float = 0.5,
    ) -> Dict[str, bool]:
        """Check which lessons in a topic meet mastery thresholds."""
        lessons = self.lessons_for_topic(topic_name, library)
        return {
            l.task: l.success_rate >= min_success_rate and l.confidence >= min_confidence
            for l in lessons
        }

    def topic_summary(self) -> str:
        """Human-readable summary of the curriculum."""
        lines = [f"Curriculum: {self.name}"]
        if self.description:
            lines.append(f"  {self.description}")
        lines.append(f"  Topics: {len(self.topics)}")
        for level in CurriculumLevel:
            at_level = self.topics_at_level(level)
            if at_level:
                lines.append(
                    f"  {level.name} ({len(at_level)}): "
                    + ", ".join(t.name for t in at_level)
                )
        return "\n".join(lines)

    def stats(self) -> Dict:
        total_lessons = sum(t.num_lessons for t in self.topics.values())
        by_level = {}
        for level in CurriculumLevel:
            topics = self.topics_at_level(level)
            by_level[level.name] = {
                "topics": len(topics),
                "lessons": sum(t.num_lessons for t in topics),
            }
        return {
            "name": self.name,
            "total_topics": len(self.topics),
            "total_lessons": total_lessons,
            "by_level": by_level,
        }
