"""cocapn_lessons — Trial-based learning for distributed agent fleets.

Every failed execution becomes structured knowledge that future agents
learn from. The math: common-error failure rate drops as O(1/n) where
n = agents who have attempted the task.

Modules:
    lesson      — Lesson, Trial, and LessonLibrary core classes
    experience  — Experience capturing agent interactions and outcomes
    extractor   — LessonExtractor finding patterns in experiences
    curriculum  — Curriculum organizing lessons by topic and difficulty
    transfer    — KnowledgeTransfer applying lessons to new contexts
"""

from cocapn_lessons.lesson import Trial, Lesson, LessonLibrary
from cocapn_lessons.experience import Experience, Outcome
from cocapn_lessons.extractor import LessonExtractor
from cocapn_lessons.curriculum import Curriculum, CurriculumLevel
from cocapn_lessons.transfer import KnowledgeTransfer

__version__ = "3.1.0"
__all__ = [
    "Trial", "Lesson", "LessonLibrary",
    "Experience", "Outcome",
    "LessonExtractor",
    "Curriculum", "CurriculumLevel",
    "KnowledgeTransfer",
]
