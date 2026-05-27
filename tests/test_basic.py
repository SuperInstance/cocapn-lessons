"""Tests for cocapn_lessons package."""
import os
import json
import tempfile
import pytest

from cocapn_lessons.lesson import (
    Trial, Lesson, LessonLibrary, FailureMode, Category,
)
from cocapn_lessons.experience import (
    Experience, Outcome, Interaction, InteractionType,
)
from cocapn_lessons.extractor import LessonExtractor
from cocapn_lessons.curriculum import Curriculum, CurriculumLevel, Topic
from cocapn_lessons.transfer import KnowledgeTransfer, ContextMapping


# ── Trial tests ──────────────────────────────────────────────────────────

class TestTrial:
    def test_fingerprint_deterministic(self):
        t1 = Trial(task="x", error="e")
        t2 = Trial(task="x", error="e")
        assert t1.fingerprint() == t2.fingerprint()

    def test_fingerprint_differs_for_different_errors(self):
        t1 = Trial(task="x", error="e1")
        t2 = Trial(task="x", error="e2")
        assert t1.fingerprint() != t2.fingerprint()

    def test_default_values(self):
        t = Trial(task="test")
        assert t.agent == "unknown"
        assert t.success is False
        assert t.tokens_used == 0
        assert t.failure_mode is None


# ── Lesson tests ─────────────────────────────────────────────────────────

class TestLesson:
    def test_empty_lesson(self):
        l = Lesson(task="empty")
        assert l.total_attempts == 0
        assert l.success_rate == 0.0
        assert l.unique_agents == 0
        assert l.confidence == 0.0

    def test_record_trial(self):
        l = Lesson(task="test")
        t = l.record_trial(agent="a1", success=True, technique="fast")
        assert t.task == "test"
        assert t.agent == "a1"
        assert t.success is True
        assert l.total_attempts == 1
        assert l.success_rate == 1.0

    def test_success_rate_mixed(self):
        l = Lesson(task="test")
        l.record_trial(success=True)
        l.record_trial(success=False)
        l.record_trial(success=True)
        assert l.success_rate == pytest.approx(2 / 3)

    def test_unique_agents(self):
        l = Lesson(task="test")
        l.record_trial(agent="a")
        l.record_trial(agent="b")
        l.record_trial(agent="a")
        assert l.unique_agents == 2

    def test_confidence_high_consistency(self):
        l = Lesson(task="test", category=Category.TOOLING)
        for _ in range(10):
            l.record_trial(success=True)
        # All successes → high consistency + good sample
        assert l.confidence > 0.8

    def test_confidence_low_consistency(self):
        l = Lesson(task="test")
        for i in range(10):
            l.record_trial(success=(i % 2 == 0))
        # 50/50 → consistency near 0
        assert l.confidence < 0.3

    def test_confidence_few_trials(self):
        l = Lesson(task="test")
        l.record_trial(success=True)
        assert l.confidence < 0.5

    def test_failure_modes(self):
        l = Lesson(task="test")
        l.record_trial(success=False, error="timeout")
        l.record_trial(success=False, error="timeout")
        l.record_trial(success=True, error="")
        modes = l.failure_modes(top_n=3)
        assert len(modes) >= 2
        # "timeout" should be the most common
        assert modes[0][0] == "timeout"
        assert modes[0][1] == 2

    def test_advice_for_new_agent(self):
        l = Lesson(task="README detection")
        l.record_trial(agent="a1", success=False, technique="gh repo view", error="fails 95%")
        l.record_trial(agent="a2", success=True, technique="gh api repos/.../readme")
        advice = l.advice_for_new_agent()
        assert "README detection" in advice
        assert "gh api repos/.../readme" in advice
        assert "Avoid" in advice

    def test_advice_empty_lesson(self):
        l = Lesson(task="nothing")
        assert "No trials yet" in l.advice_for_new_agent()

    def test_predict_failure_rate(self):
        l = Lesson(task="test")
        for _ in range(5):
            l.record_trial(success=True)
        # 100% success → 0 predicted failure
        assert l.predict_failure_rate(10) == 0.0

    def test_predict_failure_rate_uncertain(self):
        l = Lesson(task="test")
        l.record_trial(success=False)
        l.record_trial(success=False)
        # Very few failures, low data → high uncertainty
        assert l.predict_failure_rate(5) > 0.5

    def test_winning_technique(self):
        l = Lesson(task="test")
        l.record_trial(success=False, technique="bad")
        l.record_trial(success=True, technique="good")
        assert l.winning_technique() == "good"

    def test_winning_technique_none(self):
        l = Lesson(task="test")
        l.record_trial(success=False, technique="bad")
        assert l.winning_technique() is None

    def test_category_enum(self):
        l = Lesson(task="test", category=Category.API)
        assert l.category == Category.API

    def test_applicability(self):
        l = Lesson(task="test", applicability=["github", "cli"])
        assert "github" in l.applicability

    def test_save_and_load(self, tmp_path):
        l = Lesson(task="save/load test", category=Category.TOOLING)
        l.record_trial(
            agent="bot", success=True, technique="magic",
            failure_mode=FailureMode.UNKNOWN,
        )
        path = str(tmp_path / "lesson.json")
        l.save(path)

        loaded = Lesson.load(path)
        assert loaded.task == "save/load test"
        assert loaded.category == Category.TOOLING
        assert len(loaded.trials) == 1
        assert loaded.trials[0].agent == "bot"
        assert loaded.trials[0].success is True

    def test_save_preserves_failure_mode(self, tmp_path):
        l = Lesson(task="fm test")
        l.record_trial(failure_mode=FailureMode.TIMEOUT)
        path = str(tmp_path / "fm.json")
        l.save(path)
        loaded = Lesson.load(path)
        assert loaded.trials[0].failure_mode == FailureMode.TIMEOUT


# ── LessonLibrary tests ──────────────────────────────────────────────────

class TestLessonLibrary:
    def test_get_or_create(self):
        lib = LessonLibrary()
        l1 = lib.get_or_create("task1")
        l2 = lib.get_or_create("task1")
        assert l1 is l2
        assert len(lib.lessons) == 1

    def test_get_or_create_with_kwargs(self):
        lib = LessonLibrary()
        l = lib.get_or_create("task", category=Category.API)
        assert l.category == Category.API

    def test_search_by_tag(self):
        lib = LessonLibrary()
        l = lib.get_or_create("tagged")
        l.record_trial(tags=["github", "cli"])
        results = lib.search(tag="github")
        assert len(results) == 1
        assert results[0].task == "tagged"

    def test_search_by_category(self):
        lib = LessonLibrary()
        lib.get_or_create("api_task", category=Category.API)
        lib.get_or_create("tool_task", category=Category.TOOLING)
        results = lib.search(category=Category.API)
        assert len(results) == 1

    def test_search_by_confidence(self):
        lib = LessonLibrary()
        l = lib.get_or_create("confident")
        for _ in range(10):
            l.record_trial(success=True)
        lib.get_or_create("weak").record_trial(success=True)
        results = lib.search(min_confidence=0.5)
        assert any(lesson.task == "confident" for lesson in results)

    def test_fleet_stats(self):
        lib = LessonLibrary()
        l1 = lib.get_or_create("t1")
        l1.record_trial(agent="a")
        l1.record_trial(agent="b", success=True)
        stats = lib.fleet_stats()
        assert stats["lessons"] == 1
        assert stats["trials"] == 2
        assert stats["unique_agents"] == 2

    def test_fleet_stats_empty(self):
        lib = LessonLibrary()
        stats = lib.fleet_stats()
        assert stats["lessons"] == 0
        assert stats["trials"] == 0


# ── Experience tests ─────────────────────────────────────────────────────

class TestExperience:
    def test_add_interaction(self):
        e = Experience(task="test", agent="bot")
        i = e.add_interaction(
            type=InteractionType.CLI_COMMAND,
            description="ran gh repo view",
            result="success",
        )
        assert i.type == InteractionType.CLI_COMMAND
        assert e.num_interactions == 1
        assert e.total_tokens == 0

    def test_interaction_accumulates_tokens(self):
        e = Experience(task="test")
        e.add_interaction(type=InteractionType.API_CALL, description="x", tokens_used=100, duration_sec=2.0)
        e.add_interaction(type=InteractionType.API_CALL, description="y", tokens_used=50, duration_sec=1.0)
        assert e.total_tokens == 150
        assert e.total_duration_sec == 3.0

    def test_outcome_properties(self):
        e = Experience(task="test", outcome=Outcome.SUCCESS)
        assert e.succeeded is True
        assert e.failed is False

        e.outcome = Outcome.FAILURE
        assert e.succeeded is False
        assert e.failed is True

    def test_failed_interactions(self):
        e = Experience(task="test")
        e.add_interaction(type=InteractionType.CLI_COMMAND, description="ok", result="ok")
        e.add_interaction(type=InteractionType.CLI_COMMAND, description="bad", result="error: timeout")
        assert len(e.failed_interactions) == 1

    def test_interaction_summary(self):
        e = Experience(task="test")
        e.add_interaction(type=InteractionType.CLI_COMMAND, description="a")
        e.add_interaction(type=InteractionType.CLI_COMMAND, description="b")
        e.add_interaction(type=InteractionType.API_CALL, description="c")
        summary = e.interaction_summary
        assert summary["cli_command"] == 2
        assert summary["api_call"] == 1

    def test_summary_string(self):
        e = Experience(task="do stuff", agent="bot", outcome=Outcome.SUCCESS, tags=["test"])
        e.add_interaction(type=InteractionType.TOOL_USE, description="used tool")
        s = e.summary()
        assert "do stuff" in s
        assert "bot" in s
        assert "success" in s

    def test_add_observation(self):
        e = Experience(task="test")
        e.add_observation("this is slow")
        assert e.observations == ["this is slow"]


# ── LessonExtractor tests ────────────────────────────────────────────────

class TestLessonExtractor:
    def _make_experience(self, task, agent, outcome, technique_desc="did stuff", error_result=""):
        e = Experience(task=task, agent=agent, outcome=outcome, tags=["test"])
        e.add_interaction(
            type=InteractionType.CLI_COMMAND,
            description=technique_desc,
            result=error_result or "ok",
        )
        return e

    def test_repeated_failure_pattern(self):
        ext = LessonExtractor()
        ext.add_experience(self._make_experience("task1", "a1", Outcome.FAILURE, "bad method", "error: timeout"))
        ext.add_experience(self._make_experience("task1", "a2", Outcome.FAILURE, "bad method", "error: timeout"))
        lessons = ext.extract()
        assert len(lessons) >= 1
        assert any("Repeated failure" in l.pattern for l in lessons)

    def test_failure_then_success_pattern(self):
        ext = LessonExtractor()
        ext.add_experience(self._make_experience("task1", "a1", Outcome.FAILURE, "old way", "error"))
        ext.add_experience(self._make_experience("task1", "a2", Outcome.SUCCESS, "new way"))
        lessons = ext.extract()
        assert any("Failed technique replaced" in l.pattern for l in lessons)

    def test_single_failure_pattern(self):
        ext = LessonExtractor()
        ext.add_experience(self._make_experience("solo", "a1", Outcome.FAILURE, "tried X", "error: bad"))
        lessons = ext.extract()
        assert any("Single failure" in l.pattern for l in lessons)

    def test_consistent_success_pattern(self):
        ext = LessonExtractor()
        for i in range(4):
            ext.add_experience(self._make_experience("easy", f"a{i}", Outcome.SUCCESS, "works"))
        lessons = ext.extract()
        assert any("Consistent success" in l.pattern for l in lessons)

    def test_apply_to_library(self):
        ext = LessonExtractor()
        ext.add_experience(self._make_experience("task1", "a1", Outcome.FAILURE, "bad", "error: timeout"))
        ext.add_experience(self._make_experience("task1", "a2", Outcome.FAILURE, "bad", "error: timeout"))
        lib = LessonLibrary()
        result = ext.apply_to_library(lib)
        assert len(result) >= 1
        assert "task1" in lib.lessons
        assert lib.lessons["task1"].total_attempts >= 2

    def test_empty_extractor(self):
        ext = LessonExtractor()
        assert ext.extract() == []


# ── Curriculum tests ─────────────────────────────────────────────────────

class TestCurriculum:
    def test_add_topic(self):
        c = Curriculum("Test")
        t = c.add_topic("Basics", CurriculumLevel.BEGINNER, Category.TOOLING)
        assert t.name == "Basics"
        assert t.level == CurriculumLevel.BEGINNER
        assert "Basics" in c.topics

    def test_assign_lesson(self):
        c = Curriculum("Test")
        c.add_topic("Basics", CurriculumLevel.BEGINNER)
        c.assign_lesson("Basics", "lesson1")
        assert c.topics["Basics"].lesson_tasks == ["lesson1"]

    def test_assign_lesson_unknown_topic(self):
        c = Curriculum("Test")
        with pytest.raises(KeyError):
            c.assign_lesson("Nonexistent", "lesson1")

    def test_topics_at_level(self):
        c = Curriculum("Test")
        c.add_topic("B1", CurriculumLevel.BEGINNER)
        c.add_topic("B2", CurriculumLevel.BEGINNER)
        c.add_topic("A1", CurriculumLevel.ADVANCED)
        assert len(c.topics_at_level(CurriculumLevel.BEGINNER)) == 2
        assert len(c.topics_at_level(CurriculumLevel.ADVANCED)) == 1

    def test_available_topics_no_prereqs(self):
        c = Curriculum("Test")
        c.add_topic("B1", CurriculumLevel.BEGINNER)
        c.add_topic("B2", CurriculumLevel.BEGINNER)
        avail = c.available_topics()
        assert len(avail) == 2

    def test_available_topics_with_prereqs(self):
        c = Curriculum("Test")
        c.add_topic("B1", CurriculumLevel.BEGINNER)
        c.add_topic("I1", CurriculumLevel.INTERMEDIATE, prerequisites=["B1"])
        # Nothing passed → only B1 available
        avail = c.available_topics(passed=set())
        assert len(avail) == 1
        assert avail[0].name == "B1"
        # B1 passed → both available
        avail = c.available_topics(passed={"B1"})
        assert len(avail) == 2

    def test_locked_topics(self):
        c = Curriculum("Test")
        c.add_topic("B1", CurriculumLevel.BEGINNER)
        c.add_topic("I1", CurriculumLevel.INTERMEDIATE, prerequisites=["B1"])
        locked = c.locked_topics(passed=set())
        assert len(locked) == 1
        assert locked[0].name == "I1"

    def test_learning_path(self):
        c = Curriculum("Test")
        c.add_topic("Advanced", CurriculumLevel.ADVANCED)
        c.add_topic("Beginner", CurriculumLevel.BEGINNER)
        path = c.learning_path()
        assert path[0].level == CurriculumLevel.BEGINNER
        assert path[1].level == CurriculumLevel.ADVANCED

    def test_lessons_for_topic(self):
        c = Curriculum("Test")
        c.add_topic("Basics", CurriculumLevel.BEGINNER)
        c.assign_lesson("Basics", "task1")
        lib = LessonLibrary()
        lib.get_or_create("task1").record_trial(success=True)
        lessons = c.lessons_for_topic("Basics", lib)
        assert len(lessons) == 1
        assert lessons[0].task == "task1"

    def test_mastery_check(self):
        c = Curriculum("Test")
        c.add_topic("Basics", CurriculumLevel.BEGINNER)
        c.assign_lesson("Basics", "task1")
        lib = LessonLibrary()
        l = lib.get_or_create("task1")
        for _ in range(10):
            l.record_trial(success=True)
        check = c.mastery_check("Basics", lib)
        assert check["task1"] is True

    def test_topic_summary(self):
        c = Curriculum("Test", description="A test curriculum")
        c.add_topic("Basics", CurriculumLevel.BEGINNER)
        s = c.topic_summary()
        assert "Test" in s
        assert "BEGINNER" in s

    def test_stats(self):
        c = Curriculum("Test")
        c.add_topic("B1", CurriculumLevel.BEGINNER)
        c.assign_lesson("B1", "l1")
        c.assign_lesson("B1", "l2")
        stats = c.stats()
        assert stats["total_topics"] == 1
        assert stats["total_lessons"] == 2
        assert stats["by_level"]["BEGINNER"]["lessons"] == 2

    def test_remove_topic(self):
        c = Curriculum("Test")
        c.add_topic("B1", CurriculumLevel.BEGINNER)
        c.remove_topic("B1")
        assert "B1" not in c.topics

    def test_get_topic(self):
        c = Curriculum("Test")
        c.add_topic("B1", CurriculumLevel.BEGINNER)
        assert c.get_topic("B1") is not None
        assert c.get_topic("nope") is None


# ── KnowledgeTransfer tests ──────────────────────────────────────────────

class TestKnowledgeTransfer:
    def _make_lesson(self, task, n_success=3, n_fail=1, technique="do_thing"):
        l = Lesson(task=task, category=Category.TOOLING, applicability=["cli"])
        for _ in range(n_fail):
            l.record_trial(agent="failer", success=False, technique="wrong", error="bad error")
        for _ in range(n_success):
            l.record_trial(agent="worker", success=True, technique=technique)
        return l

    def test_transfer_lesson(self):
        kt = KnowledgeTransfer()
        lesson = self._make_lesson("gh CLI task")
        mapping = ContextMapping(
            source_task="gh CLI task",
            target_task="git CLI task",
            concept_map={"gh": "git"},
            similarity=0.8,
        )
        result = kt.transfer_lesson(lesson, mapping)
        assert result.adapted_task == "git CLI task"
        assert result.adapted_lesson.task == "git CLI task"
        assert result.effective_confidence > 0
        assert len(result.adapted_lesson.trials) == 4

    def test_concept_mapping(self):
        m = ContextMapping(
            source_task="a", target_task="b",
            concept_map={"old": "new", "foo": "bar"},
        )
        assert m.translate_text("old foo baz") == "new bar baz"

    def test_transfer_translates_techniques(self):
        kt = KnowledgeTransfer()
        lesson = self._make_lesson("gh detection", technique="gh api repos")
        mapping = ContextMapping(
            source_task="gh detection",
            target_task="git detection",
            concept_map={"gh api": "git cat-file"},
            similarity=0.7,
        )
        result = kt.transfer_lesson(lesson, mapping)
        winning = result.adapted_lesson.winning_technique()
        assert "git cat-file" in winning

    def test_transfer_reduces_confidence(self):
        kt = KnowledgeTransfer()
        lesson = self._make_lesson("task")
        mapping = ContextMapping(
            source_task="task", target_task="other",
            similarity=0.5,
        )
        result = kt.transfer_lesson(lesson, mapping)
        assert result.effective_confidence < lesson.confidence

    def test_transfer_library(self):
        kt = KnowledgeTransfer()
        lib = LessonLibrary()
        lib.lessons["task1"] = self._make_lesson("task1")
        mapping = ContextMapping(
            source_task="task1", target_task="task1_prime", similarity=0.9,
        )
        new_lib = kt.transfer_library(lib, mappings=[mapping])
        assert "task1_prime" in new_lib.lessons

    def test_find_applicable(self):
        kt = KnowledgeTransfer()
        lib = LessonLibrary()
        lesson = self._make_lesson("README detection via gh CLI", technique="gh api")
        # Need enough trials for decent confidence
        for _ in range(5):
            lesson.record_trial(agent="extra", success=True, technique="gh api", tags=["test"])
        lib.lessons["README detection via gh CLI"] = lesson
        results = kt.find_applicable("README detection via git CLI", lib, min_confidence=0.1)
        assert len(results) >= 1
        assert "README" in results[0].adapted_task

    def test_find_applicable_no_match(self):
        kt = KnowledgeTransfer()
        lib = LessonLibrary()
        lib.lessons["cooking pasta"] = self._make_lesson("cooking pasta")
        results = kt.find_applicable("deploying rockets", lib)
        assert len(results) == 0

    def test_transfer_summary(self):
        kt = KnowledgeTransfer()
        lesson = self._make_lesson("src")
        mapping = ContextMapping(source_task="src", target_task="tgt", similarity=0.9)
        kt.transfer_lesson(lesson, mapping)
        s = kt.transfer_summary()
        assert "1 transfers" in s
        assert "src" in s

    def test_transfer_summary_empty(self):
        kt = KnowledgeTransfer()
        assert "No transfers" in kt.transfer_summary()


# ── Integration tests ────────────────────────────────────────────────────

class TestIntegration:
    def test_full_workflow(self):
        """End-to-end: create experiences → extract → build curriculum → transfer."""
        # 1. Create experiences
        experiences = []
        for agent in ["a1", "a2", "a3", "a4", "a5"]:
            e = Experience(
                task="README detection via gh CLI",
                agent=agent,
                tags=["github", "readme"],
            )
            if agent == "a1":
                e.outcome = Outcome.FAILURE
                e.add_interaction(
                    type=InteractionType.CLI_COMMAND,
                    description="gh repo view --readme",
                    result="error: fails in non-interactive mode",
                    tokens_used=8000,
                )
            else:
                e.outcome = Outcome.SUCCESS
                e.add_interaction(
                    type=InteractionType.API_CALL,
                    description="gh api repos/{o}/{r}/readme",
                    result="ok",
                    tokens_used=3000,
                )
            experiences.append(e)

        # 2. Extract lessons
        ext = LessonExtractor()
        ext.add_experiences(experiences)
        lib = LessonLibrary()
        ext.apply_to_library(lib)

        lesson = lib.lessons["README detection via gh CLI"]
        assert lesson.total_attempts >= 3
        assert lesson.success_rate >= 0.5

        # 3. Build curriculum
        curr = Curriculum("Fleet Onboarding")
        curr.add_topic("CLI Tools", CurriculumLevel.BEGINNER, Category.TOOLING)
        curr.assign_lesson("CLI Tools", "README detection via gh CLI")

        check = curr.mastery_check("CLI Tools", lib, min_success_rate=0.5, min_confidence=0.1)
        assert check["README detection via gh CLI"] is True

        # 4. Transfer knowledge
        kt = KnowledgeTransfer()
        mapping = ContextMapping(
            source_task="README detection via gh CLI",
            target_task="README detection via git CLI",
            concept_map={"gh": "git"},
            similarity=0.7,
        )
        transferred = kt.transfer_lesson(lesson, mapping)
        assert "git CLI" in transferred.adapted_task
        assert transferred.effective_confidence > 0

    def test_import_all(self):
        """Verify package-level imports work."""
        from cocapn_lessons import (
            Trial, Lesson, LessonLibrary,
            Experience, Outcome,
            LessonExtractor,
            Curriculum, CurriculumLevel,
            KnowledgeTransfer,
        )
        assert Trial is not None
        assert Lesson is not None
