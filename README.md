# cocapn-lessons — Trial-Based Learning for Agent Fleets

**Every failure becomes a lesson. Common-error failure rate drops as O(1/n) where n = agents who attempted the task.**

## What This Gives You

- **Trials** — structured records of agent attempts (success or failure) with failure mode classification
- **Lessons** — compiled knowledge from multiple trials, retrievable by category and difficulty
- **Experience capture** — record agent interactions and outcomes automatically
- **Curriculum** — organize lessons by topic and difficulty for progressive learning
- **Knowledge transfer** — apply lessons from one agent/context to new situations
- **Proven math** — fleet with trial recording achieves O(1/n) failure rate vs O(1) without

## Quick Start

```bash
pip install cocapn-lessons
```

```python
from cocapn_lessons import Trial, Lesson, LessonLibrary, Experience, Curriculum

# Record a failed trial
trial = Trial(
    task="deploy to staging",
    agent_id="agent-3",
    success=False,
    failure_mode="timeout",
    context={"timeout_seconds": 30, "service": "api-gateway"},
    error_details="Connection timed out after 30s"
)

# Extract lessons from experiences
from cocapn_lessons import LessonExtractor
lib = LessonLibrary()
extractor = LessonExtractor()
experience = Experience(agent_id="agent-3", trials=[trial])
lesson = extractor.extract(experience)
lib.add(lesson)

# Build a curriculum
curriculum = Curriculum(name="deployment")
curriculum.add_lesson(lesson, level=CurriculumLevel.INTERMEDIATE)

# Transfer knowledge to a new agent
from cocapn_lessons import KnowledgeTransfer
transfer = KnowledgeTransfer()
applicable = transfer.find_applicable(lib, context={"task": "deploy to production"})
for l in applicable:
    print(f"Lesson: {l.title} — {l.summary}")
```

## API Reference

### Core
- **`Trial`** — Single attempt record with failure mode, context, error details
- **`Lesson`** — Compiled knowledge from trials, with confidence scoring
- **`LessonLibrary`** — Searchable collection of lessons
- **`FailureMode`** — Enum: `TIMEOUT`, `ERROR`, `REJECTION`, `WRONG_RESULT`, `CRASH`, `RESOURCE_EXHAUSTED`, etc.
- **`Category`** — Enum: `TOOLING`, `NAVIGATION`, `API`, `STRATEGY`, `RESOURCE_MANAGEMENT`, etc.

### Processing
- **`Experience`** — Agent interaction record with `Outcome` and trials
- **`LessonExtractor`** — Finds patterns in experiences and creates lessons
- **`Curriculum`** / `CurriculumLevel` — Organized lesson sequences
- **`KnowledgeTransfer`** — Cross-agent, cross-context lesson application

## How It Fits

The pedagogical engine of the [SuperInstance fleet](https://github.com/SuperInstance). Inspired by PLATO's unit-based pedagogy and spaced repetition — agents get smarter as a fleet, not just individually.

- **[cocapn](https://github.com/SuperInstance/cocapn)** — Core infrastructure (rooms train on lessons)
- **[cocapn-explain](https://github.com/SuperInstance/cocapn-explain)** — Explainability (lessons feed traces)
- **[cocapn-dreamer](https://github.com/SuperInstance/cocapn-dreamer)** — Speculative execution (avoids known failures)

## Testing

```bash
pytest tests/
```

## Installation

```bash
pip install cocapn-lessons
```

Python 3.10+. MIT license.
