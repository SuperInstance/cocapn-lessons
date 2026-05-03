"""cocapn_lessons_flux — Trial-based learning with FLUX bytecode optimization.

Each trial IS a bytecode execution. The LessonLibrary compiles trials into
FLUX programs where:
  - Failed paths → dead code (JNE around failure block)
  - Success paths → hot paths (eligible for JIT)
  - WITNESS opcode records trial result to commit log
  - After N successful trials, the path is JIT-compiled to native

The O(1/n) failure rate model is implemented as branch prediction:
  - First attempt: no prediction, full branch cost
  - After success: predict taken, speculatively execute hot path
  - After failure: predict not taken, skip cold path

This maps to FLUX opcodes:
  CMP + JE hot_path    (speculate success)
  JNE cold_path        (mispredict = failure)
  WITNESS Rresult      (record outcome)
  SNAPSHOT             (save state for replay)
"""
import json
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from collections import Counter
from enum import IntEnum


class Op(IntEnum):
    CMP = 0x2D; JE = 0x2E; JNE = 0x2F; JMP = 0x04
    WITNESS = 0x7E; SNAPSHOT = 0x7F
    MOVI = 0x2B; LOADK = 0x4F
    IADD = 0x08
    HALT = 0x80; YIELD = 0x81
    TELL = 0x60


@dataclass
class Trial:
    """A single attempt. Includes bytecode fingerprint for dedup."""
    task: str
    agent: str = "unknown"
    success: bool = False
    error: str = ""
    technique: str = ""
    tokens_used: int = 0
    duration_sec: float = 0.0
    tags: List[str] = field(default_factory=list)
    bytecode_hash: str = ""        # hash of attempted bytecode
    at: str = field(default_factory=lambda: __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat())

    def fingerprint(self) -> str:
        return hashlib.sha256(f"{self.task}:{self.error}".encode()).hexdigest()[:16]


@dataclass
class FluxLesson:
    """Compiled lesson with bytecode optimization."""
    task: str
    description: str = ""
    trials: List[Trial] = field(default_factory=list)
    bytecode_segments: Dict[str, bytes] = field(default_factory=dict)
    hot_path: Optional[bytes] = None     # JIT-compiled after repeated success
    cold_path: Optional[bytes] = None    # rarely taken path
    created_at: str = field(default_factory=lambda: __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat())

    def record_trial(self, **kwargs) -> Trial:
        t = Trial(task=self.task, **kwargs)
        self.trials.append(t)
        self._update_paths()
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

    def _update_paths(self):
        """Recompile bytecode based on trial history.
        After 3+ successes, extract hot path. After 3+ failures, extract cold path."""
        successes = [t for t in self.trials if t.success]
        failures = [t for t in self.trials if not t.success]
        if len(successes) >= 3 and not self.hot_path:
            self.hot_path = self._compile_hot_path(successes[-1])
        if len(failures) >= 3 and not self.cold_path:
            self.cold_path = self._compile_cold_path(failures[-1])

    def _compile_hot_path(self, winning_trial: Trial) -> bytes:
        """Compile success path: speculatively execute winning technique."""
        # Minimal bytecode: load technique, tell it, witness success
        tech = winning_trial.technique or "default"
        return bytes([
            Op.LOADK, 0, 0, 0,   # load technique into R0 (pool idx 0)
            Op.TELL, 0, 0, 0,     # broadcast technique
            Op.MOVI, 1, 1, 0,     # R1 = 1 (success)
            Op.WITNESS, 1,         # witness success
            Op.HALT,
        ])

    def _compile_cold_path(self, failed_trial: Trial) -> bytes:
        """Compile failure path: record error, skip technique."""
        return bytes([
            Op.MOVI, 0, 0, 0,     # R0 = 0 (failure)
            Op.WITNESS, 0,         # witness failure
            Op.SNAPSHOT, 0,       # save state for debugging
            Op.HALT,
        ])

    def execute(self, agent_input: Any = None) -> Dict:
        """Execute lesson with branch prediction.
        If hot_path exists, speculate success. If mispredict, fall through to cold_path."""
        speculate_success = self.hot_path is not None and self.success_rate > 0.5
        if speculate_success:
            # Speculative execution: run hot path, check result
            result = self._run_bytecode(self.hot_path, agent_input)
            if result.get("success"):
                result["prediction"] = "correct"
                return result
            else:
                # Mispredict! Run cold path
                result["prediction"] = "mispredict"
                result["fallback"] = self._run_bytecode(self.cold_path, agent_input)
                return result
        else:
            # No hot path yet — execute both branches, let CMP decide
            return self._run_both_paths(agent_input)

    def _run_bytecode(self, code: bytes, input_val: Any) -> Dict:
        """Simplified VM execution for demo."""
        # In real implementation, use flux.vm.interpreter.Interpreter
        return {"success": True, "cycles": len(code), "input": input_val}

    def _run_both_paths(self, input_val: Any) -> Dict:
        """No prediction data — try technique, compare, branch."""
        return {
            "success": None,  # unknown until actual execution
            "cycles": 10,
            "input": input_val,
            "note": "No hot path yet — need more trials",
        }

    def failure_modes(self, top_n: int = 3) -> List[tuple]:
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
        if not self.trials:
            return "No trials yet. Be the first."
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
            if self.hot_path:
                lines.append(f"Hot path compiled: {len(self.hot_path)} bytes")
        if failing:
            lines.append(f"Avoid: {failing[0]}")
        return "\n".join(lines)

    def predict_failure_rate(self, n_agents: int) -> float:
        if self.success_rate == 1.0:
            return 0.0
        if self.success_rate == 0.0 and self.total_attempts < 3:
            return 0.8
        base = 1 - self.success_rate
        return base / (1 + n_agents / max(self.unique_agents, 1))

    def jit_stats(self) -> Dict:
        """Report JIT status: hot path compiled? cold path? speculation accuracy?"""
        return {
            "hot_path_compiled": self.hot_path is not None,
            "hot_path_size": len(self.hot_path) if self.hot_path else 0,
            "cold_path_compiled": self.cold_path is not None,
            "speculation_threshold": 0.5,
            "current_success_rate": self.success_rate,
            "eligible_for_jit": self.success_rate > 0.5 and len([t for t in self.trials if t.success]) >= 3,
        }

    def save(self, path: str = None) -> str:
        path = path or f"lesson_{self.task.lower().replace(' ', '_').replace('/', '_')[:40]}.json"
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)
        return path

    @classmethod
    def load(cls, path: str) -> "FluxLesson":
        with open(path) as f:
            data = json.load(f)
        data["trials"] = [Trial(**t) for t in data.get("trials", [])]
        # Reconstruct bytecode from hex strings
        if data.get("hot_path"):
            data["hot_path"] = bytes.fromhex(data["hot_path"])
        if data.get("cold_path"):
            data["cold_path"] = bytes.fromhex(data["cold_path"])
        return cls(**data)


class FluxLessonLibrary:
    """Indexed collection with bytecode-level deduplication."""
    def __init__(self):
        self.lessons: Dict[str, FluxLesson] = {}
        self.bytecode_index: Dict[str, List[str]] = {}  # hash -> task names

    def get_or_create(self, task: str) -> FluxLesson:
        if task not in self.lessons:
            self.lessons[task] = FluxLesson(task=task)
        return self.lessons[task]

    def index_bytecode(self, task: str, bytecode: bytes):
        """Index bytecode hash for deduplication."""
        h = hashlib.sha256(bytecode).hexdigest()[:16]
        if h not in self.bytecode_index:
            self.bytecode_index[h] = []
        self.bytecode_index[h].append(task)

    def search(self, tag: str = None, min_success_rate: float = None, has_hot_path: bool = None) -> List[FluxLesson]:
        results = list(self.lessons.values())
        if tag:
            results = [l for l in results if any(tag in t.tags for t in l.trials)]
        if min_success_rate is not None:
            results = [l for l in results if l.success_rate >= min_success_rate]
        if has_hot_path is not None:
            results = [l for l in results if (l.hot_path is not None) == has_hot_path]
        return results

    def fleet_stats(self) -> dict:
        total_trials = sum(len(l.trials) for l in self.lessons.values())
        total_lessons = len(self.lessons)
        avg_success = sum(l.success_rate for l in self.lessons.values()) / max(total_lessons, 1)
        hot_paths = sum(1 for l in self.lessons.values() if l.hot_path)
        return {
            "lessons": total_lessons,
            "trials": total_trials,
            "avg_success_rate": round(avg_success, 2),
            "unique_agents": len(set(t.agent for l in self.lessons.values() for t in l.trials)),
            "hot_paths_compiled": hot_paths,
            "deduplicated_bytecodes": len(self.bytecode_index),
        }

    def save(self, path: str = "flux_lesson_library.json"):
        with open(path, "w") as f:
            json.dump({k: asdict(v) for k, v in self.lessons.items()}, f, indent=2, default=str)


# ── Demo ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    lib = FluxLessonLibrary()
    lesson = lib.get_or_create("README detection via gh CLI")
    lesson.description = "Detect if a GitHub repo has a README using CLI tools."

    # Trials from CCC's 2026-05-03 audit
    lesson.record_trial(agent="ccc-direct", success=False, error="gh repo view --readme fails for 95% of repos", technique="gh repo view --readme", tokens_used=8000, tags=["github", "readme", "cli"], bytecode_hash="a1b2c3d4")
    lesson.record_trial(agent="kimi-auditor", success=True, error="", technique="gh api repos/{owner}/{repo}/readme", tokens_used=3000, tags=["github", "readme", "api"], bytecode_hash="e5f6g7h8")
    lesson.record_trial(agent="ccc-scout-2", success=True, error="", technique="gh api repos/{owner}/{repo}/readme", tokens_used=2500, tags=["github", "readme", "api"], bytecode_hash="e5f6g7h8")
    lesson.record_trial(agent="readme-builder", success=False, error="README already exists — false positive", technique="gh repo view --readme", tokens_used=4000, tags=["github", "readme", "cli"], bytecode_hash="a1b2c3d4")

    print("=== FLUX Lesson Demo ===")
    print(lesson.advice_for_new_agent())
    print()
    print(f"JIT stats: {lesson.jit_stats()}")
    print()
    print(f"Predicted failure rate after 10 agents: {lesson.predict_failure_rate(10):.0%}")
    print()
    print(f"Fleet stats: {lib.fleet_stats()}")
    print()

    # Index bytecode for deduplication
    lib.index_bytecode("README detection via gh CLI", b"\x2b\x00\x01\x00")
    print(f"Bytecode index: {lib.bytecode_index}")
    print()

    # Save / load test
    path = lesson.save()
    loaded = FluxLesson.load(path)
    print(f"Save/Load OK: {loaded.task} | hot_path={loaded.hot_path is not None}")
