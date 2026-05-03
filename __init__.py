"""cocapn_lessons — FLUX v3.0 Trial-based Learning.

Two API levels:

1. **Python-native** (`cocapn_lessons`): LessonLibrary with O(1/n) failure rate
2. **FLUX v3.0** (`cocapn_lessons_flux`): WITNESS opcode profiling, JIT hot paths,
   branch prediction (JGE/JNE), bytecode deduplication

Default import is FLUX v3.0 API.
"""
from cocapn_lessons_flux import FluxLesson, FluxLessonLibrary, Trial, Op, main as run_demo

try:
    from cocapn_lessons import Lesson, LessonLibrary
except ImportError:
    Lesson = None
    LessonLibrary = None

__version__ = "3.0.0"
__all__ = ["FluxLesson", "FluxLessonLibrary", "Trial", "Op", "run_demo"]
