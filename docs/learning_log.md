# Learning Log

Current topic: JAX foundations
Current exercise: `jit` and tracing
Machine/environment: local uv environment
JAX/jaxlib/backend: JAX 0.10.2, jaxlib 0.10.2, CPU backend
Last known passing command: `uv run python experiments/02_jit_and_tracing.py`
Current failing command: none
Last attempt: compared eager calls with jitted calls on `(3,)` and `(5,)` arrays
Current bug or confusion: none recorded
Concepts learned: `jit` traces with abstract shape/dtype values; shape changes can retrace
Next action: add same-shape repeat and observe cache reuse
