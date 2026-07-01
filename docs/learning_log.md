# Learning Log

Current topic: JAX foundations
Current exercise: `vmap` complete
Machine/environment: local uv environment
JAX/jaxlib/backend: JAX 0.10.2, jaxlib 0.10.2, CPU backend
Last known passing command: `uv run python experiments/04_vmap.py`
Current failing command: none
Last attempt: compared Python loop loss with vmapped per-example losses
Current bug or confusion: none recorded
Concepts learned: `vmap` batches selected axes; `in_axes=(None, 0, 0)` shares params
Next action: start `lax.scan`
