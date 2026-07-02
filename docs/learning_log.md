# Learning Log

Current topic: JAX foundations
Current exercise: tiny Flax NNX module
Machine/environment: local uv environment
JAX/jaxlib/backend: JAX 0.10.2, jaxlib 0.10.2, CPU backend
Last known passing command: `uv run python experiments/07_flax_nnx_module.py`
Current failing command: none
Last attempt: built a tiny NNX module with explicit Param reads via `[...]`
Current bug or confusion: none recorded
Concepts learned: NNX modules keep state directly; `nnx.Param` wraps trainable arrays
Next action: connect the module to `nnx.value_and_grad` and Optax
