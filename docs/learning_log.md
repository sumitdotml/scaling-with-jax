# Learning Log

Current topic: JAX foundations
Current exercise: Optax optimizer state
Machine/environment: local uv environment
JAX/jaxlib/backend: JAX 0.10.2, jaxlib 0.10.2, CPU backend
Last known passing command: `uv run python experiments/08_optax_state.py`
Current failing command: none
Last attempt: compared SGD and Adam optimizer state on fresh identical NNX models
Current bug or confusion: none recorded
Concepts learned: `opt_state` is optimizer memory; SGD keeps empty state; Adam stores `count`, `mu`, and `nu` for each parameter
Next action: make the tiny training loop less duplicated before moving on
