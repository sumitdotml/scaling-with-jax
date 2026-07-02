# Learning Log

Current topic: JAX foundations
Current exercise: tiny Flax NNX module
Machine/environment: local uv environment
JAX/jaxlib/backend: JAX 0.10.2, jaxlib 0.10.2, CPU backend
Last known passing command: `uv run python experiments/07_flax_nnx_module.py`
Current failing command: none
Last attempt: used `nnx.value_and_grad` and a manual PyTree update on NNX params
Current bug or confusion: none recorded
Concepts learned: NNX modules keep state directly; `nnx.Param` wraps trainable arrays; `nnx.update` writes an updated param state back into the model
Next action: replace the manual update with Optax
