# Learning Log

Current topic: JAX foundations
Current exercise: tiny Flax NNX module
Machine/environment: local uv environment
JAX/jaxlib/backend: JAX 0.10.2, jaxlib 0.10.2, CPU backend
Last known passing command: `uv run python experiments/07_flax_nnx_module.py`
Current failing command: none
Last attempt: wrapped the Optax update into a typed `train_step`
Current bug or confusion: none recorded
Concepts learned: NNX modules keep state directly; `nnx.Param` wraps trainable arrays; Optax updates still need `nnx.update` to write the new param state back into the model; `opt_state` is optimizer memory
Next action: try Adam and inspect the non-empty optimizer state
