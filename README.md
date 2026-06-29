# Scaling With JAX

Learning JAX for the scaling book, small transformer experiments, scaling-law derivations, and later dense-vs-MoE systems work.

## Quickstart

```bash
uv sync
uv run python experiments/00_env_check.py
uv run python experiments/01_arrays_and_keys.py
```

## Layout

- `experiments/` contains small runnable scripts, one concept at a time.
- `docs/learning_log.md` tracks current state across sessions.
- `CONTEXT.md` gives future tutoring sessions the project state.
- `tutor_harness.md` defines the no-spoonfeeding tutoring contract.

## First Sequence

1. arrays, immutability, and `.at[]`
2. PRNG keys
3. `jit` and tracing
4. `grad` and `value_and_grad`
5. `vmap`
6. `lax.scan`
7. PyTrees
8. Flax and Optax tiny training loop
