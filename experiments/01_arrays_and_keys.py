import jax
import jax.numpy as jnp


def main():
    key = jax.random.key(0)

    x = jnp.arange(5, dtype=jnp.float32)
    y = x.at[2].set(99.0)

    print("original x:", x)
    print("updated y:", y)
    print("x stayed unchanged:", x)

    key_a, key_b = jax.random.split(key)
    sample_a = jax.random.normal(key_a, shape=(3,))
    sample_b = jax.random.normal(key_b, shape=(3,))

    print("sample_a:", sample_a)
    print("sample_b:", sample_b)

    # TODO:
    # 1. Change one more element of x without mutating x.
    # 2. Generate a 2x3 random matrix.
    # 3. Reuse key_a on purpose and observe what happens.
    # 4. Write down why explicit keys are different from torch.manual_seed.


if __name__ == "__main__":
    main()
