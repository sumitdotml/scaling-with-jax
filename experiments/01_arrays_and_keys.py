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

    z = y.at[1].set(77.0)
    print("updated z:", z)

    rand_matrix = jax.random.uniform(key, shape=(2, 3))
    print(rand_matrix)

    reusing = jax.random.normal(key_a, (7,))
    print(reusing)


if __name__ == "__main__":
    main()
