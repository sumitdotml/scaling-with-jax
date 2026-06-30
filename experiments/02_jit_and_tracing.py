import jax


@jax.jit
def operation_jit(x: jax.Array) -> jax.Array:
    # This prints during tracing. JAX knows shape and dtype here and not the values.
    print("inside jax.jit", x)
    return x * x + 1


def operation(x: jax.Array) -> jax.Array:
    # Eager Python call: x is the concrete array.
    print("inside operation (not jit)", x)
    return x * x + 1


if __name__ == "__main__":
    key = jax.random.key(0)
    key1, key2 = jax.random.split(key)
    x = jax.random.uniform(key1, (3,))
    print("x:", x)
    print(f"x with the operation: {operation(x)}\n")
    # First float32[3] jit call: trace, compile, execute.
    print(f"x with the jit operation: {operation_jit(x)}\n")

    y = jax.random.uniform(key2, (5,))
    print(f"y with the operation: {operation(y)}\n")
    # float32[5] is a new signature, so this traces again.
    print(f"y with the jit operation: {operation_jit(y)}\n")
