import jax
import jax.numpy as jnp


@jax.jit
def operation_jit(x: jax.Array) -> jax.Array:
    # This prints during tracing. JAX knows shape and dtype here and not the values.
    print("inside jax.jit", x)
    return x * x + 1


def operation(x: jax.Array) -> jax.Array:
    # Eager Python call: x is the concrete array.
    print("inside operation (not jit)", x)
    return x * x + 1


@jax.jit(static_argnames="use_bias")
def operation_w_bool(x: jax.Array, use_bias: bool) -> jax.Array:
    # use_bias is static: Python can choose a branch while tracing.
    # Changing True to False creates a separate cached program.
    print(f"tracing use_bias = {use_bias} for {x}\n")
    if use_bias:
        print(x)
        return x * x + 1
    else:
        print(x)
        return x * x


if __name__ == "__main__":
    key = jax.random.key(0)
    key1, key2, key3, key4 = jax.random.split(key, 4)
    x = jax.random.uniform(key1, (3,))
    print("x:", x)
    print(f"x with the operation: {operation(x)}\n")
    # First float32[3] jit call: trace, compile, execute.
    print(f"x with the jit operation: {operation_jit(x)}\n")
    print(f"x with the jit operation (2nd time): {operation_jit(x)}\n")

    x2 = jax.random.uniform(key2, (3,))
    print(f"x2 with the jit operation: {operation_jit(x2)}\n")

    y = jax.random.uniform(key3, (5,))
    print(f"y with the operation: {operation(y)}\n")
    # float32[5] is a new signature, so this traces again.
    print(f"y with the jit operation: {operation_jit(y)}\n")

    z = jax.random.uniform(key4, (3,), dtype=jnp.float16)
    print(f"z with the operation (size 3 but different dtype): {operation(z)}\n")
    print(f"z with the jit operation (different dtype): {operation_jit(z)}\n")

    # Static args are config and not data. They become part of the jit cache key.
    operation_w_bool(x, True)
    operation_w_bool(x, False)
