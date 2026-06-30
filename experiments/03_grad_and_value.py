import jax


def loss_fn(
    w: jax.Array | float,
    b: jax.Array | float,
    x: jax.Array | float,
    target: float | jax.Array,
) -> float | jax.Array:
    prediction = w * x + b
    loss = (prediction - target) ** 2
    return loss


def loss_from_params(params, x, target):
    prediction = params["w"] * x + params["b"]
    loss = (prediction - target) ** 2
    return loss


def apply_grad_step(params, grads, learning_rate):
    # params and grads have the same tree structure.
    # The update runs once per leaf, so this scales past {"w", "b"}.
    return jax.tree.map(
        lambda param, grad: param - learning_rate * grad,
        params,
        grads,
    )


if __name__ == "__main__":
    w = 2.0
    b = 0.5
    x = 3.0

    target = 10.0

    loss = loss_fn(w, b, x, target)
    print(loss)

    # JAX transforms functions. These calls create new functions.
    grad_w = jax.grad(loss_fn, argnums=0)
    grads_w_b = jax.grad(loss_fn, argnums=(0, 1))
    value_and_grads = jax.value_and_grad(loss_fn, argnums=(0, 1))

    print(grad_w(w, b, x, target))
    print(grads_w_b(w, b, x, target))
    print(value_and_grads(w, b, x, target))

    print("======= LOSS WITH PARAMS ========")
    params = {"w": 2.0, "b": 0.5}
    learning_rate = 0.01
    steps = 2

    for i in range(steps):
        print(f"""Parameters initially at step {i}:
            w: {params["w"]}
            b: {params["b"]}
            """)
        loss_before = loss_from_params(params, x, target)
        # Differentiating a params tree returns a matching gradient tree.
        grads = jax.grad(loss_from_params)(params, x, target)
        print(f"""Gradients calculated at step {i}:
            w: {grads["w"]}
            b: {grads["b"]}
            """)
        params = apply_grad_step(params, grads, learning_rate)

        loss_after = loss_from_params(params, x, target)

        print(f"""Parameters at step {i} after grad step:
            w: {params["w"]}
            b: {params["b"]}
            """)
        print(f"At step {i}, loss went from {loss_before} to {loss_after}.\n")
