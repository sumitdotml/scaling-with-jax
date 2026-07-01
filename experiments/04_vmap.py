import jax
import jax.numpy as jnp


def loss_one(params: dict[str, float | jax.Array], x: jax.Array, target: jax.Array) -> jax.Array:
    prediction = params["w"] * x + params["b"]
    return (prediction - target) ** 2


if __name__ == "__main__":
    xs = jnp.array([1.0, 2.0, 3.0, 4.0])
    targets = jnp.array([3.0, 5.0, 7.0, 9.0])
    params = {"w": 2.0, "b": 0.5}

    loss = jnp.array(0.0)
    for i in range(len(xs)):
        loss += loss_one(params, xs[i], targets[i])
        print(f"Loss at pass {i}: {loss}")
    print(f"Total loss: {loss}\n")

    # params is shared. xs and targets are batched over axis 0.
    loss_many = jax.vmap(loss_one, in_axes=(None, 0, 0), out_axes=0)
    # vmap returns one loss per example; summing matches the loop total.
    print(loss_many(params, xs, targets))
    print(loss_many(params, xs, targets).sum())
