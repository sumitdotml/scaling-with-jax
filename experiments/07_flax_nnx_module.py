import jax
import jax.numpy as jnp
from flax import nnx
import optax


class Model(nnx.Module):
    def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs) -> None:
        # keeping trainable arrays as direct module state.
        self.w = nnx.Param(rngs.params.uniform((din, dout)))
        self.b = nnx.Param(rngs.params.uniform((dout,)))
        self.din, self.dout = din, dout

    def __call__(self, x: jax.Array):
        # reading Param arrays with [...] keeps this on the current NNX API.
        return x @ self.w[...] + self.b[...]


def loss_fn(model: Model, x: jax.Array, target: jax.Array):
    pred = model(x)
    return jnp.mean((pred - target) ** 2)


if __name__ == "__main__":
    key = jax.random.key(0)
    x_key, target_key = jax.random.split(key)
    x = jax.random.uniform(x_key, (1, 3))
    target = jax.random.uniform(target_key, (1, 2))
    model = Model(3, 2, rngs=nnx.Rngs(0))
    # tracking the shape flow: x [batch, din] -> y [batch, dout].
    y = model(x)
    print("================ FORWARD PASS =================")
    print(f"x: {x.shape}")
    print(f"target: {target.shape}")
    print(f"params: w {model.w[...].shape}, b {model.b[...].shape}")
    print(f"y: {y.shape}")

    print("============== OPTAX UPDATE ===============")
    learning_rate = 0.01
    # taking a Param-only snapshot of this model's current w and b.
    params = nnx.state(model, nnx.Param)

    optimizer = optax.sgd(learning_rate)
    opt_state = optimizer.init(params)

    loss, grads = nnx.value_and_grad(loss_fn)(model, x, target)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    updated_params = optax.apply_updates(params, updates)
    nnx.update(model, updated_params)

    loss_after = loss_fn(model, x, target)
    # reading again after nnx.update because params was the old snapshot.
    current_params = nnx.state(model, nnx.Param)
    print(f"learning rate: {learning_rate}")
    print(f"loss before update: {loss}")
    print(f"loss after update: {loss_after}")
    print(f"loss change: {loss_after - loss}")
    print(f"grad shapes: w {grads['w'].shape}, b {grads['b'].shape}")
    print(
        "current param shapes: "
        f"w {current_params['w'].shape}, b {current_params['b'].shape}"
    )
