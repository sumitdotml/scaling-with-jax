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


def train_step(
    model: Model, optimizer, opt_state: tuple[...], x: jax.Array, target: jax.Array
):
    # reading the model's current Param state before this update.
    params = nnx.state(model, nnx.Param)

    loss, grads = nnx.value_and_grad(loss_fn)(model, x, target)

    # opt_state is optimizer memory; SGD keeps it empty, Adam would not.
    updates, opt_state = optimizer.update(grads, opt_state, params)
    updated_params = optax.apply_updates(params, updates)

    # writing the updated Param state back into the same model object.
    nnx.update(model, updated_params)

    return loss, opt_state


if __name__ == "__main__":
    print("============== OPTAX UPDATE ===============")
    key = jax.random.key(0)
    x_key, target_key = jax.random.split(key)
    x = jax.random.uniform(x_key, (1, 3))
    target = jax.random.uniform(target_key, (1, 2))
    model = Model(3, 2, rngs=nnx.Rngs(0))

    # initializing optimizer memory from the initial parameter structure.
    params = nnx.state(model, nnx.Param)

    learning_rate = 0.01
    optimizer = optax.sgd(learning_rate)

    opt_state = optimizer.init(params)

    for step in range(5):
        loss, opt_state = train_step(model, optimizer, opt_state, x, target)
        print(f"step {step}: loss={loss}, opt_state={opt_state}")
    print("\n:::::::: AFTER THE LOOP :::::::::")
    print(f"loss_fn(model, x, target) returns: {loss_fn(model, x, target)}")
