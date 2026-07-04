import jax
import jax.numpy as jnp
import optax
from flax import nnx


class Model(nnx.Module):
    def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs) -> None:
        self.w = nnx.Param(rngs.params.uniform((din, dout)))
        self.b = nnx.Param(rngs.params.uniform((dout,)))
        self.din = din
        self.dout = dout

    def __call__(self, x: jax.Array) -> jax.Array:
        return x @ self.w[...] + self.b[...]


def loss_fn(model: Model, x: jax.Array, target: jax.Array) -> jax.Array:
    pred = model(x)
    return jnp.mean((pred - target) ** 2)


def train_step(
    model: Model,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    x: jax.Array,
    target: jax.Array,
) -> tuple[jax.Array, optax.OptState]:
    params = nnx.state(model, nnx.Param)
    loss, grads = nnx.value_and_grad(loss_fn)(model, x, target)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    updated_params = optax.apply_updates(params, updates)
    nnx.update(model, updated_params)
    return loss, opt_state


if __name__ == "__main__":
    key = jax.random.key(0)
    x_key, target_key = jax.random.split(key)

    print("========== INITIALIZATION STAGE ==========")

    x = jax.random.uniform(x_key, (1, 2))
    target = jax.random.uniform(target_key, (1, 3))

    # using the same seed keeps the optimizer comparison about the update rule.
    sgd_model = Model(2, 3, rngs=nnx.Rngs(0))
    adam_model = Model(2, 3, rngs=nnx.Rngs(0))

    sgd_params = nnx.state(sgd_model, nnx.Param)
    adam_params = nnx.state(adam_model, nnx.Param)
    learning_rate = 0.01

    optimizer_sgd = optax.sgd(learning_rate)
    optimizer_adam = optax.adam(learning_rate)

    initial_loss_sgd = loss_fn(sgd_model, x, target)
    initial_loss_adam = loss_fn(adam_model, x, target)

    opt_state_sgd = optimizer_sgd.init(sgd_params)
    opt_state_adam = optimizer_adam.init(adam_params)

    print("Optimizers: SGD and Adam")
    print(f"initial losses: sgd={initial_loss_sgd}, adam={initial_loss_adam}")
    print("initial optimizer memory:")
    print(f"  sgd: {opt_state_sgd}")
    print(f"  adam: {opt_state_adam}")

    for step in range(5):
        # each optimizer carries its own model and its own optimizer memory.
        loss_sgd, opt_state_sgd = train_step(sgd_model, optimizer_sgd, opt_state_sgd, x, target)
        loss_adam, opt_state_adam = train_step(adam_model, optimizer_adam, opt_state_adam, x, target)
        print(f"step {step}: sgd loss={loss_sgd}, memory={opt_state_sgd}")
        print(f"        adam loss={loss_adam}, memory={opt_state_adam}")
    print("\n:::::::: AFTER THE LOOP :::::::::")
    print(f"final sgd loss: {loss_fn(sgd_model, x, target)}")
    print(f"final adam loss: {loss_fn(adam_model, x, target)}")
