import jax
import jax.numpy as jnp
import optax
from flax import nnx

DIN = 3
DOUT = 2
BATCH = 32
STEPS = 200
LEARNING_RATE = 0.05


class Model(nnx.Module):
    def __init__(self, din: int = DIN, dout: int = DOUT, *, rngs: nnx.Rngs) -> None:
        self.w = nnx.Param(rngs.params.uniform((din, dout)))
        self.b = nnx.Param(rngs.params.uniform((dout,)))
        self.din = din
        self.dout = dout

    def __call__(self, x: jax.Array) -> jax.Array:
        return x @ self.w[...] + self.b[...]


def loss_fn(model: Model, x: jax.Array, target: jax.Array) -> jax.Array:
    pred = model(x)
    return jnp.mean((target - pred) ** 2)


def mean_abs_err(pred: jax.Array, target: jax.Array) -> jax.Array:
    return jnp.mean(jnp.abs(pred - target))


def train_step(
    model: Model, optimizer: optax.GradientTransformation, opt_state: optax.OptState, x: jax.Array, target: jax.Array
) -> tuple[jax.Array, optax.OptState]:
    params = nnx.state(model, nnx.Param)
    loss, grads = nnx.value_and_grad(loss_fn)(model, x, target)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    updated_params = optax.apply_updates(params, updates)
    nnx.update(model, updated_params)
    return loss, opt_state


if __name__ == "__main__":
    key = jax.random.key(0)
    data_key, true_w_key, true_b_key, model_key = jax.random.split(key, 4)
    x = jax.random.uniform(data_key, (BATCH, DIN))

    true_w = jax.random.uniform(true_w_key, (DIN, DOUT))
    true_b = jax.random.uniform(true_b_key, (DOUT,))

    # defining the hidden linear rule the model should recover
    target = x @ true_w + true_b
    print(f"x: {x.shape}")
    print(f"true_w: {true_w.shape}")
    print(f"true_b: {true_b.shape}")
    print(f"target: {target.shape}")

    model = Model(DIN, DOUT, rngs=nnx.Rngs(model_key))
    params = nnx.state(model, nnx.Param)
    optimizer = optax.adam(LEARNING_RATE)

    initial_loss = loss_fn(model, x, target)
    opt_state = optimizer.init(params)

    print(f"initial loss: {initial_loss}\n")

    for step in range(STEPS):
        loss, opt_state = train_step(model, optimizer, opt_state, x, target)
        if step % 50 == 0:
            print(f"step {step}: loss={loss}")
    print(f"final loss (after {STEPS} steps): {loss_fn(model, x, target)}")

    # reading learned params from the model and not from Adam's optimizer memory
    learned_w = model.w[...]
    learned_b = model.b[...]

    print("\n=========== MEAN ABSOLUTE ERRORS: w and b ============")
    mae_w = mean_abs_err(learned_w, true_w)
    mae_b = mean_abs_err(learned_b, true_b)
    print(f"w: {mae_w}\nb: {mae_b}")
