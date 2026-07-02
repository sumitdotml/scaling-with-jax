import jax
from flax import nnx


class Model(nnx.Module):
    def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs) -> None:
        # keeping trainable arrays as direct module state.
        self.w = nnx.Param(rngs.params.uniform((din, dout)))
        self.b = nnx.Param(rngs.params.uniform((dout,)))
        self.din, self.dout = din, dout

    def __call__(self, x: jax.Array):
        # reading Param arrays with [...] keeps this on the current NNX API.
        return x @ self.w[...] + self.b[...]


if __name__ == "__main__":
    x = jax.random.uniform(jax.random.key(0), (1, 3))
    model = Model(3, 2, rngs=nnx.Rngs(0))
    # tracking the shape flow: x [batch, din] -> y [batch, dout].
    y = model(x)
    print(f"x: {x.shape}\n{x}")
    print(f"model:\n    model.w[...].shape: {model.w[...].shape}")
    print(f"    model.b[...].shape: {model.b[...].shape}")
    print(f"y: {y.shape}\n{y}")
