import jax
import jax.numpy as jnp


def main():
    x = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)

    print(f"jax: {jax.__version__}")
    print(f"backend: {jax.default_backend()}")
    print(f"devices: {jax.devices()}")
    print(f"x:\n{x}")
    print(f"x device: {x.devices()}")
    print(f"x squared:\n{x * x}")


if __name__ == "__main__":
    main()
