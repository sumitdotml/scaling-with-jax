import jax
import jax.numpy as jnp


if __name__ == "__main__":
    params = {
        "linear1": {
            "w": jnp.array([1.0, 2.0, 3.0]),
            "b": jnp.array(0.5),
        },
        "linear2": {
            "w": jnp.array([0.1, 0.2]),
            "b": jnp.array(-1.0),
        },
    }

    fake_grads = {
        "linear1": {
            "w": jnp.array([4.0, -0.33, 3.5]),
            "b": jnp.array(-0.67),
        },
        "linear2": {
            "w": jnp.array([-1.5, -3.85]),
            "b": jnp.array(0.1),
        },
    }

    print("============ PARAMS ==============")
    print(f"Leaves:\n{jax.tree.leaves(params)}\n")
    print(f"Structure:\n{jax.tree.structure(params)}\n")
    print(f"Original params:\n{params}\n")

    # tree.map preserves the nested structure and changes each leaf.
    scaled_params = jax.tree.map(lambda leaf: leaf * 2, tree=params)
    print(f"Scaled params (each leaf * 2):\n{scaled_params}\n")

    print("============ GRAD UPDATE ==============")
    learning_rate = 0.01
    updated_params = jax.tree.map(
        lambda param, grad: param - learning_rate * grad,
        params,
        fake_grads,
    )
    print(f"Fake grads:\n{fake_grads}\n")
    print(f"Updated params:\n{updated_params}\n")
    print(f"Original params after update:\n{params}\n")
