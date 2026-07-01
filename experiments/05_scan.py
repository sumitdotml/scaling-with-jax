import jax
import jax.numpy as jnp


def step(
    running_sum: jax.Array, current_number: jax.Array
) -> tuple[jax.Array, jax.Array]:
    # The first return value is passed to the next step.
    # The second return value is stacked across all steps.
    next_running_sum = running_sum + current_number
    saved_value = next_running_sum**2
    return next_running_sum, saved_value


if __name__ == "__main__":
    xs = jnp.array([1, 3, 6, 10])
    running_sum = jnp.array(0)
    saved_values = []

    for i in range(len(xs)):
        running_sum, saved_value = step(running_sum, xs[i])
        saved_values.append(saved_value)
        print(
            f"On step {i}:\n"
            f"  running_sum: {running_sum}\n"
            f"  saved_value: {saved_value}\n"
            f"  saved_values: {saved_values}\n"
        )
    stacked_saved_values = jnp.stack(saved_values)

    print("============= python loop ==============\n")
    print(
        f"Final running sum (python): {running_sum}\n"
        f"Stacked saved values (python): {stacked_saved_values}\n"
    )

    print("============= jax lax.scan ==============\n")
    initial_running_sum = jnp.array(0)
    # scan runs the same step function and stacks saved_value for us.
    final_running_sum_jax, stacked_outputs_jax = jax.lax.scan(
        step, initial_running_sum, xs
    )
    print(
        f"Final running sum (jax): {final_running_sum_jax}\n"
        f"Stacked saved values (jax): {stacked_outputs_jax}\n"
    )
