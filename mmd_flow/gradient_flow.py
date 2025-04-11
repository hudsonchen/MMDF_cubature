from typing import Optional, Callable

import optax
import jax
import jax.numpy as jnp
from jax import grad, random
from jax.tree_util import tree_map
from jax_tqdm import scan_tqdm
import numpy as np
from .typing import Array, Divergence


def gradient_flow(
    divergence: Divergence,
    rng_key: Array,
    Y: Array,
    save,
    args
):
    optimizer = optax.sgd(learning_rate=args.step_size)
    opt_state = optimizer.init(Y)

    threshold = 1e5
    if args.step_num <= threshold:
        step_num = int(args.step_num)
    else:
        step_num = int(threshold)

    if not save:
        @scan_tqdm(step_num)
        def one_step(dummy, i: Array):
            opt_state, rng_key, Y = dummy
            optimizer = optax.sgd(learning_rate=args.step_size)

            first_variation = divergence.get_first_variation(Y)
            velocity_field = jax.vmap(grad(first_variation))
            u = jax.random.normal(rng_key, shape=Y.shape)
            beta = args.inject_noise_scale * jnp.sqrt(1 / (i / 100 + 1))
            updates, new_opt_state = optimizer.update(velocity_field(Y + beta * u), opt_state)
            Y_next = optax.apply_updates(Y, updates)

            rng_key, _ = random.split(rng_key)
            dummy_next = (new_opt_state, rng_key, Y_next)
            return dummy_next, None 
        
        if args.step_num <= threshold:
            info_dict, _ = jax.lax.scan(one_step, (opt_state, rng_key, Y), jnp.arange(step_num))
            _, _, Y = info_dict
        else:
            for iter in range(int(args.step_num // threshold)):
                info_dict, _ = jax.lax.scan(one_step, (opt_state, rng_key, Y), jnp.arange(threshold))
                _, _, Y = info_dict
                opt_state = optimizer.init(Y)
                rng_key, _ = random.split(rng_key)
        return info_dict, Y
    else:
        @scan_tqdm(step_num)
        def one_step_save_trajectory(dummy, i: Array):
            opt_state, rng_key, Y = dummy
            optimizer = optax.sgd(learning_rate=args.step_size)

            first_variation = divergence.get_first_variation(Y)
            velocity_field = jax.vmap(grad(first_variation))
            u = jax.random.normal(rng_key, shape=Y.shape)
            beta = args.inject_noise_scale * jnp.sqrt(1 / (i / 100 + 1))
            updates, new_opt_state = optimizer.update(velocity_field(Y + beta * u), opt_state)
            Y_next = optax.apply_updates(Y, updates)

            rng_key, _ = random.split(rng_key)
            dummy_next = (new_opt_state, rng_key, Y_next)
            return dummy_next, Y_next

        if args.step_num <= threshold:
            info_dict, trajectory = jax.lax.scan(one_step_save_trajectory, (opt_state, rng_key, Y), jnp.arange(step_num))
            return info_dict, trajectory
        else:
            # This is to reduce the memory usage of saving the entire trajectory, this does not affect the gradient flow updates
            save_every = 4
            saved_steps_per_chunk = threshold // save_every
            trajectory_all = np.zeros((args.step_num // save_every, Y.shape[0], Y.shape[1]))

            for iter in range(int(args.step_num // threshold)):
                info_dict, trajectory = jax.lax.scan(one_step_save_trajectory, (opt_state, rng_key, Y), jnp.arange(threshold))
                Y = trajectory[-1, :, :]
                opt_state = optimizer.init(Y)
                rng_key, _ = random.split(rng_key)

                # Save every 3rd step from this chunk
                trajectory_subsampled = trajectory[::save_every]  # shape: (threshold // 3, N, D)
                start_idx = int(iter * saved_steps_per_chunk)
                end_idx = int((iter + 1) * saved_steps_per_chunk)
                trajectory_all[start_idx:end_idx, :, :] = trajectory_subsampled

            return info_dict, trajectory_all

