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
    args
):
    optimizer = optax.sgd(learning_rate=args.step_size)
    opt_state = optimizer.init(Y)
    threshold = 1e5

    if args.step_num <= threshold:
        step_num = int(args.step_num)
    else:
        step_num = int(threshold)

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
        return dummy_next, Y_next

    if args.step_num <= threshold:
        info_dict, trajectory = jax.lax.scan(one_step, (opt_state, rng_key, Y), jnp.arange(step_num))
        return info_dict, trajectory
    else:
        trajectory_all = np.zeros((args.step_num, Y.shape[0], Y.shape[1]))
        for iter in range(int(args.step_num // threshold)):
            info_dict, trajectory = jax.lax.scan(one_step, (opt_state, rng_key, Y), jnp.arange(threshold))
            Y = trajectory[-1, :, :]
            opt_state = optimizer.init(Y)
            rng_key, _ = random.split(rng_key)
            trajectory_all[iter * int(threshold): (iter + 1) * int(threshold), :, :] = trajectory
        return info_dict, trajectory_all

