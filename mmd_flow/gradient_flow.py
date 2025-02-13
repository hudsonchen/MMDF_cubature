from typing import Optional, Callable

import optax
import jax
import jax.numpy as jnp
from jax import grad, random
from jax.tree_util import tree_map
from jax_tqdm import scan_tqdm
from .typing import Array, Divergence


def gradient_flow(
    divergence: Divergence,
    rng_key: Array,
    Y: Array,
    args
):
    optimizer = optax.sgd(learning_rate=args.step_size)
    opt_state = optimizer.init(Y)
    
    @scan_tqdm(args.step_num)
    def one_step(dummy, i: Array):
        opt_state, rng_key, Y = dummy
        optimizer = optax.sgd(learning_rate=args.step_size)

        first_variation = divergence.get_first_variation(Y)
        velocity_field = jax.vmap(grad(first_variation))
        updates, new_opt_state = optimizer.update(velocity_field(Y), opt_state)
        Y_next = optax.apply_updates(Y, updates)

        rng_key, _ = random.split(rng_key)
        dummy_next = (new_opt_state, rng_key, Y_next)
        return dummy_next, Y_next

    info_dict, trajectory = jax.lax.scan(one_step, (opt_state, rng_key, Y), jnp.arange(args.step_num))
    return info_dict, trajectory
