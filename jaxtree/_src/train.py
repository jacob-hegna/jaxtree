from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import optax

from jaxtree._src.tree import DTree, PyTree, evaluate

DEFAULT_LEARNING_RATE = 1e-3

def _l2_loss(tree, params, x, y):
    return (jnp.sum(partial(evaluate, tree)(params, x)) - y) ** 2



def get_update_fn(tree, optimizer, loss_fn: Callable) -> Callable:
    _loss_fn = partial(loss_fn, tree)
    def _update(params: PyTree, opt_state: PyTree, x: jnp.ndarray, y: jnp.ndarray):
        grads = jax.grad(_loss_fn)(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    return _update


def ez_train(
    tree: DTree,
    params: PyTree,
    x: jnp.ndarray,
    y: jnp.ndarray,
    *,
    steps=1000,
    learning_rate=DEFAULT_LEARNING_RATE
) -> PyTree:
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    update = jax.jit(get_update_fn(tree, optimizer, _l2_loss))
    for _ in range(steps):
        for a, b in zip(x, y):
            params, opt_state = update(params, opt_state, a, b)

    return params
