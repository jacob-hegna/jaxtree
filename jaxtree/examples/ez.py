from functools import partial

import jax
import jax.numpy as jnp
import optax

import jaxtree
from jaxtree._src.train import ez_train

def main():
    key = jax.random.PRNGKey(42)

    tree = jaxtree.make_tree(in_shape=(3,), depth=3, batch=1024)
    params = jaxtree.make_tree_params(tree, key)

    data = [
        (jnp.array([1.0, 2.0, 3.0]), -5.0),
        (jnp.array([2.0, 3.0, 4.0]), 5.0),
        (jnp.array([3.0, 4.0, 5.0]), 0.0),
    ]
    x, y = zip(*data)

    params = jaxtree.ez_train(tree, params, x, y)

    f = jax.jit(partial(jaxtree.evaluate, tree))

    print(f(params, jnp.array([1.0, 2.0, 3.0])))
    print(f(params, jnp.array([2.0, 3.0, 4.0])))
    print(f(params, jnp.array([3.0, 4.0, 5.0])))


if __name__ == "__main__":
    main()
