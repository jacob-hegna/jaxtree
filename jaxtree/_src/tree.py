from __future__ import annotations

import dataclasses
from functools import partial
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

PyTree = Any


def _logistic_activation(x, k):
    return jnp.sum(1.0 / (1 + jnp.exp(-x * k)))


@dataclasses.dataclass
class DTree:
    kind: str

    in_shape: Tuple

    batch: int

    lt: Optional[DTree] = None
    gt: Optional[DTree] = None


def make_tree(
    *,
    in_shape: Tuple,
    depth: int,
    batch: int = 1,
) -> DTree:
    assert depth >= 0

    if depth == 0:
        return DTree(kind="leaf", in_shape=in_shape, batch=batch)
    return DTree(
        kind="internal",
        in_shape=in_shape,
        batch=batch,
        lt=make_tree(in_shape=in_shape, depth=depth - 1, batch=batch),
        gt=make_tree(in_shape=in_shape, depth=depth - 1, batch=batch),
    )


def _evaluate(tree: DTree, params: PyTree, val: jnp.ndarray) -> jnp.ndarray:
    if tree.kind == "leaf":
        return params

    assert tree.lt and tree.gt

    pivot, k, lt_params, gt_params = params

    lt_val = jnp.multiply(
        _logistic_activation(pivot - val, k), _evaluate(tree.lt, lt_params, val)
    )
    gt_val = jnp.multiply(
        _logistic_activation(val - pivot, k), _evaluate(tree.gt, gt_params, val)
    )
    return lt_val + gt_val


def evaluate(tree: DTree, params: PyTree, val: jnp.ndarray) -> float:
    evaluate_fn = partial(_evaluate, tree)
    return jnp.sum(jax.vmap(evaluate_fn)(params, jnp.tile(val, (tree.batch, 1))))


def make_tree_params(tree: DTree, key: jax.random.KeyArray):
    shape = (tree.batch,) + tree.in_shape

    if tree.kind == "leaf":
        return jax.random.normal(key, shape=shape)

    assert tree.lt and tree.gt

    key, pivot_key = jax.random.split(key)
    key, k_key = jax.random.split(key)
    key, lt_key = jax.random.split(key)
    key, gt_key = jax.random.split(key)
    return [
        jax.random.normal(pivot_key, shape=shape),
        jax.random.normal(k_key, shape=shape),
        make_tree_params(tree.lt, lt_key),
        make_tree_params(tree.gt, gt_key),
    ]
