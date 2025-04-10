import jax
import jax.numpy as jnp

import timeit
import torch

def test_torch():
    n = int(1e6)
    random_voxels = torch.randint(0, 1000, (n, 3))
    print(timeit.timeit(lambda: torch.unique(random_voxels, return_counts=True), number=1))

    start = timeit.default_timer()
    voxel_counts = {}
    for voxel in random_voxels:
        key = tuple(voxel.tolist())
        voxel_counts[key] = voxel_counts.get(key, 0) + 1
    end = timeit.default_timer()
    print(end - start)

def test_jax():
    n = int(1e6)
    random_voxels = jax.random.randint(jax.random.PRNGKey(0), (n, 3), 0, 1000)
    print(timeit.timeit(lambda: jax.lax.stop_gradient(jax.lax.dynamic_update_slice(jnp.zeros((1000,), dtype=jnp.int32), jax.ops.index_add(random_voxels, jnp.zeros((n,), dtype=jnp.int32), 1))), number=1))

    start = timeit.default_timer()
    voxel_counts = jax.lax.dynamic_update_slice(jnp.zeros((1000,), dtype=jnp.int32), jax.ops.index_add(random_voxels, jnp.zeros((n,), dtype=jnp.int32), 1))
    end = timeit.default_timer()
    print(end - start)

if __name__ == "__main__":
    test_torch()