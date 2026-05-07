from __future__ import annotations

from typing import List, Sequence#, Optional
import numpy as np


def dirichlet_partition(
    labels: Sequence[int],
    num_clients: int,
    alpha: float,
    seed: int = 0,
    min_size: int = 10,
) -> List[List[int]]:
    """
    Non-IID partitioning using Dirichlet distribution over label proportions.

    Returns a list of indices for each client.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=np.int64)
    n = len(labels)
    num_classes = int(labels.max()) + 1 if n > 0 else 1

    idx_by_class = [np.where(labels == c)[0] for c in range(num_classes)]
    for c in range(num_classes):
        rng.shuffle(idx_by_class[c])

    client_indices = [[] for _ in range(num_clients)]

    # Repeat until each client has at least min_size examples
    while True:
        client_indices = [[] for _ in range(num_clients)]
        for c in range(num_classes):
            idx_c = idx_by_class[c]
            if len(idx_c) == 0:
                continue
            # Sample proportions for this class across clients
            proportions = rng.dirichlet(alpha * np.ones(num_clients))
            # Convert to counts
            counts = (proportions * len(idx_c)).astype(int)

            # Fix rounding so total matches
            diff = len(idx_c) - counts.sum()
            if diff > 0:
                for i in rng.choice(num_clients, size=diff, replace=True):
                    counts[i] += 1
            elif diff < 0:
                for i in rng.choice(np.where(counts > 0)[0], size=-diff, replace=True):
                    counts[i] -= 1

            start = 0
            for k in range(num_clients):
                end = start + counts[k]
                if end > start:
                    client_indices[k].extend(idx_c[start:end].tolist())
                start = end

        sizes = [len(x) for x in client_indices]
        if min(sizes) >= min_size:
            break
        # if alpha extremely small, this can be hard; relax a bit
        min_size = max(1, int(min_size * 0.7))

    # Shuffle within each client
    for k in range(num_clients):
        rng.shuffle(client_indices[k])

    return client_indices
