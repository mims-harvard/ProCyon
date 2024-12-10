import numpy as np
from typing import List, Tuple, Dict

def negative_sampling_random_tail(
    cur_entity: Tuple[int, int], # TODO(debug) unused var
    num_neg_sample: int,
    num_entity: int = None,
    mask: np.ndarray = None,
    probs: np.ndarray = None,
    rng: np.random.Generator = None,
) -> List[int]:
    """
    Random tail negative sampling for CL.

    Args:
        cur_entity: Head entity with relation. Used to construct negative samples with another entity sampled from a specific entity set.
        num_neg_sample: Number of negative samples.
        num_entity: The size of the set that negative tails are sampled from.
        mask: The mask for the set that negative tails are sampled from. For GO, it is a generic mask where we only allow for sampling within namespaces (BP, CC, MF).
        probs: The sampling probabilities for each entity in the set that negative tails are sampled from. By default it is a uniform distribution.
    """
    if probs is None:
        probs = np.ones(num_entity)
    if mask is not None:
        probs = probs * mask  # remove entities that are not in the same namespace as the head entity for GO
    probs /= probs.sum()

    if num_neg_sample > num_entity:
        num_neg_sample = num_entity
        raise Warning("Number of negative samples is larger than the number of entities in the set. Set number of negative samples to the number of entities in the set.")
    if rng is None:
        negative_indices = np.random.choice(np.arange(num_entity),
                                            size=num_neg_sample,
                                            replace=False,
                                            p=probs).tolist()
    else:
        negative_indices = rng.choice(np.arange(num_entity),
                                      size=num_neg_sample,
                                      replace=False,
                                      p=probs).tolist()
    return negative_indices
