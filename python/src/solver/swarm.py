import numpy as np
from src.solver.placement import generate_compliant_positions, jiggle_positions
from src.solver.metric import check_positions_valid, calculate_happiness


new_positions_vecs = np.array([
    (1, 0),
    (0, 1),
    (-1, 0),
    (0, -1)
]) * 5


def swarm_step(positions, bounds, old_metric, gradient_step: float = 1, max_step: float = 100, **metric_params):
    results = [
        jiggle_positions(positions + cur_vec[np.newaxis, :], bounds)
        for cur_vec in new_positions_vecs
    ]
    if np.any([r[0] is None for r in results]):
        print('jiggling error', [r[1] for r in results])
        return None
    possible_positions = [r[0] for r in results]
    gradients = [calculate_happiness(pos, **metric_params, reduce='attendee') - old_metric
                 for pos in possible_positions]
    print()
    step_vecs = np.array([pos[np.newaxis, :] * g[:, np.newaxis] * gradient_step
                          for pos, g in zip(new_positions_vecs, gradients)]).sum(axis=0)
    step_vecs = np.minimum(max_step, np.maximum(-max_step, step_vecs))
    new_positions = positions + step_vecs
    new_positions[:, 0] = np.maximum(bounds[0], new_positions[:, 0])
    new_positions[:, 1] = np.maximum(bounds[1], new_positions[:, 1])
    new_positions[:, 0] = np.minimum(bounds[2], new_positions[:, 0])
    new_positions[:, 1] = np.minimum(bounds[3], new_positions[:, 1])
    new_positions_r, num_iter = jiggle_positions(new_positions, bounds)
    if new_positions_r is None:
        print('new jiggling error', num_iter)
        return None
    new_metric = calculate_happiness(new_positions_r, **metric_params, reduce='attendee')
    return new_positions_r, new_metric


def mutate_positions(positions, bounds, max_step: float):
    step_vecs = np.random.uniform(-max_step, max_step, size=positions.shape)
    new_positions = positions + step_vecs
    new_positions[:, 0] = np.maximum(bounds[0], new_positions[:, 0])
    new_positions[:, 1] = np.maximum(bounds[1], new_positions[:, 1])
    new_positions[:, 0] = np.minimum(bounds[2], new_positions[:, 0])
    new_positions[:, 1] = np.minimum(bounds[3], new_positions[:, 1])
    new_positions_r, num_iter = jiggle_positions(new_positions, bounds)
    if new_positions_r is None:
        print('new jiggling error', num_iter)
        return None
    return new_positions_r
