import numpy as np
from src.solver.metric import check_positions_valid
from src.solver.geometry import calculate_bounce_force
from src.mytypes import ProblemSolution, Placement
import typing as t


def get_placements(width: float, height: float, min_distance: float = 10.):
    return np.meshgrid(np.arange(min_distance, width - min_distance + 1e-7, min_distance),
                       np.arange(min_distance, height - min_distance + 1e-7, min_distance))


def get_placements_hexagonal(width: float, height: float, min_distance: float = 10., eps: float = 1e-7):
    xs, ys = np.meshgrid(np.arange(min_distance, width - 1.5 * min_distance + eps, min_distance + eps),
                         np.arange(min_distance, height - min_distance + eps, (np.sqrt(3) * min_distance / 2) + eps))
    xs[::2, :] += min_distance / 2
    return xs, ys


def filter_placements(
        placements: t.Iterable[t.Tuple[float, float]],
        width: float,
        height: float,
        min_distance: float = 10.
) -> t.List[t.Tuple[float, float]]:
    return list(filter(lambda pl: min_distance < pl[0] < (width - min_distance)
                                  and min_distance < pl[1] < (height - min_distance),
                       placements))


def placements_to_solution(
        placements: t.Iterable[t.Tuple[float, float]],
        volumes: t.Iterable[float]
) -> ProblemSolution:
    return ProblemSolution(placements=[Placement(x=pl[0], y=pl[1]) for pl in placements], volumes=list(volumes))


def generate_compliant_positions(bounds, num_to_generate: int, min_distance: float = 10, max_iter: int = 100):
    xmin, ymin, xmax, ymax = bounds
    positions = []
    for cur_num in range(num_to_generate):
        for _ in range(max_iter):
            new_pos_x = np.random.uniform(xmin + min_distance, xmax - min_distance + 1e-7)
            new_pos_y = np.random.uniform(xmin + min_distance, xmax - min_distance + 1e-7)
            if check_positions_valid(np.array(positions + [(new_pos_x, new_pos_y)]),
                                     xmin, xmax, ymin, ymax, min_distance):
                positions.append((new_pos_x, new_pos_y))
                break
        if len(positions) <= cur_num:
            return None
    return positions


def jiggle_positions(positions: np.ndarray, bounds, step_size: float = 0.1, max_iter: int = 100, min_distance: float = 10):
    xmin, ymin, xmax, ymax = bounds
    iter_idx = 0
    step_size_r = step_size * min_distance
    while not check_positions_valid(positions, xmin, xmax, ymin, ymax, min_distance):
        iter_idx += 1
        if iter_idx >= max_iter:
            return None, iter_idx
        forces = calculate_bounce_force(positions, bounds, min_distance=min_distance)
        # forces = forces / np.linalg.norm(forces + 1e-7, axis=1, keepdims=True)
        positions = positions + forces * step_size_r  # * np.cos(np.pi / 2 * (iter_idx - 1) / max_iter)
        # print('jiggle', iter_idx, np.max(forces))
        # delta = np.max(np.abs(forces))
    return positions, iter_idx


def mutate_positions(positions, bounds, max_step: float, jiggle_max_ste: int = 100):
    step_vecs = np.random.uniform(-max_step, max_step, size=positions.shape)
    # print(max_step)
    new_positions = positions + step_vecs
    new_positions[:, 0] = np.maximum(bounds[0] + 10, new_positions[:, 0] - 10)
    new_positions[:, 1] = np.maximum(bounds[1] + 10, new_positions[:, 1] - 10)
    new_positions[:, 0] = np.minimum(bounds[2] + 10, new_positions[:, 0] - 10)
    new_positions[:, 1] = np.minimum(bounds[3] + 10, new_positions[:, 1] - 10)
    new_positions_r, num_iter = jiggle_positions(new_positions, bounds, max_iter=jiggle_max_ste)
    if new_positions_r is None:
        print('new jiggling error', num_iter)
        return None
    return new_positions_r
