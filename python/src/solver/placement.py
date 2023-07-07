import numpy as np
from src.mytypes import ProblemSolution, Placement
import typing as t


def get_placements(width: float, height: float, min_distance: float = 10.):
    return np.meshgrid(np.arange(min_distance, width - min_distance, min_distance),
                       np.arange(min_distance, height - min_distance, min_distance))


def get_placements_hexagonal(width: float, height: float, min_distance: float = 10., eps: float = 1e-7):
    xs, ys = np.meshgrid(np.arange(min_distance, width - min_distance, min_distance + eps),
                         np.arange(min_distance, height - min_distance, (np.sqrt(3) * min_distance / 2) + eps))
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


def placements_to_solution(placements: t.Iterable[t.Tuple[float, float]]) -> ProblemSolution:
    return ProblemSolution(placements=[Placement(x=pl[0], y=pl[1]) for pl in placements])
