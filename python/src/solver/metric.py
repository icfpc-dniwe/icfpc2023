import numpy as np
from numba import njit
from scipy.spatial.distance import cdist, pdist, squareform
from src.solver.geometry import get_line_coefficients, distance_to_line, distances_to_segments
import typing as t


def distance_singular_sqr(place_a: np.ndarray, place_b: np.ndarray) -> float:
    return float(((place_a - place_b) ** 2).sum())


def distance(places_a: np.ndarray, places_b: np.ndarray) -> np.ndarray:
    return cdist(places_a, places_b, 'euclidean')


def check_positions_valid(
        musician_positions: np.ndarray,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        min_distance: float = 10.
) -> bool:
    distances = pdist(musician_positions, metric='euclidean')
    if np.any(distances < min_distance):
        return False
    xmin += min_distance
    ymin += min_distance
    xmax -= min_distance
    ymax -= min_distance
    return np.all(
        (musician_positions[:, 0] >= xmin) & (musician_positions[:, 0] <= xmax)
        & (musician_positions[:, 1] >= ymin) & (musician_positions[:, 1] <= ymax)
    )


# @njit
def calculate_happiness(
        musicians: np.ndarray,
        instruments: np.ndarray,
        attendees: np.ndarray,
        attendee_tastes: np.ndarray,
        pillars: t.Optional[np.ndarray] = None,
        use_ext2: bool = False,
        reduce: str = 'sum'
) -> t.Union[float, np.ndarray]:
    """
    :param musicians: NDArray with Mx2 shape. Every row is (X, Y)
    :param instruments: NDArray with Mx1 shape. Every row is (Instrument,)
    :param attendees: NDArray with Ax2 shape. Every row is (X, Y)
    :param attendee_tastes: NDArray with AxT shape. Every row is tastes for a particular attendee
    :param pillars: NDArray with Px3 shape. Every row is (X, Y, R)
    :param use_ext2: bool, whether to use extension2 of the rules
    :param reduce: either 'sum' or 'none'
    :return: happiness score
    """
    total_happiness = np.zeros((len(attendees), len(musicians)), dtype=np.int32)
    distances_sqr = cdist(attendees, musicians, 'euclidean') ** 2
    for cur_attendee_idx in range(attendees.shape[0]):
        segments = (musicians, np.repeat(attendees[cur_attendee_idx][np.newaxis, :],
                                         musicians.shape[0]).reshape((2, musicians.shape[0])).T)
        musician_not_blocked = np.all((distances_to_segments(
            musicians,
            segments
        ) + np.eye(musicians.shape[0]) * 100) > 5, axis=0)
        if pillars is not None:
            pillar_blocked = np.all(distances_to_segments(
                pillars[:, :2],
                segments
            ) > pillars[:, 2:], axis=0)
            musician_not_blocked = musician_not_blocked & pillar_blocked
        cur_tastes = attendee_tastes[cur_attendee_idx]
        attendee_happiness = np.ceil(1000000 * cur_tastes[instruments] * musician_not_blocked / distances_sqr[cur_attendee_idx])
        total_happiness[cur_attendee_idx] = attendee_happiness
    if use_ext2:
        musician_distances = squareform(pdist(musicians, 'euclidean')) + np.eye(len(musicians))
        instruments_tiled = np.tile(instruments[np.newaxis, :], (len(instruments), 1))
        musicians_same_instrument = instruments_tiled == instruments_tiled.T
        q = 1 + ((1 / musician_distances) * musicians_same_instrument
                 - np.eye(len(musicians))).sum(axis=0, keepdims=True)
        total_happiness = np.ceil(total_happiness * q)
    total_happiness = total_happiness.sum(axis=1)
    if reduce == 'sum':
        return total_happiness.sum()
    elif reduce == 'none':
        return total_happiness
    return total_happiness.sum()

