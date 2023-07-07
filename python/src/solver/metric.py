import numpy as np
from numba import njit
from scipy.spatial.distance import cdist, pdist
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
def calculate_happiness(musicians: np.ndarray, instruments: np.ndarray, attendees: np.ndarray, attendee_tastes: np.ndarray) -> float:
    """
    :param musicians: NDArray with Mx2 shape. Every row is (X, Y)
    :param instruments: NDArray with Mx1 shape. Every row is (Instrument,)
    :param attendees: NDArray with Ax2 shape. Every row is (X, Y)
    :param attendee_tastes: NDArray with AxT shape. Every row is tastes for a particular attendee
    :return: happiness score
    """
    total_happiness = 0
    distances_sqr = cdist(attendees, musicians, 'euclidean') ** 2
    for cur_attendee_idx in range(attendees.shape[0]):
        musician_not_blocked = np.all((distances_to_segments(
            musicians,
            (musicians, np.repeat(attendees[cur_attendee_idx][np.newaxis, :],
                                         musicians.shape[0]).reshape((musicians.shape[0], 2)))
        ) + np.eye(musicians.shape[0]) * 100) > 5, axis=1)
        cur_tastes = attendee_tastes[cur_attendee_idx]
        attendee_happiness = cur_tastes[instruments] * musician_not_blocked / distances_sqr[cur_attendee_idx]
        total_happiness += attendee_happiness.sum()
    return total_happiness

