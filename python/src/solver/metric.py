import numpy as np
from numba import njit
from scipy.spatial.distance import cdist, pdist
from src.solver.geometry import get_line_coefficients, distance_to_line
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


@njit
def calculate_happiness(musicians: np.ndarray, attendees: np.ndarray, attendee_tastes: np.ndarray) -> float:
    """
    :param musicians: NDArray with Mx3 shape. Every row is (X, Y, Instrument)
    :param attendees: NDArray with Ax2 shape. Every row is (X, Y)
    :param attendee_tastes: NDArray with AxT shape. Every row is tastes for a particular attendee
    :return: happiness score
    """
    total_happiness = 0
    for cur_attendee_idx in range(attendees.shape[0]):
        line_coefficients = get_line_coefficients(
            musicians[:, :2],
            np.tile(attendees[cur_attendee_idx:cur_attendee_idx+1, :2], (musicians.shape[0], 1))
        )
        musician_blocked = np.all(distance_to_line(line_coefficients, musicians[:, :2]) > 5, axis=1)
        cur_tastes = attendee_tastes[cur_attendee_idx]
        attendee_happiness = cur_tastes[musicians[:, 2]] * musician_blocked

