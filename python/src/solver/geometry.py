import numpy as np
from numba import njit
import typing as t


# @njit
def get_line_coefficients(left_points: np.ndarray, right_points: np.ndarray) -> np.ndarray:
    c = (left_points[:, 0] - right_points[:, 0]) * left_points[:, 1]\
        + (right_points[:, 1] - left_points[:, 1]) * left_points[:, 0]
    return np.stack((left_points[:, 1] - right_points[:, 1],
                     right_points[:, 0] - left_points[:, 0],
                     c)).T


@njit
def distance_to_line(line_coefficients: np.ndarray, points: np.ndarray) -> np.ndarray:
    normalizer = np.sqrt(line_coefficients[:, 0] ** 2 + line_coefficients[:, 1] ** 2)
    points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    return np.abs(line_coefficients @ points.T) / normalizer


# https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment/6853926#6853926
@njit
def distance_to_segment(x: float, y: float, x1: float, y1: float, x2: float, y2: float) -> float:
    A = x - x1
    B = y - y1
    C = x2 - x1
    D = y2 - y1

    dot = A * C + B * D
    len_sq = C * C + D * D
    param = -1
    if len_sq != 0:  # in case of 0 length line
        param = dot/len_sq

    if param < 0:
        xx = x1
        yy = y1
    elif param > 1:
        xx = x2
        yy = y2
    else:
        xx = x1 + param * C
        yy = y1 + param * D

    dx = x - xx
    dy = y - yy
    return np.sqrt(dx * dx + dy * dy)


@njit
def distances_to_segments(points: np.ndarray, segments: t.Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    distances = np.zeros((points.shape[0], segments[0].shape[0]), dtype=np.float64)
    for cur_point_idx in range(points.shape[0]):
        for cur_segment_idx in range(segments[0].shape[0]):
            distances[cur_point_idx, cur_segment_idx] = distance_to_segment(
                points[cur_point_idx, 0], points[cur_point_idx, 1],
                segments[0][cur_segment_idx, 0], segments[0][cur_segment_idx, 1],
                segments[1][cur_segment_idx, 0], segments[1][cur_segment_idx, 1]
            )
    return distances
