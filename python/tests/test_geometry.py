import numpy as np
from src.solver.geometry import get_line_coefficients, distance_to_line, distance_to_segment, distances_to_segments
import pytest


def test_get_line_coefficients():
    points = np.array([
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1)
    ], dtype=np.float64)
    line_coefficients = get_line_coefficients(points[:-1], points[1:]).astype(np.int32)
    assert line_coefficients[0, 1] == 0
    assert line_coefficients[0, 2] == 0
    assert line_coefficients[1, 0] == line_coefficients[1, 1]
    assert line_coefficients[1, 0] == -line_coefficients[1, 2]
    assert line_coefficients[2, 1] == 0
    assert line_coefficients[2, 0] == -line_coefficients[2, 2]


def test_distance_to_segment():
    eps = 1e-7
    assert np.abs(distance_to_segment(0, 0, 0, 0, 1, 1)) < eps
    assert np.abs(distance_to_segment(0, 0, 1, 1, -1, -1)) < eps
    assert np.abs(distance_to_segment(0, 0, 1, 0, 0, 1) - (np.sqrt(2) / 2)) < eps
    assert np.abs(distance_to_segment(0, 0, 1, 0, 1, 1) - 1) < eps


def test_distances_to_segments():
    eps = -7
    zero_point = np.array((0, 0), dtype=np.float64).reshape(1, 2)
    segment_first = np.array([
        (0, 0),
        (1, 1),
        (1, 0),
        (1, 0)
    ], dtype=np.float64)
    segment_second = np.array([
        (1, 1),
        (-1, -1),
        (0, 1),
        (1, 1)
    ], dtype=np.float64)
    real_distances = np.array([0, 0, np.sqrt(2) / 2, 1], dtype=np.float64)
    assert np.allclose(distances_to_segments(zero_point, (segment_first, segment_second)), real_distances)
    assert np.allclose(distances_to_segments(zero_point, (segment_second, segment_first)), real_distances)
