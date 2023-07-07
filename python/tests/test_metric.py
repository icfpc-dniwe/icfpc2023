import numpy as np
import pytest
from src.solver.metric import calculate_happiness


def test_calculate_happiness():
    eps = 1e-7
    musician_pos = np.array([
        (0, 0),
        (10, 0),
        (10, 10)
    ], dtype=np.float64)
    musician_instruments = np.array([
        0,
        0,
        1
    ], dtype=np.int32)
    attendee_positions = np.array([
        (20, 0),
        (20, 20)
    ], dtype=np.float64)
    attendee_tastes = np.array([
        (1, -1),
        (-1, 1)
    ], dtype=np.float64)
    score = np.ceil(1000000 / 400)
    assert np.abs(calculate_happiness(musician_pos[:1], musician_instruments[:1],
                                      attendee_positions[:1], attendee_tastes[:1, :1]) - score) < eps
    score = np.ceil(1000000 / 100)
    assert np.abs(calculate_happiness(musician_pos[:2], musician_instruments[:2],
                                      attendee_positions[:1], attendee_tastes[:1, :1]) - score) < eps
    score = np.ceil(1000000 / 100) + np.ceil(-1000000 / (np.sqrt(2) * 100)) + np.ceil(1000000 / (np.sqrt(2) * 100))
    assert np.abs(calculate_happiness(musician_pos, musician_instruments,
                                      attendee_positions, attendee_tastes) - score) < eps
    example_musicians_pos = np.array([
        (590, 10),
        (1100, 100),
        (1100, 150)
    ], dtype=np.float64)
    example_instruments = np.array([0, 1, 0], dtype=np.int32)
    example_attendee_positions = np.array([
        (100, 500),
        (200, 1000),
        (1100, 800)
    ], dtype=np.float64)
    example_tastes = np.array([
        (1000, -1000),
        (200, 200),
        (800, 1500)
    ], dtype=np.float64)
    example_result = 5343
    assert np.isclose(calculate_happiness(example_musicians_pos, example_instruments,
                                          example_attendee_positions, example_tastes), example_result)
