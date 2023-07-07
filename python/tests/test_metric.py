import numpy as np
import pytest
from src.solver.metric import calculate_happiness


def test_calculate_happiness():
    eps = 1e-7
    musician_pos = np.array([
        (0, 0),
        (1, 0),
        (1, 1)
    ], dtype=np.float64)
    musician_instruments = np.array([
        0,
        0,
        1
    ], dtype=np.int32)
    attendee_positions = np.array([
        (2, 0),
        (2, 2)
    ], dtype=np.float64)
    attendee_tastes = np.array([
        (1, -1),
        (-1, 1)
    ], dtype=np.float64)
    assert np.abs(calculate_happiness(musician_pos[:1], musician_instruments[:1],
                                      attendee_positions[:1], attendee_tastes[:1, :1]) - 0.25) < eps
    assert np.abs(calculate_happiness(musician_pos[:2], musician_instruments[:2],
                                      attendee_positions[:1], attendee_tastes[:1, :1]) - 1) < eps
