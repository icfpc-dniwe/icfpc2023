import numpy as np


def get_line_coefficients(left_points: np.ndarray, right_points: np.ndarray) -> np.ndarray:
    c = (left_points[:, 0] - right_points[:, 0]) * left_points[:, 1]\
        + (right_points[:, 1] - left_points[:, 1]) * left_points[:, 0]
    return np.concatenate((left_points[:, 1] - right_points[: 1],
                           right_points[:, 0] - left_points[:, 0],
                           c), axis=1)
