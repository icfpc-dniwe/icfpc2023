import numpy as np
import scipy as sp
from scipy.spatial.distance import cdist
import typing as t


def distance_singular_sqr(place_a: np.ndarray, place_b: np.ndarray) -> float:
    return float(((place_a - place_b) ** 2).sum())


def distance(places_a: np.ndarray, places_b: np.ndarray) -> np.ndarray:
    return cdist(places_a, places_b, 'euclidean')
