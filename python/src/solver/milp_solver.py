import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import LinearConstraint, Bounds
from src.mytypes import ProblemInfo
from src.solver.geometry import distances_to_segments
from src.solver.placement import get_placements
import typing as t


def calculate_taste_cost_matrix(problem_info: ProblemInfo, placements: t.Optional[np.ndarray] = None) -> np.ndarray:
    if placements is None:
        xs, ys = get_placements(problem_info.stage.width, problem_info.stage.height)
        xs += problem_info.stage.bottom_x
        ys += problem_info.stage.bottom_y
        placements = np.array(list(zip(xs.flatten(), ys.flatten())), dtype=np.float64, copy=True)
    attendee_placements = np.array([(a.x, a.y) for a in problem_info.attendees])
    attendee_tastes = np.array([[taste for taste in a.tastes] for a in problem_info.attendees])
    num_tastes = attendee_tastes.shape[1]
    distance_matrix = cdist(placements, attendee_placements, 'euclidean') ** 2
    taste_cost = np.zeros((distance_matrix.shape[0], num_tastes), dtype=np.float64)
    for cur_taste in range(num_tastes):
        taste_cost[:, cur_taste] = np.ceil(1000000 * attendee_tastes[:, cur_taste] / distance_matrix).sum(axis=1)
    return taste_cost


def calculate_taste_cost_matrix_with_blockers(
        problem_info: ProblemInfo,
        placements: np.ndarray,
        already_placed: np.ndarray
) -> np.ndarray:
    attendee_placements = np.array([(a.x, a.y) for a in problem_info.attendees])
    attendee_tastes = np.array([[taste for taste in a.tastes] for a in problem_info.attendees])
    num_tastes = attendee_tastes.shape[1]
    distance_matrix = cdist(placements, attendee_placements, 'euclidean') ** 2
    musician_not_blocked = np.zeros(distance_matrix.shape, dtype=np.bool_)
    for cur_attendee_idx in range(len(attendee_placements)):
        cur_not_blocked = np.all(distances_to_segments(
            already_placed,
            (placements, np.repeat(attendee_placements[cur_attendee_idx][np.newaxis, :],
                                   placements.shape[0]).reshape((2, placements.shape[0])).T)
        ) > 5, axis=0)
        musician_not_blocked[:, cur_attendee_idx] = cur_not_blocked
    taste_cost = np.zeros((distance_matrix.shape[0], num_tastes), dtype=np.float64)
    for cur_taste in range(num_tastes):
        taste_cost[:, cur_taste] = (np.ceil(1000000 * attendee_tastes[:, cur_taste] / distance_matrix)
                                    * musician_not_blocked).sum(axis=1)
    return taste_cost


def create_constraints(musicians: t.Sequence[int], num_placements: int) -> t.List[LinearConstraint]:
    num_musicians = len(musicians)
    num_musicians_per_instrument = {inst: num for inst, num in zip(*np.unique(musicians, return_counts=True))}
    num_tastes = len(num_musicians_per_instrument)
    total_number = LinearConstraint(np.ones(num_placements * num_tastes, dtype=np.bool_),
                                        np.array(num_musicians), np.array(num_musicians))
    instrument_constraints = []
    for cur_instrument, cur_num in num_musicians_per_instrument.items():
        matrix = np.zeros((num_placements, num_tastes), dtype=np.bool_)
        matrix[:, cur_instrument] = 1
        instrument_constraints.append(LinearConstraint(matrix.reshape(1, -1),
                                                       np.zeros(1), np.array(cur_num)))
    placement_constraints = []
    for cur_place_idx in range(num_placements):
        matrix = np.zeros((num_placements, num_tastes), dtype=np.bool_)
        matrix[cur_place_idx, :] = 1
        placement_constraints.append(LinearConstraint(matrix.reshape(1, -1),
                                                      np.zeros(1), np.ones(1)))
    return instrument_constraints + placement_constraints + [total_number]
