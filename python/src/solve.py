from pathlib import Path
import numpy as np
from scipy.optimize import Bounds, milp
from src.solver.milp_solver import calculate_taste_cost_matrix, calculate_taste_cost_matrix_with_blockers
from src.solver.placement import get_placements, get_placements_hexagonal, placements_to_solution
from src.io.data import read_problem, save_solution
from src.solver.metric import check_positions_valid, calculate_happiness
from scipy.spatial.distance import cdist, pdist
from multiprocessing import Pool
from functools import partial


def solve(
        problem_id: int,
        num_recalc_step: int = 10,
        distance_coeff: float = 1e2,
        solutions_prefix: str = 'nonte',
        first_recalcs: int = 0,
        use_ext2: bool = False
):
    problem_path = Path('../problems/json/') / f'{problem_id}.json'
    save_path = Path('../solutions') / solutions_prefix / f'{problem_id}.json'
    save_path.parent.mkdir(exist_ok=True, parents=True)
    info = read_problem(problem_path)
    xs, ys = get_placements(info.stage.width, info.stage.height)
    xs += info.stage.bottom_x
    ys += info.stage.bottom_y
    placements = list(zip(xs.flatten(), ys.flatten()))
    num_musicians_per_instrument = {ins: num for ins, num in zip(*np.unique(info.musicians, return_counts=True))}
    initial_matrix = calculate_taste_cost_matrix_with_blockers(info, np.array(placements), np.zeros((0, 2)), use_ext2=use_ext2)
    top_k = len(info.musicians) * 3
    top_placements = np.argsort(initial_matrix, axis=0)[::-1, :]
    top_placements = np.unique(top_placements[:top_k].flatten())
    placements = [placements[p] for p in top_placements]
    initial_matrix = calculate_taste_cost_matrix_with_blockers(info, np.array(placements), np.zeros((0, 2)), use_ext2=use_ext2)
    im = np.argmax(initial_matrix)
    num_tastes = initial_matrix.shape[1]
    instr = im % num_tastes
    pos = im // num_tastes
    placed = [placements[pos]]
    del placements[pos]
    instruments = [instr]
    num_musicians_per_instrument[instr] -= 1
    musicians_placed = 1
    # cached_matrix = initial_matrix.copy()
    next_matrix = initial_matrix.copy()
    next_matrix = np.delete(next_matrix, pos, axis=0)
    recalc_every = max(1, len(info.musicians) // num_recalc_step)
    while musicians_placed < len(info.musicians):
        print(musicians_placed, placed[-1], instruments[-1])
        if musicians_placed % recalc_every == 0 or musicians_placed < first_recalcs:
            next_matrix = calculate_taste_cost_matrix_with_blockers(info, np.array(placements), np.array(placed), use_ext2=use_ext2)
        for ins, num in num_musicians_per_instrument.items():
            if num < 1:
                next_matrix[:, ins] = -1e8
        # placements_distance = cdist(np.array(placements), np.array(placed), 'euclidean').mean(axis=1, keepdims=True)
        # next_matrix += distance_coeff * placements_distance
        im = np.argmax(next_matrix)
        instr = im % num_tastes
        pos = im // num_tastes
        print(next_matrix[pos, instr])
        if next_matrix[pos, instr] < 1e-9:
            if next_matrix[pos, instr] < 0:
                recalc_every = 1
            else:
                recalc_every = 1e10
        placed.append(placements[pos])
        instruments.append(instr)
        del placements[pos]
        next_matrix = np.delete(next_matrix, pos, axis=0)
        musicians_placed += 1
        num_musicians_per_instrument[instr] -= 1
        assert num_musicians_per_instrument[instr] >= 0

    sorted_placements = []
    for cur_instrument in info.musicians:
        for cur_instr_idx in range(len(instruments)):
            if instruments[cur_instr_idx] == cur_instrument:
                sorted_placements.append(placed[cur_instr_idx])
                del instruments[cur_instr_idx]
                del placed[cur_instr_idx]
                break
    assert len(sorted_placements) == len(info.musicians)
    save_solution(placements_to_solution(sorted_placements), save_path)


def random_solver(problem_id: int, solutions_prefix: str = 'random'):
    problem_path = Path('../problems/json/') / f'{problem_id}.json'
    save_path = Path('../solutions') / solutions_prefix / f'{problem_id}.json'
    save_path.parent.mkdir(exist_ok=True, parents=True)
    info = read_problem(problem_path)
    positions = []
    xmin, ymin = info.stage.bottom_x, info.stage.bottom_y
    xmax, ymax = xmin + info.stage.width, ymin + info.stage.height
    for cur_musician in info.musicians:
        while True:
            new_pos_x = np.random.uniform(xmin + 10, xmax - 10)
            new_pos_y = np.random.uniform(xmin + 10, xmax - 10)
            if check_positions_valid(np.array(positions + [(new_pos_x, new_pos_y)]), xmin, xmax, ymin, ymax):
                positions.append((new_pos_x, new_pos_y))
                break
    assert len(positions) == len(info.musicians)
    save_solution(placements_to_solution(positions), save_path)


def try_milp_solver(problem_id: int, solutions_prefix: str = 'nonte', use_ext2: bool = False):
    problem_path = Path('../problems/full_round/') / f'{problem_id}.json'
    save_path = Path('../solutions') / solutions_prefix / f'{problem_id}.json'
    save_path.parent.mkdir(exist_ok=True, parents=True)
    info = read_problem(problem_path)
    xs, ys = get_placements(info.stage.width, info.stage.height)
    xs += info.stage.bottom_x
    ys += info.stage.bottom_y
    placements = list(zip(xs.flatten(), ys.flatten()))
    num_musicians_per_instrument = {ins: num for ins, num in zip(*np.unique(info.musicians, return_counts=True))}
    initial_matrix = calculate_taste_cost_matrix_with_blockers(info, np.array(placements), np.zeros((0, 2)),
                                                               use_ext2=use_ext2)
    top_k = len(info.musicians) * 3
    top_placements = np.argsort(initial_matrix, axis=0)[::-1, :]
    top_placements = np.unique(top_placements[:top_k].flatten())
    new_placements = np.array([placements[p] for p in top_placements])


if __name__ == '__main__':
    # for problem_id in range(1, 11):
    #     solve(problem_id, 25)
    with Pool(6) as p:
        for _ in map(
                partial(solve,
                        num_recalc_step=1,
                        first_recalcs=0,
                        use_ext2=False,
                        solutions_prefix='recal_step_1_all'),
                range(18, 19)
        ):
            pass
