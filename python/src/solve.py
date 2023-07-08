from pathlib import Path
import numpy as np
from src.solver.milp_solver import calculate_taste_cost_matrix, calculate_taste_cost_matrix_with_blockers
from src.solver.placement import get_placements, get_placements_hexagonal, placements_to_solution
from src.io.data import read_problem, save_solution
from multiprocessing import Pool
from functools import partial


def solve(problem_id: int, recalc_step: int = 10):
    problem_path = Path('../problems/json/') / f'{problem_id}.json'
    save_path = Path('../solutions/') / f'{problem_id}.json'
    info = read_problem(problem_path)
    xs, ys = get_placements(info.stage.width, info.stage.height)
    xs += info.stage.bottom_x
    ys += info.stage.bottom_y
    placements = list(zip(xs.flatten(), ys.flatten()))
    num_musicians_per_instrument = {ins: num for ins, num in zip(*np.unique(info.musicians, return_counts=True))}
    initial_matrix = calculate_taste_cost_matrix(info, np.array(placements))
    top_k = len(info.musicians) * 3
    top_placements = np.argsort(initial_matrix, axis=0)[::-1, :]
    top_placements = np.unique(top_placements[:top_k].flatten())
    placements = [placements[p] for p in top_placements]
    initial_matrix = calculate_taste_cost_matrix(info, np.array(placements))
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
    recalc_every = len(info.musicians) // recalc_step
    while musicians_placed < len(info.musicians):
        print(musicians_placed, placed[-1], instruments[-1])
        if musicians_placed % recalc_every == 0:
            next_matrix = calculate_taste_cost_matrix_with_blockers(info, np.array(placements), np.array(placed))
        for ins, num in num_musicians_per_instrument.items():
            if num < 1:
                next_matrix[:, ins] = -1e6
        im = np.argmax(next_matrix)
        instr = im % num_tastes
        pos = im // num_tastes
        print(next_matrix[pos, instr])
        if next_matrix[pos, instr] < 1e-9:
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


if __name__ == '__main__':
    # for problem_id in range(1, 11):
    #     solve(problem_id, 25)
    with Pool(4) as p:
        p.map(partial(solve, recalc_step=25), range(11, 56))
