from pathlib import Path
import numpy as np
from src.solver.milp_solver import calculate_taste_cost_matrix, calculate_taste_cost_matrix_with_blockers
from src.solver.placement import get_placements, get_placements_hexagonal, placements_to_solution
from src.io.data import read_problem, save_solution


def solve(problem_id: int):
    problem_path = Path('../problems/json/') / f'{problem_id}.json'
    save_path = Path('../solutions/') / f'{problem_id}.json'
    info = read_problem(problem_path)
    xs, ys = get_placements(info.stage.width, info.stage.height)
    xs += info.stage.bottom_x
    ys += info.stage.bottom_y
    placements = list(zip(xs.flatten(), ys.flatten()))
    num_musicians_per_instrument = {ins: num for ins, num in zip(*np.unique(info.musicians, return_counts=True))}
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
    while musicians_placed < len(info.musicians):
        print(musicians_placed, placed[-1], instruments[-1])
        next_matrix = calculate_taste_cost_matrix_with_blockers(info, np.array(placements), np.array(placed))
        for ins, num in num_musicians_per_instrument.items():
            if num < 1:
                next_matrix[:, ins] = -1e6
        im = np.argmax(next_matrix)
        instr = im % num_tastes
        pos = im // num_tastes
        print(next_matrix[pos, instr])
        placed.append(placements[pos])
        instruments.append(instr)
        del placements[pos]
        musicians_placed += 1
        num_musicians_per_instrument[instr] -= 1
        assert num_musicians_per_instrument[instr] >= 0

    sorted_placements = []
    for cur_instrument in info.musicians:
        for cur_instr_idx in instruments:
            if instruments[cur_instr_idx] == cur_instrument:
                sorted_placements.append(placed[cur_instr_idx])
                del instruments[cur_instr_idx]
                del placed[cur_instr_idx]
                break
    assert len(sorted_placements) == len(info.musicians)
    save_solution(placements_to_solution(sorted_placements), save_path)


if __name__ == '__main__':
    solve(6)
