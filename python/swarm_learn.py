from pathlib import Path
import numpy as np
from src.io.data import read_problem
from src.solver.placement import generate_compliant_positions
from src.solver.metric import calculate_happiness
from src.solver.swarm import swarm_step


if __name__ == '__main__':
    info = read_problem(Path('../problems/json/21.json'))
    instruments = np.array(info.musicians, dtype=np.int32)
    bounds = (info.stage.bottom_x,
              info.stage.bottom_y,
              info.stage.bottom_x + info.stage.width,
              info.stage.bottom_y + info.stage.height)
    max_step = min(info.stage.width, info.stage.height) * 0.7
    attendees = np.array([(a.x, a.y) for a in info.attendees])
    attendee_tastes = np.array([[t for t in a.tastes] for a in info.attendees], dtype=np.float64)
    initial_positions = np.array(generate_compliant_positions(bounds, len(info.musicians)))
    metric_params = dict(instruments=instruments, attendees=attendees, attendee_tastes=attendee_tastes)
    initial_metric = calculate_happiness(initial_positions, **metric_params, reduce='attendee')
    print('Initial metric', initial_metric.sum())
    metric = initial_metric
    positions = initial_positions
    swarm_iters = 10_000
    gradient_step = 1
    min_delta = 2
    for cur_iter in range(swarm_iters):
        cur_gradient_step = gradient_step * np.cos(np.pi * cur_iter / swarm_iters)
        res = swarm_step(positions, bounds, metric,
                         gradient_step=cur_gradient_step, max_step=max_step, **metric_params)
        if res is None:
            print('error on step', cur_iter)
            break
        positions, new_metric = res
        metric_delta = (new_metric - metric).max()
        print('Iter', cur_iter, 'metric delta', metric_delta, 'new metric', new_metric.sum())
        if metric_delta < min_delta:
            print('Aborting')
            break
        metric = new_metric
