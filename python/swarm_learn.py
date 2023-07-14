from pathlib import Path
import numpy as np
from src.io.data import read_problem
from src.solver.placement import generate_compliant_positions, mutate_positions
from src.solver.metric import calculate_happiness
from src.solver.swarm import swarm_step


def swarm_learn():
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
    min_delta = -10000
    for cur_iter in range(swarm_iters):
        cur_gradient_step = gradient_step * np.cos(np.pi * cur_iter / swarm_iters)
        res = swarm_step(positions, bounds, metric,
                         gradient_step=cur_gradient_step, max_step=max_step, **metric_params)
        if res is None:
            print('error on step', cur_iter)
            break
        new_positions, new_metric = res
        metric_delta = (new_metric.sum() - metric.sum())
        print('Iter', cur_iter, 'pos delta', np.abs(new_positions - positions).max(),
              'metric delta', metric_delta, (new_metric - metric).max(), 'new metric', new_metric.sum())
        if metric_delta < min_delta:
            print('Aborting')
            break
        metric = new_metric
        positions = new_positions


def anneal_learn():
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
    initial_metric = calculate_happiness(initial_positions, **metric_params, reduce='sum')
    print('Initial metric', initial_metric)
    metric = initial_metric
    positions = initial_positions
    anneal_iters = 1_000
    min_delta = -10000
    metric_coeff = 1e-5
    all_positions = [initial_positions]
    all_metrics = [initial_metric * metric_coeff]
    best_metric = all_metrics[0]
    for cur_iter in range(anneal_iters):
        # choose position to mutate
        weights = np.array(all_metrics)
        weights = weights / np.sum(weights)
        weights = np.cumsum(weights)
        coin = np.random.rand()
        cur_idx = 0
        while weights[cur_idx] < coin:
            cur_idx += 1
        cur_idx = min(cur_idx, len(all_positions) - 1)
        # mutate
        old_pos = all_positions[cur_idx]
        old_metric = all_metrics[cur_idx]
        new_pos = mutate_positions(old_pos, bounds, 1 + max_step * np.cos(np.pi * cur_iter / anneal_iters))
        if new_pos is None:
            continue
        metric = calculate_happiness(new_pos, **metric_params, reduce='sum') * metric_coeff
        prob = np.exp(-(old_metric - metric) / (1 - (cur_iter / anneal_iters)))
        # print('Pr', prob)
        is_added = False
        if metric > np.max(all_metrics):
            # print('New best')
            all_positions.append(new_pos)
            all_metrics.append(metric)
            is_added = True
        elif prob > np.random.rand():
            all_positions.append(new_pos)
            all_metrics.append(metric)
            is_added = True
        # drop positions
        to_delete = []
        best_metric = np.max(all_metrics)
        for search_iter, metric_idx in enumerate(np.argsort(all_metrics)[::-1]):
            cur_metric = all_metrics[metric_idx]
            if cur_metric == best_metric:
                continue
            prob = np.exp(-(best_metric - cur_metric) / (1 - (search_iter / len(all_metrics))))
            # print(prob, best_metric, cur_metric, metric_idx, len(all_metrics))
            if prob < np.random.rand():
                to_delete.append(metric_idx)
        for cur_del in np.sort(to_delete)[::-1]:
            del all_positions[cur_del]
            del all_metrics[cur_del]
        # all_positions = all_positions[-1:]
        # all_metrics = all_metrics[-1:]
        # log
        print('Iter', cur_iter, 'pos delta', np.abs(new_pos - old_pos).max(),
              'metric delta', (metric - old_metric) / metric_coeff, 'new metric', metric / metric_coeff, 
              'is_added', is_added, 'best', np.max(all_metrics) / metric_coeff, 'dropped', len(to_delete))


if __name__ == '__main__':
    anneal_learn()
