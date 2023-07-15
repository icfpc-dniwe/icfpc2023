from pathlib import Path
import numpy as np
import pygame
from src.io.data import read_problem, load_solution
from src.solver.placement import generate_compliant_positions, mutate_positions, jiggle_positions
from src.solver.metric import calculate_happiness
from src.solver.swarm import swarm_step, attendee_gradient


class GameEnv:
    
    def __init__(self, room_height, room_width, stage_width, stage_height, stage_bottom_left, attendee_placements, pillars):
        self.room_height = room_height
        self.room_width = room_width
        self.stage_width = stage_width
        self.stage_height = stage_height
        self.stage_bottom_left = stage_bottom_left
        # self.musician_placements = musician_placements
        self.attendee_placements = attendee_placements
        self.pillars = pillars
        
        self.screen_width = 800
        self.screen_height = 600
        self.render_scale = min(self.screen_height / self.stage_height, self.screen_width / self.stage_width)

        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()

        self.colors = {
            'white': (255, 255, 255),
            'gray': (100, 100, 100),
            'black': (0, 0, 0),
            'red': (255, 0, 0),
            'blue': (0, 0, 255),
            'green': (0, 255, 0),
            'yellow': (255, 255, 0),
            'purple': (255, 0, 255)
        }

        self.radius = 5
        self.stage_color = self.colors['gray']
        self.attendee_color = self.colors['blue']
        self.musician_color = self.colors['green']
        self.pillar_color = self.colors['yellow']
        self.just_placed_color = self.colors['red']
    
    def render_step(self, new_positions):
        self.musician_placements = new_positions
        self.screen.fill(self.colors['white'])

        # Draw room
        # pygame.draw.rect(self.screen, self.colors['black'],
        #                  (0, 0, int(self.render_scale * self.room_width), int(self.render_scale * self.room_height)), 2)

        # Draw stage
        stage_x = 0  # self.stage_bottom_left[0]
        stage_y = 0  # self.stage_bottom_left[1]
        pygame.draw.rect(self.screen, self.stage_color,
                         (int(self.render_scale * stage_x),
                          int(self.render_scale * stage_y),
                          int(self.render_scale * self.stage_width),
                          int(self.render_scale * self.stage_height)))

        # Draw musicians
        for i, musician in enumerate(self.musician_placements):
            musician_x = self.musician_placements[i, 0] - self.stage_bottom_left[0]
            musician_y = self.musician_placements[i, 1] - self.stage_bottom_left[1]
            pygame.draw.circle(self.screen, self.musician_color, (int(self.render_scale * musician_x),
                                                                  int(self.render_scale * musician_y)),
                               self.radius)
        # Draw attendees
        # for i, attendee in enumerate(self.attendee_placements):
        #     attendee_x = attendee[0]
        #     attendee_y = attendee[1]
        #     pygame.draw.circle(self.screen, self.attendee_color, (int(self.render_scale * attendee_x),
        #                                                           int(self.render_scale * attendee_y)),
        #                        self.radius)
        # Draw pillars
        # for i, pillar in enumerate(self.pillars):
        #     pillar_x = pillar[0]
        #     pillar_y = pillar[1]
        #     pygame.draw.circle(self.screen, self.pillar_color, (int(self.render_scale * pillar_x),
        #                                                         int(self.render_scale * pillar_y)),
        #                        int(self.render_scale * pillar[2]))

        pygame.display.flip()
        self.clock.tick(60)
    
    def close(self):
        pygame.display.quit()
        pygame.quit()


def swarm_learn():
    info = read_problem(Path('../problems/json/18.json'))
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
    problem_id = 1
    info = read_problem(Path(f'../problems/json/{problem_id}.json'))
    solution = load_solution(Path(f'../solutions/recal_step_7_vol/{problem_id}.json'))
    initial_placements = np.array([[p.x, p.y] for p in solution.placements], dtype=np.float32)
    instruments = np.array(info.musicians, dtype=np.int32)
    bounds = (info.stage.bottom_x,
              info.stage.bottom_y,
              info.stage.bottom_x + info.stage.width,
              info.stage.bottom_y + info.stage.height)
    max_step = min(info.stage.width, info.stage.height) * 0.2
    attendees = np.array([(a.x, a.y) for a in info.attendees])
    attendee_tastes = np.array([[t for t in a.tastes] for a in info.attendees], dtype=np.float64)
    initial_positions = initial_placements  # np.array(generate_compliant_positions(bounds, len(info.musicians)))
    metric_params = dict(instruments=instruments, attendees=attendees, attendee_tastes=attendee_tastes)
    initial_metric = calculate_happiness(initial_positions, **metric_params, reduce='sum')
    print('Initial metric', initial_metric)
    metric = initial_metric
    positions = initial_positions
    anneal_iters = 1000
    min_delta = -10000
    metric_coeff = 1e-7
    all_positions = [initial_positions]
    all_metrics = [initial_metric * metric_coeff]
    best_metric = all_metrics[0]
    renderer = GameEnv(
        room_height=info.room.height, 
        room_width=info.room.width, 
        stage_width=info.stage.width, 
        stage_height=info.stage.height, 
        stage_bottom_left=(info.stage.bottom_x, info.stage.bottom_y), 
        attendee_placements=attendees, 
        pillars=[]
    )
    renderer.render_step(initial_positions)
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
        mut_pos = mutate_positions(old_pos, bounds, 1 + max_step * np.cos(np.pi / 2 * cur_iter / anneal_iters), 10000)
        if mut_pos is None:
            # continue
            mut_pos = old_pos
        force = attendee_gradient(mut_pos, **metric_params)
        force_norm = np.linalg.norm(force, axis=-1)
        print(np.max(force_norm), np.min(force_norm), np.mean(force_norm))
        new_pos, _ = jiggle_positions(mut_pos + force * metric_coeff * 10 * np.cos(np.pi / 2 * cur_iter / anneal_iters), bounds)
        if new_pos is None:
            continue
        # new_pos = mut_pos
        metric = calculate_happiness(new_pos, **metric_params, reduce='sum') * metric_coeff
        prob = np.exp(-(old_metric - metric) / (1 - (cur_iter / anneal_iters)))
        # print('Pr', prob)
        is_added = False
        if metric > np.max(all_metrics):
            print('New best')
        if metric > old_metric or prob > np.random.rand():
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
        renderer.render_step(new_pos)
        print('Iter', cur_iter, 'pos delta', np.abs(new_pos - old_pos).max(),
              'metric delta', (metric - old_metric) / metric_coeff, 'new metric', metric / metric_coeff, 
              'is_added', is_added, 'best', np.max(all_metrics) / metric_coeff, 'dropped', len(to_delete),
              'num_pos', len(all_metrics))
    renderer.close()


if __name__ == '__main__':
    anneal_learn()
