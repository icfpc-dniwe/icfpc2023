import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from src.solver.metric import calculate_happiness, check_positions_valid
from src.mytypes import ProblemInfo

class MusicianPlacementEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(
            self,
            problem_info: ProblemInfo,
            render_mode=None,
            calculate_happiness_for_all: bool = False,
            reward_mult: float = 1e-7,
            initial_placements=None,
            skip_instead_of_fail: bool = False
    ):
        super(MusicianPlacementEnv, self).__init__()

        self.render_mode = render_mode
        self.room_height = problem_info.room.height
        self.room_width = problem_info.room.width
        self.stage_height = problem_info.stage.height
        self.stage_width = problem_info.stage.width
        self.stage_bottom_left = problem_info.stage.bottom_x, problem_info.stage.bottom_y
        xmin = self.stage_bottom_left[0]
        ymin = self.stage_bottom_left[1]
        xmax = xmin + self.stage_width
        ymax = ymin + self.stage_height
        self.musicians = np.array(problem_info.musicians, dtype=np.int32)
        self.attendees = problem_info.attendees
        self.calculate_happiness_for_all = calculate_happiness_for_all
        self.reward_mult = reward_mult
        self.skip_instead_of_fail = skip_instead_of_fail
        self.initial_placements = initial_placements

        self.num_musicians = len(self.musicians)
        self.num_attendees = len(self.attendees)

        self.musicians_placed = 0
        self.attendee_placements = np.array([(a.x, a.y) for a in self.attendees]).astype(np.float32)
        self.attendee_tastes = np.array([[taste for taste in a.tastes] for a in self.attendees]).astype(np.float32)
        self.pillars = np.array([(p.x, p.y, p.radius) for p in problem_info.pillars], dtype=np.float32)
        if self.initial_placements is None:
            self.musician_placements = None  # self.generate_valid_placements(self.num_musicians)
        else:
            self.musician_placements = self.initial_placements.copy()
        self.prev_reward = None
        mus_low = np.tile(np.array([[xmin, ymin]]), (self.num_musicians, 1))
        mus_high = np.tile(np.array([[xmax, ymax]]), (self.num_musicians, 1))
        att_high = np.tile(np.array([[self.room_width, self.room_height]]), (self.num_attendees, 1))
        pil_high = np.tile(np.array([[self.room_width, self.room_height, 100]]), (len(self.pillars), 1))
        self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'musicians_placed': spaces.Discrete(len(self.musicians)),
            # 'musician_instruments': spaces.Box(low=0, high=np.max(self.musicians),
            #                                    shape=(self.num_musicians,), dtype=np.int32),
            'musician_placements': spaces.Box(low=mus_low, high=mus_high,
                                              shape=(self.num_musicians, 2), dtype=np.float32),
            'attendee_placements': spaces.Box(low=0, high=att_high,
                                              shape=(self.num_attendees, 2), dtype=np.float32),
            'attendee_happiness': spaces.Box(low=-1, high=1, shape=(self.num_attendees,), dtype=np.float32),
            # 'pillars': spaces.Box(low=0, high=pil_high, shape=(len(self.pillars) * 3,), dtype=np.float32),
            # 'attendee_tastes': spaces.Box(low=-1e6, high=1e6, shape=self.attendee_tastes.shape, dtype=np.float32)
        })

        self.screen = None
        if self.render_mode == 'human':
            self.screen_width = 1920
            self.screen_height = 1080
            self.render_scale = min(self.screen_height / self.room_height, self.screen_width / self.room_width)

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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize musician placements randomly
        initial_musicians = 0
        self.musicians_placed = 0
        if self.initial_placements is None:
            self.musician_placements = self.generate_valid_placements(self.num_musicians)
        else:
            self.musician_placements = self.initial_placements.copy()
        if options is None:
            initial_musicians = 0
        elif options['place_musicians']:
            initial_musicians = np.random.randint(0, self.num_musicians - 1)
        if initial_musicians > 0:
            self.musicians_placed = initial_musicians
        if self.calculate_happiness_for_all:
            happiness = calculate_happiness(
                self.musician_placements,
                self.musicians,
                self.attendee_placements,
                self.attendee_tastes,
                self.pillars,
                reduce='none'
            )
        else:
            if self.musicians_placed > 0:
                happiness = calculate_happiness(
                    self.musician_placements[:self.musicians_placed],
                    self.musicians[:self.musicians_placed],
                    self.attendee_placements,
                    self.attendee_tastes,
                    self.pillars,
                    reduce='none'
                )
            else:
                happiness = np.zeros((self.num_attendees,), dtype=np.int32)
        if not self.calculate_happiness_for_all and self.musicians_placed < 1:
            self.prev_reward = -1
        else:
            self.prev_reward = self.reward_mult * happiness.sum()
        observation = {
            'musicians_placed': self.musicians_placed,
            'musician_placements': self.musician_placements.copy(),
            # 'musician_instruments': self.musicians.copy(),
            'attendee_placements': self.attendee_placements.copy(),
            # 'pillars': self.pillars.flatten().copy(),
            # 'attendee_tastes': self.attendee_tastes.copy(),
            'attendee_happiness': happiness * 1e-9
        }
        return observation, {}

    def step(self, action):
        xmin, ymin = self.stage_bottom_left
        xmax, ymax = xmin + self.stage_width, ymin + self.stage_height
        next_placement = action
        next_placement[0] = next_placement[0] * (self.stage_width - 20) + xmin + 10
        next_placement[1] = next_placement[1] * (self.stage_height - 20) + ymin + 10
        # print(self.musician_placements.shape)
        # print(next_placement)
        self.musician_placements[self.musicians_placed] = np.array(next_placement)
        self.musicians_placed += 1
        if self.musicians_placed >= len(self.musicians):
            self.musicians_placed = 0
        # print(self.musician_placements.shape)
        if not check_positions_valid(self.musician_placements[:self.musicians_placed], xmin, xmax, ymin, ymax):
            reward = 0
            done = True
            is_success = False
            attendee_happiness = np.zeros((len(self.attendees),), dtype=np.float32)
        else:
            if self.calculate_happiness_for_all:
                attendee_happiness = calculate_happiness(
                    self.musician_placements,
                    self.musicians,
                    self.attendee_placements,
                    self.attendee_tastes,
                    self.pillars,
                    reduce='none'
                )
            else:
                attendee_happiness = calculate_happiness(
                    self.musician_placements[:self.musicians_placed],
                    self.musicians[:self.musicians_placed],
                    self.attendee_placements,
                    self.attendee_tastes,
                    self.pillars,
                    reduce='none'
                )
            reward = attendee_happiness.sum() * self.reward_mult
            tmp = reward
            reward = max(0, (reward - self.prev_reward) + 1) ** 4
            self.prev_reward = tmp
            done = False  # self.musicians_placed >= len(self.musicians)
            is_success = done

        # Assemble the observation
        observation = {
            'musicians_placed': self.musicians_placed,
            'musician_placements': self.musician_placements.copy(),
            # 'musician_instruments': self.musicians.copy(),
            'attendee_placements': self.attendee_placements.copy(),
            # 'pillars': self.pillars.flatten().copy(),
            # 'attendee_tastes': self.attendee_tastes.copy(),
            'attendee_happiness': attendee_happiness * 1e-9
        }

        if done:
            info = {'is_success': is_success}
        else:
            info = {}
        return observation, reward, done, False, info

    def render(self):
        if self.screen is not None:
            self.screen.fill(self.colors['white'])

            # Draw room
            pygame.draw.rect(self.screen, self.colors['black'],
                             (0, 0, int(self.render_scale * self.room_width), int(self.render_scale * self.room_height)), 2)

            # Draw stage
            stage_x = self.stage_bottom_left[0]
            # stage_y = self.room_height - self.stage_bottom_left[1] - self.stage_height
            stage_y = self.stage_bottom_left[1]
            pygame.draw.rect(self.screen, self.stage_color,
                             (int(self.render_scale * stage_x),
                              int(self.render_scale * stage_y),
                              int(self.render_scale * self.stage_width),
                              int(self.render_scale * self.stage_height)))

            # Draw musicians
            for i, musician in enumerate(self.musician_placements):
                musician_x = self.musician_placements[i, 0]
                musician_y = self.musician_placements[i, 1]
                pygame.draw.circle(self.screen, self.musician_color, (int(self.render_scale * musician_x),
                                                                      int(self.render_scale * musician_y)),
                                   self.radius)
            # Draw attendees
            for i, attendee in enumerate(self.attendee_placements):
                attendee_x = attendee[0]
                attendee_y = attendee[1]
                pygame.draw.circle(self.screen, self.attendee_color, (int(self.render_scale * attendee_x),
                                                                      int(self.render_scale * attendee_y)),
                                   self.radius)
            # Draw pillars
            for i, pillar in enumerate(self.pillars):
                pillar_x = pillar[0]
                pillar_y = pillar[1]
                pygame.draw.circle(self.screen, self.pillar_color, (int(self.render_scale * pillar_x),
                                                                    int(self.render_scale * pillar_y)),
                                   int(self.render_scale * pillar[2]))

            pygame.display.flip()
            self.clock.tick(60)

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

    @staticmethod
    def check_line_intersection(p1, p2, p3, p4):
        def ccw(p1, p2, p3):
            return (p3[1] - p1[1]) * (p2[0] - p1[0]) > (p2[1] - p1[1]) * (p3[0] - p1[0])

        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    def generate_valid_placements(self, num_placements: int, max_tries: int = 100):
        placed = []
        xmin, ymin = self.stage_bottom_left
        xmax, ymax = xmin + self.stage_width, ymin + self.stage_height
        for _ in range(num_placements):
            next_x = 0
            next_y = 0
            for _ in range(max_tries):
                next_x = np.random.uniform(xmin+10, xmax-10+1e-7)
                next_y = np.random.uniform(ymin+10, ymax-10+1e-7)
                if check_positions_valid(np.array(placed + [(next_x, next_y)]), xmin, xmax, ymin, ymax):
                    break
            placed.append((next_x, next_y))
        return np.array(placed, dtype=np.float32)
