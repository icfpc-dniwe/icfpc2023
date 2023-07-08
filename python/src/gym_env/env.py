import gym
from gym import spaces
import pygame
import numpy as np
from src.solver.metric import calculate_happiness, check_positions_valid
from src.mytypes import ProblemInfo

class MusicianPlacementEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, problem_info: ProblemInfo):
        super(MusicianPlacementEnv, self).__init__()

        self.room_height = problem_info.room.height
        self.room_width = problem_info.room.width
        self.stage_height = problem_info.stage.height
        self.stage_width = problem_info.stage.width
        self.stage_bottom_left = problem_info.stage.bottom_x, problem_info.stage.bottom_y
        xmin = self.stage_bottom_left[0]
        ymin = self.stage_bottom_left[1]
        xmax = xmin + self.stage_width
        ymax = ymin + self.stage_height
        self.musicians = problem_info.musicians
        self.attendees = problem_info.attendees

        self.num_musicians = len(self.musicians)
        self.num_attendees = len(self.attendees)

        self.musician_placements = None
        self.attendee_placements = np.array([(a.x, a.y) for a in self.attendees])
        self.attendee_tastes = np.array([[taste for taste in a.tastes] for a in self.attendees])
        self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        mus_low = np.tile(np.array([[xmin, ymin]]), (self.num_musicians, 1))
        mus_high = np.tile(np.array([[xmax, ymax]]), (self.num_musicians, 1))
        att_high = np.tile(np.array([[self.room_width, self.room_height]]), (self.num_attendees, 1))
        self.observation_space = spaces.Dict({
            'musician_placements': spaces.Box(low=mus_low, high=mus_high,
                                              shape=(self.num_musicians, 2), dtype=np.float32),
            'attendee_placements': spaces.Box(low=0, high=att_high,
                                              shape=(self.num_attendees, 2), dtype=np.float32),
            'attendee_happiness': spaces.Box(low=-1e6, high=1e6, shape=(self.num_attendees,), dtype=np.float32)
        })

        self.screen_width = 800
        self.screen_height = 600

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

        self.radius = 10
        self.stage_color = self.colors['gray']
        self.attendee_color = self.colors['blue']
        self.musician_color = self.colors['green']
        self.just_placed_color = self.colors['red']

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize musician placements randomly
        initial_musicians = 0
        if options is None:
            initial_musicians = 0
        elif options['place_musicians']:
            initial_musicians = np.random.randint(0, self.num_musicians - 1)
        if initial_musicians > 0:
            self.musician_placements = self.generate_valid_placements(initial_musicians)
            self.musician_placements[:, 0] = self.musician_placements[:, 0] * self.stage_width + self.stage_bottom_left[0]
            self.musician_placements[:, 1] = self.musician_placements[:, 1] * self.stage_height + self.stage_bottom_left[1]
        else:
            self.musician_placements = np.zeros((0, 2), dtype=np.float32)

        observation = {
            'musician_placements': self.musician_placements.copy(),
            'attendee_placements': self.attendee_placements.copy(),
            'attendee_happiness': calculate_happiness(
                self.musician_placements,
                self.musicians[:initial_musicians],
                self.attendee_placements,
                self.attendee_tastes,
                reduce='none'
            )
        }
        info = {}
        return observation, info

    def step(self, action):
        xmin, ymin = self.stage_bottom_left
        xmax, ymax = xmin + self.stage_width, ymin + self.stage_height
        next_placement = action
        next_placement[0] = next_placement[0] * (self.stage_width - 20) + xmin + 10
        next_placement[1] = next_placement[1] * (self.stage_height - 20) + ymin + 10
        # print(self.musician_placements.shape)
        # print(next_placement)
        self.musician_placements = np.concatenate((self.musician_placements, [next_placement]), axis=0)
        # print(self.musician_placements.shape)
        if not check_positions_valid(self.musician_placements, xmin, xmax, ymin, ymax):
            reward = -1e7
            done = True
            attendee_happiness = np.zeros((len(self.attendees),))
        else:
            attendee_happiness = calculate_happiness(
                self.musician_placements,
                self.musicians[:len(self.musician_placements)],
                self.attendee_placements,
                self.attendee_tastes,
                reduce='none'
            )
            reward = attendee_happiness.sum()
            done = len(self.musician_placements) >= len(self.musicians)

        # Assemble the observation
        observation = {
            'musician_placements': self.musician_placements.copy(),
            'attendee_placements': self.attendee_placements.copy(),
            'attendee_happiness': attendee_happiness
        }

        return observation, reward, done, {}

    def render(self):
        self.screen.fill(self.colors['white'])

        # Draw room
        pygame.draw.rect(self.screen, self.colors['black'],
                         (0, 0, self.room_width, self.room_height), 2)

        # Draw stage
        stage_x = self.stage_bottom_left[0]
        # stage_y = self.room_height - self.stage_bottom_left[1] - self.stage_height
        stage_y = self.stage_bottom_left[1]
        pygame.draw.rect(self.screen, self.stage_color,
                         (stage_x, stage_y, self.stage_width, self.stage_height))

        # Draw musicians
        for i, musician in enumerate(self.musician_placements):
            musician_x = int(self.musician_placements[i, 0])
            musician_y = int(self.musician_placements[i, 1])
            pygame.draw.circle(self.screen, self.musician_color, (musician_x, musician_y), self.radius)
        # Draw attendees
        for i, attendee in enumerate(self.attendee_placements):
            attendee_x = int(attendee[0])
            attendee_y = int(attendee[1])
            pygame.draw.circle(self.screen, self.attendee_color, (attendee_x, attendee_y), self.radius)

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
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
        return np.array(placed)
