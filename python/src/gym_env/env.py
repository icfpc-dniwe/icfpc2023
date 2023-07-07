import gym
from gym import spaces
import pygame
import numpy as np

class MusicianPlacementEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, room_height, room_width, stage_height, stage_width, stage_bottom_left, musicians, attendees):
        super(MusicianPlacementEnv, self).__init__()

        self.room_height = room_height
        self.room_width = room_width
        self.stage_height = stage_height
        self.stage_width = stage_width
        self.stage_bottom_left = stage_bottom_left
        self.musicians = musicians
        self.attendees = attendees

        self.num_musicians = len(musicians)
        self.num_attendees = len(attendees)

        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_musicians, 2), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'placements': spaces.Box(low=0, high=1, shape=(self.num_musicians, 2), dtype=np.float32),
            'attendee_happiness': spaces.Box(low=0, high=1, shape=(self.num_attendees,), dtype=np.float32)
        })

        self.screen_width = 800
        self.screen_height = 600

        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()

        self.colors = {
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'red': (255, 0, 0),
            'blue': (0, 0, 255),
            'green': (0, 255, 0),
            'yellow': (255, 255, 0),
            'purple': (255, 0, 255)
        }

        self.radius = 10
        self.stage_color = self.colors['green']
        self.musician_colors = [self.colors['red'], self.colors['blue'], self.colors['yellow'], self.colors['purple']]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize musician placements randomly
        self.placements = np.random.uniform(low=0, high=1, size=(self.num_musicians, 2))

        return {
            'placements': self.placements.copy(),
            'attendee_happiness': np.zeros(self.num_attendees)
        }

    def step(self, action):
        # Update musician placements based on the action
        self.placements = action

        # Compute attendee happiness
        attendee_happiness = np.zeros(self.num_attendees)
        for i, attendee in enumerate(self.attendees):
            for k, musician in enumerate(self.musicians):
                d = ((attendee['x'] - self.placements[k, 0]) ** 2) + ((attendee['y'] - self.placements[k, 1]) ** 2)
                blocked = False
                for k_prime, other_musician in enumerate(self.musicians):
                    if k_prime != k:
                        d_prime = ((other_musician['x'] - self.placements[k_prime, 0]) ** 2) + ((other_musician['y'] - self.placements[k_prime, 1]) ** 2)
                        if d_prime <= 5 ** 2 and self.check_line_intersection(self.placements[k], attendee, self.placements[k_prime]):
                            blocked = True
                            break
                if not blocked:
                    impact = 1_000_000 * attendee['tastes'][musician['instrument']]
                    attendee_happiness[i] += impact / d**2

        # Normalize attendee happiness
        attendee_happiness /= np.max(attendee_happiness) if np.max(attendee_happiness) > 0 else 1

        # Assemble the observation
        observation = {
            'placements': self.placements.copy(),
            'attendee_happiness': attendee_happiness
        }

        # Compute reward (optional)
        reward = np.mean(attendee_happiness)

        # Check if the episode is done (optional)
        done = False

        return observation, reward, done, {}

    def render(self):
        self.screen.fill(self.colors['white'])

        # Draw room
        pygame.draw.rect(self.screen, self.colors['black'],
                         (0, 0, self.room_width, self.room_height), 2)

        # Draw stage
        stage_x = self.stage_bottom_left[0]
        stage_y = self.room_height - self.stage_bottom_left[1] - self.stage_height
        pygame.draw.rect(self.screen, self.stage_color,
                         (stage_x, stage_y, self.stage_width, self.stage_height))

        # Draw musicians
        for i, musician in enumerate(self.musicians):
            musician_x = int(self.placements[i, 0] * self.stage_width) + stage_x
            musician_y = int(self.placements[i, 1] * self.stage_height) + stage_y
            pygame.draw.circle(self.screen, self.musician_colors[i], (musician_x, musician_y), self.radius)

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        pygame.quit()

    def check_line_intersection(self, p1, p2, p3, p4):
        def ccw(p1, p2, p3):
            return (p3[1] - p1[1]) * (p2[0] - p1[0]) > (p2[1] - p1[1]) * (p3[0] - p1[0])

        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
