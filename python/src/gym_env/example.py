from src.gym_env.env import *
from src.io.data import *


info = read_problem(Path('../problems/json/1.json'))

env = MusicianPlacementEnv(info)
obs = env.reset()

done = False
while not done:
    action = np.random.uniform(low=0, high=1, size=(2,))
    obs, reward, done, _ = env.step(action)
    print(reward, done)
    env.render()
env.close()
