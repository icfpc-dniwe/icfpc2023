from src.io.data import read_problem
from src.gym_env.env import MusicianPlacementEnv
from pathlib import Path
import numpy as np
from stable_baselines3 import A2C, PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv


if __name__ == '__main__':
    num_steps = 1_000_000
    problem_id = 55
    info = read_problem(Path(f'../problems/json/{problem_id}.json'))
    env = make_vec_env(lambda: MusicianPlacementEnv(info, calculate_happiness_for_all=True), n_envs=8, vec_env_cls=SubprocVecEnv)
    model = A2C("MultiInputPolicy", env, device="cpu", verbose=2)
    model.learn(total_timesteps=num_steps)
    model.save(f"a2c_{problem_id}_{num_steps}")
    # model.load(f"a2c_{problem_id}_{num_steps}", device='cpu')

    del env
    env = make_vec_env(lambda: MusicianPlacementEnv(info, render_mode='human', calculate_happiness_for_all=True), n_envs=1)
    obs = env.reset()
    my_result = 33008083  # 14021
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, _ = env.step(action)
        env.render("human")
        print(action, rewards, dones)
        if rewards[0] > my_result:
            print('New best!', rewards)
        print(action, rewards, dones)
        if np.all(dones):
            break
    input()
