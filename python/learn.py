from src.io.data import read_problem, load_solution
from src.gym_env.env import MusicianPlacementEnv
from src.gym_env.nn.extractors import MusiciansCombinedExtractor
from pathlib import Path
import numpy as np
from stable_baselines3 import A2C, PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv


if __name__ == '__main__':
    num_steps = 1_000_000
    problem_id = 10
    info = read_problem(Path(f'../problems/json/{problem_id}.json'))
    # solution = load_solution(Path(f'../solutions/recal_step_7_ext2_full_p/{problem_id}.json'))
    # initial_placements = np.array([[p.x, p.y] for p in solution.placements], dtype=np.float32)
    env_fn = lambda render_mode: MusicianPlacementEnv(
        info, render_mode=render_mode, calculate_happiness_for_all=False,
        # initial_placements=initial_placements
    )
    env = make_vec_env(env_fn, n_envs=8, vec_env_cls=SubprocVecEnv, env_kwargs={'render_mode': None})

    policy_kwargs = dict(
        # features_extractor_class=MusiciansCombinedExtractor,
        # features_extractor_kwargs=dict(features_dim=128),
        log_std_init=-2,
        ortho_init=False
    )
    params = dict(
        # normalize=True,
        # n_envs=4,
        # n_timesteps=2e6,
        ent_coef=0.0,
        max_grad_norm=2,
        n_steps=5,
        gae_lambda=0.9,
        vf_coef=0.4,
        gamma=0.99,
        use_rms_prop=False,
        normalize_advantage=False,
        learning_rate=3e-4,
        use_sde=True
    )
    
    model = A2C("MultiInputPolicy", env, policy_kwargs=policy_kwargs, device="cpu", verbose=2, **params)
    model.learn(total_timesteps=num_steps, progress_bar=True)
    model.save(f"a2c_{problem_id}_{num_steps}")
    # model.load(f"a2c_{problem_id}_{num_steps}", device='cpu')

    del env
    env = make_vec_env(env_fn, n_envs=1, env_kwargs={'render_mode': 'human'})
    obs = env.reset()
    my_result = 0  # 14021
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, _ = env.step(action)
        env.render("human")
        # print(action, rewards, dones)
        # if rewards[0] > my_result:
        #     print('New best!', rewards)
        print(action, rewards, dones)
        # input()
        if np.all(dones):
            break
    input()
