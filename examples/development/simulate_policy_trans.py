import argparse
import json
import os
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from softlearning.environments.utils import get_environment_from_params
from softlearning import policies
from softlearning.samplers import rollouts
from softlearning.utils.tensorflow import set_gpu_memory_growth
from softlearning.utils.video import save_video

try:
    from .main import ExperimentRunner
except ImportError:
    from main import ExperimentRunner


########################################################################
# Rather than collecting transitions from X episodes, use this script to
# explicitly collect Y transitions.
########################################################################


DEFAULT_RENDER_KWARGS = {
    'mode': 'none',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path',
                        type=str,
                        help='Path to the checkpoint.')
    parser.add_argument('--max-path-length',
                        '-l',
                        type=int,
                        default=1000,
                        help='Maximum episode length.')
    parser.add_argument('--n_trans', '-t', type=int, help='Number of transitions to collect.')
    parser.add_argument('--steps', '-s', type=str, help='Number of steps the SAC policy was trained for.')
    parser.add_argument('--iteration', '-i', type=int, help='SAC policy iteration')
    parser.add_argument('--render-kwargs', '-r',
                        type=json.loads,
                        default='{}',
                        help="Kwargs for rollouts renderer.")
    parser.add_argument('--rollout-save-path',
                        type=Path,
                        default=None)

    args = parser.parse_args()

    return args


def load_variant_progress_metadata(checkpoint_path):
    checkpoint_path = checkpoint_path.rstrip('/')
    trial_path = os.path.dirname(checkpoint_path)

    variant_path = os.path.join(trial_path, 'params.pkl')
    with open(variant_path, 'rb') as f:
        variant = pickle.load(f)

    metadata_path = os.path.join(checkpoint_path, ".tune_metadata")
    if os.path.exists(metadata_path):
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
    else:
        metadata = None

    progress_path = os.path.join(trial_path, 'progress.csv')
    progress = pd.read_csv(progress_path)

    return variant, progress, metadata


def load_environment(variant):
    environment_params = (
        variant['environment_params']['training']
        if 'evaluation' in variant['environment_params']
        else variant['environment_params']['training'])

    environment = get_environment_from_params(environment_params)
    return environment


def load_policy(checkpoint_dir, variant, environment):
    policy_params = variant['policy_params'].copy()
    policy_params['config'] = {
        **policy_params['config'],
        'action_range': (environment.action_space.low,
                         environment.action_space.high),
        'input_shapes': environment.observation_shape,
        'output_shape': environment.action_shape,
    }

    policy = policies.get(policy_params)

    policy_save_path = ExperimentRunner._policy_save_path(checkpoint_dir)
    status = policy.load_weights(policy_save_path)
    status.assert_consumed().run_restore_ops()

    return policy


def simulate_policy(checkpoint_path,
                    n_trans,
                    steps,
                    iteration,
                    max_path_length,
                    render_kwargs,
                    rollout_save_path,
                    ):
    checkpoint_path = os.path.abspath(checkpoint_path.rstrip('/'))
    variant, progress, metadata = load_variant_progress_metadata(
        checkpoint_path)
    environment = load_environment(variant)
    policy = load_policy(checkpoint_path, variant, environment)
    render_kwargs = {**DEFAULT_RENDER_KWARGS, **render_kwargs}

    rollout_dir = os.path.normpath(checkpoint_path).split(os.sep)[-1]
    cp_rollout_save_path = os.path.join(rollout_save_path, rollout_dir)

    if os.path.isdir(cp_rollout_save_path):
        raise FileExistsError('Please delete existing rollout dir.')
    else:
        os.makedirs(cp_rollout_save_path)

    # Save the model path that was used to create the data
    with open(os.path.join(cp_rollout_save_path, f'sac_policy_model_path_P{iteration}.txt'), 'w') as f:
        f.write(checkpoint_path)

    t_count = 0
    fin_arr = None
    while t_count < n_trans:
        paths = rollouts(
            10,
            environment,
            policy,
            path_length=max_path_length,
            render_kwargs=render_kwargs
        )

        for _, path in enumerate(paths):
            observations = path["observations"]["observations"]
            actions = path["actions"]
            next_observations = path["next_observations"]["observations"]
            rewards = path["rewards"]
            terminals = path["terminals"]
            policies = np.zeros((observations.shape[0], 1))

            arr = np.hstack((observations, actions, next_observations, rewards, terminals, policies))

        if fin_arr is None:
            fin_arr = arr
        else:
            fin_arr = np.vstack((fin_arr, arr))

        if len(fin_arr) > n_trans:
            fin_arr = fin_arr[:n_trans,:]

        t_count = len(fin_arr)

    assert len(fin_arr) == n_trans
    np.save(os.path.join(cp_rollout_save_path, f'SAC-RT-{steps}M-{iteration}-P0_{n_trans}.npy'), fin_arr)


if __name__ == '__main__':
    set_gpu_memory_growth(True)
    args = parse_args()
    simulate_policy(**vars(args))
