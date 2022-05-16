#! /bin/bash

# Run the example simulation job
# Need to run this on a remote desktop
python -m examples.development.simulate_policy \
    "/home/ajc348/rds/hpc-work/softlearning/gym/HalfCheetah/v3/2022-05-16T12-29-56-my-sac-experiment-1/id=8592b_00000-seed=4378_0_num_warmup_samples=10000,evaluation={'domain': 'HalfCheetah', 'task': 'v3', 'universe': 'gym', 'kwargs'_2022-05-16_12-30-01/checkpoint_10" \
    --max-path-length 1000 \
    --num-rollouts 1 \
    --render-kwargs '{"mode": "human"}'
