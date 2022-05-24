#! /bin/bash

# Run the example simulation job
# Need to run this on a remote desktop
python -m examples.development.simulate_policy \
    "/home/ajc348/rds/hpc-work/softlearning/gym/HalfCheetah/v2/2022-05-17T18-38-36-half_cheetah_v2_3M/id=31acc_00000-seed=9479/checkpoint_120" \
    --max-path-length 1000 \
    --num-rollouts 1 \
    --render-kwargs '{"mode": "human"}'
