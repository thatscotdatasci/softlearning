#! /bin/bash

# Run the example simulation job
# Need to run this on a remote desktop
python -m examples.development.simulate_policy_trans \
    "/home/ajc348/rds/hpc-work/softlearning/gym/Hopper/v2/2022-09-01T11-35-30-hopper_v2_1M/id=cea8a_00000-seed=2702/checkpoint_4" \
    --max-path-length 1000 \
    --n_trans 100000 \
    --steps 1 \
    --iteration 1 \
    --rollout-save-path .
