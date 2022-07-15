#! /bin/bash

# Run the example simulation job
# Need to run this on a remote desktop
python -m examples.development.simulate_policy \
    "/home/ajc348/rds/hpc-work/dogo_results/mopo/ray_mopo/HalfCheetah/halfcheetah_d3rlpy_rt_2m_3_1000e3/seed:1234_2022-07-03_21-32-15newc_0je/ray_mopo/HalfCheetah/halfcheetah_d3rlpy_rt_2m_3_1000e3/seed:1234_2022-07-03_21-32-15newc_0je/checkpoint_501" \
    --max-path-length 1000 \
    --num-rollouts 1 \
    --render-kwargs '{"mode": "human"}'
