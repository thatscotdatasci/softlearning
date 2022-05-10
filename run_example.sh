softlearning run_example_local examples.development \
--gpus 1 \
--local-dir . \
--algorithm SAC \
--universe gym \
--domain HalfCheetah \
--task v3 \
--exp-name my-sac-experiment-1 \
--checkpoint-frequency 1000
