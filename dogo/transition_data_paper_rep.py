import os
from glob import glob
import numpy as np

####################################################################################
# Purpose: 
# Replicates the data merging technique employed in Leveraging Invariance in Offline
# RL. Takes in a collection of arrays, each representing a single epsiode. Splits
# these in half, with each being allocated to a different policy identifier (0 for 
# the first half, 1 for the second).
####################################################################################

IND_ARRAY_DIR = "/home/ajc348/rds/hpc-work/dogo_results/softlearning/gym/HalfCheetah/v2/2022-05-24T20-41-24-half_cheetah_v2_1M/id=8341c_00000-seed=8238/SAC-PAP3/checkpoint_10"
OUTPUT_DIR = "/home/ajc348/rds/hpc-work/dogo_results/softlearning/gym/HalfCheetah/v2/2022-05-24T20-41-24-half_cheetah_v2_1M/id=8341c_00000-seed=8238/SAC-PAP3/checkpoint_10"

def main():
    # First ensure that the directory containing the individual arrays does exist
    if not os.path.isdir(IND_ARRAY_DIR):
        raise FileNotFoundError('Input directory does not exist')

    # Ensure the output directory already exists
    if not os.path.exists(OUTPUT_DIR):
        raise FileNotFoundError('Please create output directory before re-running')

    # Ensure the output file does not already exist
    output_file = os.path.join(OUTPUT_DIR, f'combined_transitions.npy')
    if os.path.exists(output_file):
        raise FileExistsError('Please delete existing output file before re-running')

    # Identify the individual array files
    # Expecting there to be 101 of these 
    ind_arrs = glob(os.path.join(IND_ARRAY_DIR, 'rollout_1000_*.npy'))
    assert len(ind_arrs) == 101

    # Loop through the arrays in the directory
    # Add a column which indicates the policy the data came from
    # The first half gets a 0, the second half a 1
    for i, arr_path in enumerate(ind_arrs):
        # Load data
        trans_arr = np.load(arr_path)

        # Expecting all the arrays to have 1000 transitions
        assert trans_arr.shape[0] == 1000

        # The first array dictates the number of columns that should be present
        if i == 0:
            cols = trans_arr.shape[1]
        assert trans_arr.shape[1] == cols

        # Add a column to the array with the ID of the trajectory
        # Set this to 0 for the first 500 transitions, and 1 for the second 500
        policy_id = np.vstack((np.zeros((500,1)), np.ones((500,1))))
        trans_arr = np.hstack((trans_arr, policy_id))

        # Combine the trajectories
        if i == 0:
            final_arr = np.copy(trans_arr)
        else:
            final_arr = np.vstack((final_arr, np.copy(trans_arr)))

    # Assert the at the final array has the expected lenght
    assert final_arr.shape[0] == 1000*101

    np.save(output_file, final_arr)

if __name__ == "__main__":
    main()
