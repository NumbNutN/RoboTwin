import numpy as np
from envs.utils.parse_hdf5 import read_hdf5

# Load the data from the HDF5 file
data_dict = read_hdf5("/home/numbnut/repo/RobotTwin2/data/beat_block_hammer/demo_randomized/data/episode0.hdf5")
joint_action = data_dict["joint_action"]

print("Keys available in joint_action:", joint_action.keys())

# --- Verification Logic ---
verification_passed = False
details = []

# Check if all required keys exist
required_keys = ["vector", "left_arm", "right_arm", "left_gripper", "right_gripper"]
missing_keys = [key for key in required_keys if key not in joint_action]

if missing_keys:
    details.append(f"Verification failed: Missing keys in joint_action: {missing_keys}")
else:
    details.append("All required keys are present.")
    
    # Extract arrays
    vector = joint_action["vector"]
    left_arm = joint_action["left_arm"]
    right_arm = joint_action["right_arm"]
    left_gripper = joint_action["left_gripper"]
    right_gripper = joint_action["right_gripper"]

    try:
        # Reshape 1D gripper arrays to 2D column vectors
        left_gripper_2d = left_gripper.reshape(-1, 1)
        right_gripper_2d = right_gripper.reshape(-1, 1)

        # Concatenate the individual arrays in the specified order
        concatenated_vector = np.concatenate(
            [left_arm, left_gripper_2d, right_arm, right_gripper_2d], 
            axis=1
        )
        
        details.append(f"Shape of 'vector': {vector.shape}")
        details.append(f"Shape of concatenated array: {concatenated_vector.shape}")

        # Compare with the 'vector' array
        # if np.array_equal(vector, concatenated_vector):
        if np.allclose(vector, concatenated_vector, atol=1e-6):
            details.append("SUCCESS: Concatenated array is identical to the 'vector' array.")
            verification_passed = True
        else:
            details.append("FAILURE: Concatenated array is NOT identical to the 'vector' array.")
            
    except ValueError as e:
        details.append(f"An error occurred during concatenation: {e}")


# --- Output the final result ---
print("\n--- Verification Result ---")
if verification_passed:
    print("Yes, your assumption is correct.")
    print("`data_dict['joint_action']['vector']` is a concatenation of `left_arm`, `right_arm`, `left_gripper`, and `right_gripper`.")
else:
    print("No, your assumption appears to be incorrect based on the data.")

print("\n--- Details ---")
for detail in details:
    print(f"- {detail}")