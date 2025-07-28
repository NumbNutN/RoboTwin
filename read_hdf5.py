import numpy as np
from envs.utils.parse_hdf5 import read_hdf5

data_dict = read_hdf5("/home/numbnut/repo/RobotTwin2/data/beat_block_hammer/demo_randomized/data/episode0.hdf5")

# Ensure the entire numpy array is printed
np.set_printoptions(threshold=np.inf)

with open("output.txt", "w") as f:
    f.write("joint_action[vector]:\n")
    f.write(np.array2string(data_dict["joint_action"]["vector"]))
    f.write("\n\n")

    f.write("joint_action[left_arm]:\n")
    f.write(np.array2string(data_dict["joint_action"]["left_arm"]))
    f.write("\n\n")

    f.write("joint_action[left_gripper]:\n")
    f.write(np.array2string(data_dict["joint_action"]["left_gripper"]))
    f.write("\n")

print("Data has been written to output.txt")