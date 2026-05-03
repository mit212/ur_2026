import time
import numpy as np
import math
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive

# Parameters
rtde_frequency = 125.0
dt = 1 / rtde_frequency
ur_cap_port = 50002
robot_ip = "192.168.137.2" # make sure to update this to your robot's IP address
flags = RTDEControl.FLAG_VERBOSE | RTDEControl.FLAG_UPLOAD_SCRIPT

# ur_rtde realtime priorities
rt_receive_priority = 90
rt_control_priority = 85

rtde_r = RTDEReceive(robot_ip, rtde_frequency, [], True, False, rt_receive_priority)
rtde_c = RTDEControl(robot_ip, rtde_frequency, flags, ur_cap_port, rt_control_priority)

# Initialize task variables
task_frame = [0, 0, 0, 0, 0, 0]
selection_vector = [1, 0, 0, 0, 0, 0]
force_type = 2
limits = [5, 2, 1.5, 1, 1, 1]
joint_q = [-1.54, -1.83, -2.28, -0.59, 1.60, 0.023]
K = 100.0 # N/m

# Function to get desired end effector pose
def get_TCP_des(pose, timestep, radius=0.20, freq=0.5):
    p = pose[:]
    p[0] = pose[0] + radius * math.cos((2 * math.pi * freq * timestep))
    return p

def main():
    # Move to initial joint position with a regular moveJ
    rtde_c.moveJ(joint_q)
    vel = 0.1
    acc = 0.1

    # Controller loop
    timecounter = 0.0

    # Move to init position using moveL
    initial_tcp_pose = rtde_r.getActualTCPPose()

    while True:
        t_start = rtde_c.initPeriod()

        # Compute next desired TCP pose
        next_p  = get_TCP_des(initial_tcp_pose, timecounter)

        # Get current pose
        current_p = rtde_r.getActualTCPPose()

        # Calculate delta X and use to get "restorative" force
        dx = next_p[0] - current_p[0]
        Fx = K * dx

        # Construct wrench
        wrench = [Fx, 0, 0, 0, 0, 0]

        # Command wrench in force mode
        rtde_c.forceMode(task_frame, selection_vector, wrench, force_type, limits)

        # Increment time counter
        timecounter += dt

        # Pause for remainder of realtime period
        rtde_c.waitPeriod(t_start)
        

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user, stopping robot safely.")
        rtde_c.forceModeStop()
        rtde_c.stopScript()
        print("RTDE scripts finished")