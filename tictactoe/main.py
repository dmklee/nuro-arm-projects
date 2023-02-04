import time
import numpy as np
import cv2


# End effector paths to draw x or o
X_PATH = [
    np.zeros((10,3)),
    np.zeros((10,3)),
]
O_PATH = [
    np.zeros(),
]

GRID_SIZE = 0.1 # in meters
CENTER_GRID_LOC = np.zero(3)


def execute_path(robot, path):
    arm_jposs = []
    for waypt in path:
        arm_jpos, _ = robot.mp.calculate_ik(waypt)
        arm_jposs.append(arm_jpos)

    robot.move_arm_jpos(waypt[0])
    arm_joint_ids = robot.controller.arm_joint_ids
    for arm_jpos in arm_jposs:
        robot.controller.move_servos(arm_joint_ids,
                                     arm_jpos,
                                     duration=15)
        time.sleep(0.015)


def draw_char(row, col, char):
    '''Draw X or O at a given location, such that it fits within the grid
    '''
    x = CENTER_GRID_LOC[0] + GRID_SIZE * (col-1)
    y = CENTER_GRID_LOC[1] + GRID_SIZE * (row-1)
    z = CENTER_GRID_LOC[2]

    path = X_PATH if char == 'x' else O_PATH
    for subpath in path:
        execute_path(robot, subpath)
    pass

def play(row, col, char='x'):
    '''Perform drawing action at given row and column
    '''
    # go to premove location

    # perform 
    pass

def 



