import time
import numpy as np
from nuro_arm import RobotArm

PICK_JPOS = np.array((1.210560369183267, -0.012566370614359171, 1.650383340685838, 1.1979939985689076, 0.0))
FIRST_COLUMN_LOC = np.array((0.163, 0.109, 0.345))
COLUMN_VEC = np.array((0, -0.0325, 0))
HOME_JPOS = np.array((-0.0419, -1.340, 1.081, 1.633, 0.0))
SPEED=0.8

def passive_pos_reading(robot):
    robot.passive_mode()
    while 1:
        print(robot.get_hand_pose()[0])
        time.sleep(0.5)

def passive_jpos_reading(robot):
    robot.passive_mode()
    while 1:
        print(robot.get_arm_jpos())
        time.sleep(0.5)

def pick_piece(robot: RobotArm):
    robot.move_arm_jpos(PICK_JPOS, speed=SPEED)

    while 1:
        if robot.set_gripper_state(robot.GRIPPER_CLOSED, speed=SPEED) > 0.2:
            break
        robot.set_gripper_state(robot.GRIPPER_OPENED, speed=SPEED)
        time.sleep(0.5)

    go_home(robot)

def place_piece(robot: RobotArm, col: int):
    target_pos = FIRST_COLUMN_LOC + col * COLUMN_VEC
    robot.move_hand_to(target_pos, speed=SPEED)
    robot.open_gripper()

    # back away from board to prevent collision
    current_jpos = robot.get_arm_jpos()
    current_jpos[1] -= 0.4
    current_jpos[2] += 0.2
    robot.move_arm_jpos(current_jpos, speed=SPEED)
    go_home(robot)

def go_home(robot: RobotArm):
    robot.move_arm_jpos(HOME_JPOS, speed=SPEED)

if __name__ == "__main__":
    robot = RobotArm()
    # passive_jpos_reading(robot)
    for i in [6]:
        pick_piece(robot)
        place_piece(robot, i)


