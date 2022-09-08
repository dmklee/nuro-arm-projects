import time
import cv2
from nuro_arm import RobotArm

import vision
import planning
import actuation


robot = RobotArm()

cap = cv2.VideoCapture(0)
# vision.interactive_aligner(cap)

while 1:
    board = vision.get_board_state(cap)
    best_col = planning.determine_action(board, 4)
    print(board)
    print('best col', best_col)

    # wait for enter command to allow human to move
    input('')
