import argparse
import time
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from nuro_arm import RobotArm

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from vision import get_board_state, interactive_aligner
from planning import determine_action
from actuation import pick_piece, place_piece, go_home


class Visualization:
    def __init__(self,
                 figsize=(15,8),
                ):
        self.fig = plt.figure(figsize=figsize)

        gs = GridSpec(3, 5, self.fig, wspace=0.1, hspace=0.1, left=0.1, right=0.9)

        self.crop_ax = self.fig.add_subplot(gs[:2,:2])
        self.crop_ax.axis('off')
        self.red_ax = self.fig.add_subplot(gs[2,0])
        self.red_ax.axis('off')
        self.yellow_ax = self.fig.add_subplot(gs[2,1])
        self.yellow_ax.axis('off')
        self.board_ax = self.fig.add_subplot(gs[:,2:])
        self.board_ax.set_xticklabels([])
        self.board_ax.set_yticklabels([])

        piece_size = 60
        self.sel_col, = self.board_ax.plot([-1,-1],[-1,10], 'b-', linewidth=25, alpha=0.3)
        self.red_pieces, = self.board_ax.plot([],[], 'o', mfc='r', markersize=piece_size,
                                              mew=3, mec='k')
        self.yellow_pieces, = self.board_ax.plot([],[], 'o', mfc='y', markersize=piece_size,
                                                 mew=3, mec='k')
        self.empty_pieces, = self.board_ax.plot([],[], 'o', mfc=(0.3,0.3,0.3),
                                                markersize=piece_size, mew=3,
                                                mec='k', alpha=0.5)

        self.col_texts = [self.board_ax.text(0,0,'', fontsize=14, fontweight='bold', ha='center',va='center') for _ in range(7)]

        self.board_ax.set_xlim(-0.5, 6.5)
        self.board_ax.set_ylim(-0.5, 5.5)

        self.im_objs = dict()

    def update_vision_info(self,
                           board_state,
                           cropped_img,
                           red_mask,
                           yellow_mask,
                           grids,
                          ):
        if 'cropped' not in self.im_objs:
            self.im_objs['cropped'] = self.crop_ax.imshow(cropped_img)
            self.im_objs['grids'] = self.crop_ax.scatter(grids[0].flatten(),
                                                         grids[1].flatten(),
                                                         s=80, color=(0,0,0,0),
                                                         edgecolors='g', marker='s')
            red_img = np.repeat(red_mask[...,None],3,2)
            red_img[...,1:] = 0
            self.im_objs['red'] = self.red_ax.imshow(red_img)
            yellow_img = np.repeat(yellow_mask[...,None],3,2)
            yellow_img[...,2] = 0
            self.im_objs['yellow'] = self.yellow_ax.imshow(yellow_img)
        else:
            self.im_objs['cropped'].set_data(cropped_img)
            red_img = np.repeat(red_mask[...,None],3,2)
            red_img[...,1:] = 0
            self.im_objs['red'].set_data(red_img)
            yellow_img = np.repeat(yellow_mask[...,None],3,2)
            yellow_img[...,2] = 0
            self.im_objs['yellow'].set_data(yellow_img)

        self.empty_pieces.set_data(*np.where(board_state[::-1]==0)[::-1])
        self.yellow_pieces.set_data(*np.where(board_state[::-1]==1)[::-1])
        self.red_pieces.set_data(*np.where(board_state[::-1]==-1)[::-1])

        self.fig.canvas.draw()
        plt.show(block=False)

    def update_planning_info(self, best_col, scores, col_heights):
        for col_id in range(7):
            if col_heights[col_id] < 6:
                new_text = str(scores[col_id])
                self.col_texts[col_id].set_text(f'{scores[col_id]:+}')
                self.col_texts[col_id].set_x(col_id)
                self.col_texts[col_id].set_y(col_heights[col_id])
            else:
                self.col_texts[col_id].set_text('')
        self.sel_col.set_xdata([best_col, best_col])

        self.fig.canvas.draw()
        plt.show(block=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera_id', type=int, default=2)
    parser.add_argument('--recalibrate_camera', action='store_true')
    parser.add_argument('--search_depth', type=int, default=4)
    args = parser.parse_args()

    robot = RobotArm()
    go_home(robot)
    robot.open_gripper()

    cap = cv2.VideoCapture(args.camera_id)
    if args.recalibrate_camera:
        np.save('crop_corners.npy', interactive_aligner(cap))

    vis = Visualization()
    board, vision_info = get_board_state(cap)
    vis.update_vision_info(board, **vision_info)
    while 1:
        # wait for enter command to allow human to move
        input('Hit enter when robot should play... ')

        board, vision_info = get_board_state(cap)
        vis.update_vision_info(board, **vision_info)

        # pick up piece while we plan the next move to save time
        with ThreadPoolExecutor() as exe:
            task = exe.submit(lambda : pick_piece(robot))
            best_col, planning_info = determine_action(board, 4)
            task.result()

        vis.update_planning_info(best_col, **planning_info)

        place_piece(robot, best_col)
