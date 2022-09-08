import numpy as np
import cv2
import matplotlib.pyplot as plt

RED_HUE_RANGE = (170, 190)
YELLOW_HUE_RANGE = (20, 35)

# (min_x, max_x), (min_y, max_y)
CROP_EXTENTS = ((45, 309), (166, 504))
PATCH_SIZE = 8

def capture_image(cap, n_discarded: int=4) -> np.ndarray:
    count = 0
    while True:
        ret, img = cap.read()

        if ret:
            count += 1
        if count > n_discarded:
            break

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def show_image(img):
    plt.figure()
    plt.imshow(img)
    plt.show()

def crop_image_to_board(img: np.ndarray):
    (a,b), (c,d) = CROP_EXTENTS
    crop = img[a:b, c:d]
    return crop

def get_red_yellow_masks(img: np.ndarray):
    ''' assumes rgb array '''
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue_img = hsv_img[:, :, 0]

    # to determine ranges, it is helpful to plot the hue img
    red_mask = cv2.inRange(hue_img, *RED_HUE_RANGE)
    yellow_mask = cv2.inRange(hue_img, *YELLOW_HUE_RANGE)
    return red_mask, yellow_mask

def read_board_from_masks(red_mask: np.ndarray, yellow_mask: np.ndarray):
    board = np.zeros((6, 7), dtype=int)

    height, width = red_mask.shape
    grid_height = height / 6
    grid_width = width / 7

    grid_xs = (width / 7 * (np.arange(7) + 0.5)).astype(int)
    grid_ys = (height / 6 * (np.arange(6) + 0.5)).astype(int)

    ps = PATCH_SIZE
    for i, y in enumerate(grid_ys):
        for j, x in enumerate(grid_xs):
            red_patch = red_mask[y-ps:y+ps, x-ps:x+ps]
            yellow_patch = yellow_mask[y-ps:y+ps, x-ps:x+ps]

            is_red = (red_patch == 255).mean() > 0.3
            is_yellow = (yellow_patch == 255).mean() > 0.3

            assert not (is_yellow and is_red), 'Detected red and yellow at some spot'

            if is_red:
                board[i, j] = -1
            if is_yellow:
                board[i, j] = 1

    return board

def interactive_aligner(cap):
    (a, b), (c, d) = CROP_EXTENTS
    while True:
        ret, img = cap.read()

        if ret:
            cv2.rectangle(img, (c,a), (d,b), (0,0,255), 1)
            cv2.imshow('aligner', img)
            cv2.waitKey(50)

def get_board_state(cap):
    img = capture_image(cap)
    cropped_img = crop_image_to_board(img)
    red_mask, yellow_mask = get_red_yellow_masks(cropped_img)
    board = read_board_from_masks(red_mask, yellow_mask)
    return board

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    board = get_board_state(cap)
    print(board)
