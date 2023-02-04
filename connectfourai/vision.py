import numpy as np
import cv2
import matplotlib.pyplot as plt

RED_HUE_RANGE = ((170, 190), (0, 10))
YELLOW_HUE_RANGE = (20, 35)

CROP_CORNERS = np.load('crop_corners.npy')

PATCH_SIZE = 20

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

def extract_crop_using_corners(img: np.ndarray, corners):
    # corners must be in order (clockwise from top left)
    src = np.array(corners).astype(np.float32)
    dest = np.array([(0,0), (img.shape[1], 0),
                     (img.shape[1], img.shape[0]), (0, img.shape[0])]
                   ).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dest)
    return cv2.warpPerspective(img, M, (img.shape[1],img.shape[0]))


def get_red_yellow_masks(img: np.ndarray):
    ''' assumes rgb array '''
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue_img = hsv_img[:, :, 0]
    # show_image(hsv_img)

    # to determine ranges, it is helpful to plot the hue img
    red_mask = np.maximum(cv2.inRange(hue_img, *RED_HUE_RANGE[0]),
                          cv2.inRange(hue_img, *RED_HUE_RANGE[1]))
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

    return board, np.meshgrid(grid_xs, grid_ys)

alignment_corners = []
def interactive_aligner(cap):
    # https://www.includehelp.com/python/capturing-mouse-click-events-with-python-and-opencv.aspx
    window = 'Click to indicate corners (clockwise from top-left). Hit enter when done'

    cv2.namedWindow(window)

    def capture_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(alignment_corners) != 4:
                alignment_corners.append((x,y))
            else:
                print('You can only select four corners, middle click to reset')
        if event == cv2.EVENT_MBUTTONDOWN:
            while alignment_corners:
                alignment_corners.pop()

    cv2.setMouseCallback(window, capture_event)
    while True:
        ret, img = cap.read()

        if ret:
            for x,y in alignment_corners:
                cv2.circle(img, (x,y), 3, (0,0,255), -1)
            cv2.imshow(window, img)
            if cv2.waitKey(1) == 13:
                cv2.destroyAllWindows()
                break
    CROP_CORNERS[:] = np.array(alignment_corners)
    return CROP_CORNERS

def get_board_state(cap):
    img = capture_image(cap)
    cropped_img = extract_crop_using_corners(img, CROP_CORNERS)
    red_mask, yellow_mask = get_red_yellow_masks(cropped_img)
    board, grids = read_board_from_masks(red_mask, yellow_mask)
    return board, dict(cropped_img=cropped_img, red_mask=red_mask, yellow_mask=yellow_mask, grids=grids)

if __name__ == "__main__":
    '''
    (360, 640, 3)
    [(3, 4), (635, 6), (633, 355), (9, 355)]

    '''
    cap = cv2.VideoCapture(0)
    # img = capture_image(cap)
    # print(img.shape)
    # corners = interactive_aligner(cap)
    # print(corners)
    # exit()
    # corners = [(205, 23), (532, 77), (514, 319), (192, 350)]
    img = capture_image(cap)
    new_img = extract_crop_using_corners(img, CROP_CORNERS)
    show_image(new_img)
    board = get_board_state(cap)[0]
    print(board)
