import numpy as np

# SCORING_PARAMETERS = dict(three_in_a_rows: +2,
                          # two_in_a_rows

def available_columns(board: np.ndarray):
    avail_cols = []
    for col in range(7):
        if board[0, col] == 0:
            avail_cols.append(col)
    return np.array(avail_cols)

def create_hypo_board(board: np.ndarray, col: int, piece: int):
    '''return copy of the board with a piece placed in the given column

    assumes the col is valid
    '''
    # find last row that is zero
    empty_row = np.where(board[:, col] == 0)[0][-1]

    hypo_board = np.copy(board)
    hypo_board[empty_row, col] = piece

    return hypo_board

def find_patterns(board: np.ndarray, player: int):
    def safe_check(board, i, j):
        if 0 <= i < board.shape[0] and 0 <= j < board.shape[1]:
            return board[i, j]
        else:
            return None

    # dont need all 8 directions, (never go straight up)
    dir_ = ((0,1), (1,1), (1,-1),(0,-1),(-1,-1),(-1,0),(-1,1))

    patterns = {2: 0, 3: 0, 4: 0}

    max_length = 4

    # start from any zero
    for x,y in np.argwhere(board == 0):
        # prevents double counting
        for dx, dy in dir_:
            tmpx = x
            tmpy = y
            length = 0
            while 1:
                tmpx += dx
                tmpy += dy
                val = safe_check(board, tmpx, tmpy)
                if val == player and length < max_length:
                    length += 1
                else:
                    if length > 1:
                        patterns[length] += 1
                    break

    return patterns

def score_board(board: np.ndarray, my_patterns: dict, opp_patterns: dict) -> float:
    '''
    '''
    my_score = my_patterns[2] * 1 + my_patterns[3] * 10 + my_patterns[4] * 100
    opp_score = opp_patterns[2] * 1 + opp_patterns[3] * 10 + opp_patterns[4] * 100
    score = my_score - opp_score
    return score

def minimax(board, depth, max_depth):
    # check for end state
    my_patterns = find_patterns(board, 1)
    opp_patterns = find_patterns(board, -1)
    score = score_board(board, my_patterns, opp_patterns)

    if my_patterns[4] > 0 or opp_patterns[4] > 0 or depth == max_depth:
        return score

    scores = []
    piece = 1 if depth % 2 == 0 else -1
    for col in available_columns(board):
        hypo_board = create_hypo_board(board, col, piece)
        this_score = minimax(hypo_board, depth+1, max_depth)
        scores.append( this_score )

    if depth % 2 == 0:
        return max(scores)
    else:
        return min(scores)

def determine_action(board: np.ndarray, search_depth: int):
    scores = []

    avail_cols = available_columns(board)
    for col in avail_cols:
        hypo_board = create_hypo_board(board, col, 1)
        scores.append( minimax(hypo_board, depth=1, max_depth=search_depth) )

    print(scores)
    best_idxs = np.where(scores == np.max(scores))[0]
    best_cols = avail_cols[best_idxs]

    # randomly break ties
    # prob = 1/np.minimum(np.abs(best_cols - 3), 0.3)
    # prob /= prob.sum()
    # best_col = np.random.choice(best_cols)

    # break ties based on closest to middle
    dist_from_center = np.abs(best_cols - 3)
    best_col = best_cols[ np.argsort( dist_from_center )[0] ]

    return best_col
