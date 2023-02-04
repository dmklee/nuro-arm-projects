import numpy as np

UNAVAIL = -4242

# normal
SCORING_PARAMS = dict(me={2: 1, 3: 10, 4: 100}, opp={2: 1, 3: 10, 4: 100})
# SCORING_PARAMS = dict( me= {2: 0, 3:0, 4:0}, opp={2: 30, 3:100, 4:100})
# def set_scoring_params(level):
    # if level == 'easy':
        # return dict( me={2: 30, 3: 50, 4: 100}, opp={2: 25, 3: 60, 4: 100})
    # elif level == 'hard':
        # return dict( me={2: 1, 3: 10, 4: 100}, opp={2: 1, 3: 10, 4: 100})
    # elif level == 'annoying':
        # return dict( me= {2: 0, 3:0, 4:0}, opp={2: 30, 3:100, 4:100})

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
    dir_ = ((0,1), (1,1), (1,-1), (0,-1), (-1,-1), (1,0), (-1,1))

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
    my_score = sum(SCORING_PARAMS['me'][i]*my_patterns[i] for i in (2,3,4))
    opp_score = sum(SCORING_PARAMS['opp'][i]*opp_patterns[i] for i in (2,3,4))
    score = my_score - opp_score
    return score

def minimax(board, depth, max_depth):
    # check for end state
    my_patterns = find_patterns(board, 1)
    opp_patterns = find_patterns(board, -1)
    score = score_board(board, my_patterns, opp_patterns)

    if depth == max_depth or my_patterns[4] > 0 or opp_patterns[4] > 0:
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
    scores = [UNAVAIL for _ in range(7)]

    avail_cols = available_columns(board)
    for col in avail_cols:
        hypo_board = create_hypo_board(board, col, 1)
        scores[col] = minimax(hypo_board, depth=1, max_depth=search_depth)

    best_cols = np.where(scores == np.max(scores))[0]

    # randomly break ties
    prob = 1/np.abs(best_cols - 2.9)**0.8
    prob /= prob.sum()
    best_col = np.random.choice(best_cols)

    # break ties based on closest to middle
    # dist_from_center = np.abs(best_cols - 3)
    # best_col = best_cols[ np.argsort( dist_from_center )[0] ]

    return best_col, dict(scores=scores, col_heights=(board!=0).sum(axis=0))
