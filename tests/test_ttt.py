from tensornn import *
from tensornn.nn import *
from tensornn.activation import *
from tensornn.loss import *
from tensornn.layers import *
from tensornn.optimizers import *
from tensornn.debug import debug

import numpy as np
import pytest

debug.disable()

def game_state(board):
    # -1 = win for x
    # 0 = tie
    # 1 = win for o
    # 2 = not finished
    lines = [
        (0,1,2), (3,4,5), (6,7,8),  # rows
        (0,3,6), (1,4,7), (2,5,8),  # cols
        (0,4,8), (2,4,6),           # diags
    ]

    for a,b,c in lines:
        if board[a] == board[b] == board[c] != 0:
            return board[a]

    if all(board):
        return 0

    return 2

def best_move(board: np.ndarray):
    # -1 on board = x
    # 0 on board = empty
    # 1 on board = o

    if game_state(board) != 2:
        return None, game_state(board)

    # determine turn
    turn = np.count_nonzero(board == 0) & 1
    
    if turn:
        # x turn
        # minimize eval
        bmove = -1
        beval = float("inf")
        for i in np.where(board == 0)[0]:
            board[i] = -1
            eval = best_move(board)[1]
            board[i] = 0
            if eval < beval:
                beval = eval
                bmove = i
            if beval in (-1, 1):
                break
    else:
        # o turn
        # maximize eval
        bmove = -1
        beval = float("-inf")
        for i in np.where(board == 0)[0]:
            board[i] = 1
            eval = best_move(board)[1]
            board[i] = 0
            if eval > beval:
                beval = eval
                bmove = i
            if beval in (-1, 1):
                break
    return bmove, beval

def all_possible():
    last_batch = [np.zeros(9)]
    last_moves = [best_move(last_batch[0])[0]]
    move = -1
    all_batches = [last_batch]
    all_moves = [last_moves]
    seen = {last_batch[0].tobytes()}

    for movecnt in range(9):
        new_batch = []
        new_moves = []
        for board in last_batch:
            for i in np.where(board == 0)[0]:
                new_board = board.copy()
                new_board[i] = move
                if game_state(new_board) == 2 and new_board.tobytes() not in seen:
                    new_batch.append(new_board)
                    new_moves.append(best_move(new_board)[0])
                    seen.add(new_board.tobytes())

        all_batches.append(new_batch)
        all_moves.append(new_moves)
        last_batch = new_batch
        move *= -1

        if len(new_batch) == 0:
            break

    all_batches = [np.array(batch) for batch in all_batches[:-1]]
    all_moves = [np.fromiter(mv, dtype=np.int8) for mv in all_moves[:-1]]

    return all_batches, all_moves

def generate_data():
    all_batches, all_moves = all_possible()
    boards = np.concatenate(all_batches)
    moves = np.concatenate(all_moves)

    # Vectorized turn indicator assignment
    turn_indicators = np.zeros((boards.shape[0], 1))
    non_zero_counts = np.count_nonzero(boards, axis=1)
    zero_counts = 9 - non_zero_counts
    turn_indicators[:, 0] = np.where(zero_counts % 2 == 1, -1, 1)

    boards_with_turn = np.concatenate([boards, turn_indicators], axis=1)
    return Tensor(boards_with_turn), one_hot(moves, 9)

def create_nn():
    nn = NeuralNetwork([
        Input(10),
        Dense(64, activation=ReLU()),
        Dense(64, activation=ReLU()),
        Dense(9, activation=Softmax()),
    ])
    nn.register(CategoricalCrossEntropy(), Adam(0.01))
    return nn

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ttt(seed):
    set_seed(seed)

    inputs, outputs = generate_data()
    nn = create_nn()

    nn.train(inputs, outputs, epochs=1)

    pred_indices = nn.predict(inputs)
    expected_indices = np.argmax(outputs, axis=1)
    accuracy = np.mean(pred_indices == expected_indices)

    assert accuracy > 0.75, f"Accuracy {accuracy:.2%} is not greater than 75%"