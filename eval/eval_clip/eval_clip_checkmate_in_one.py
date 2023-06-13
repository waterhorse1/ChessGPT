import json
import sys
sys.path.append('../../')
sys.path.append('../../chessclip/src')

from chess_ai.feature_converter import get_lc0_input_planes_tf
import chess.pgn
import io
import torch

import numpy as np

from chess_ai.datasets.tfds.pgn_base import generate_examples_from_game_no_comment
from open_clip.factory import get_tokenizer, load_checkpoint
import open_clip

device = "cpu"
model_name = 'chessclip-quickgelu'
model = open_clip.create_model(model_name, pretrained='openai')
model.to(device)
tokenizer = get_tokenizer(model_name)

print(load_checkpoint(model, './ChessCLIP/epoch_latest.pt'))
def generate_representation_for_final(pgn):
    game = chess.pgn.read_game(io.StringIO(pgn))
    data = list(generate_examples_from_game_no_comment(game))[-1]
    for key in data.keys():
        data[key] = np.array(data[key])
    board = get_lc0_input_planes_tf(data).numpy()
    action = data['probs']
    return board, action

import math

def mean(arr):
    return sum(arr) / len(arr)

def sample_stddev(arr):

    mu = mean(arr)
    return math.sqrt(sum([(x - mu) ** 2 for x in arr]) / (len(arr) - 1))


def mean_stderr(arr):
    return sample_stddev(arr) / math.sqrt(len(arr))

def get_fen_from_pgn(pgn):
    game = chess.pgn.read_game(io.StringIO(pgn))
    board = game.board()
    for move in game.mainline_moves():
        board.push(move)
    return board, board.fen(), [board.san(legal_move) for legal_move in board.legal_moves]


def evaluate_action_one(pgn, text, device='cpu'):
    board, fen, legal_moves = get_fen_from_pgn(pgn)
    pgn_sufix_list = []
    action_input = []
    for move in legal_moves:
        pgn_sufix = pgn + f" {move}"
        board_features, action_features = generate_representation_for_final(pgn_sufix)
        action_input.append(action_features)

    board_features = torch.from_numpy(board_features)
    action_input = torch.from_numpy(np.stack(action_input, axis=0)).to(device)
    image_input = board_features[None].repeat(action_input.shape[0], 1, 1, 1).to(device)
    text_tokens = tokenizer(text).to(device)
    with torch.no_grad():
        image_features = model.encode_image((image_input, action_input)).float()
        text_features = model.encode_text(text_tokens).float()
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
    choice = similarity[0].argmax(axis=0)
    return legal_moves[choice]

def evaluate_control_action(data, device="cpu"):
    examples = data['examples']
    score_list = []
    for idx, e in enumerate(examples):
        if idx % 10 ==0:
            print(idx)
        pgn = e['input']
        if pgn.endswith('.'):
            text = ['White successfully checkmate in one move']
        else:
            text = ['Black successfully checkmate in one move']
        choice_text = evaluate_action_one(pgn, text, device=device)
        score = e['target_scores'][choice_text]
        score_list.append(score)
    value = mean(score_list)
    stderr_value = mean_stderr(score_list)
    print(f"{value}, {stderr_value}")

data_checkmate_in_one = json.load(open('../eval_task/checkmate_in_one.json'))
evaluate_control_action(data_checkmate_in_one, device=device)