import json
import sys
sys.path.append('../../')
sys.path.append('../../open_clip/src')

from chess_ai.feature_converter import get_lc0_input_planes_tf
import chess.pgn
import io
import torch

import numpy as np

from chess_ai.datasets.tfds.pgn_base import generate_examples_from_game_no_comment

def generate_representation_for_final(pgn):
    game = chess.pgn.read_game(io.StringIO(pgn))
    data = list(generate_examples_from_game_no_comment(game))[-1]
    for key in data.keys():
        data[key] = np.array(data[key])
    board = get_lc0_input_planes_tf(data).numpy()
    action = data['probs']
    return board, action

from open_clip.factory import get_tokenizer, load_checkpoint
import open_clip
device = "cpu"
model_name = 'chessclip-quickgelu'
model = open_clip.create_model(model_name, pretrained='openai')
model.to(device)
tokenizer = get_tokenizer(model_name)

print(load_checkpoint(model, './ChessCLIP/epoch_latest.pt'))
import math

def mean(arr):
    return sum(arr) / len(arr)

def sample_stddev(arr):
    mu = mean(arr)
    return math.sqrt(sum([(x - mu) ** 2 for x in arr]) / (len(arr) - 1))

def mean_stderr(arr):
    return sample_stddev(arr) / math.sqrt(len(arr))

# evaluation code for clip on comment evaluation

# for comment multi-choice
def generate_similarity(pgn_choice, text_choice):
    board_list = []
    action_list = []
    for pgn in pgn_choice:
        board, action = generate_representation_for_final(pgn)
        board_list.append(board)
        action_list.append(action)
    text_tokens = tokenizer(text_choice)
    image_input = torch.from_numpy(np.stack(board_list, axis=0))
    action_input = torch.from_numpy(np.stack(action_list, axis=0))
    with torch.no_grad():
        image_features = model.encode_image((image_input, action_input)).float()
        text_features = model.encode_text(text_tokens).float()
    image_features /= image_features.norm(dim=-1, keepdim=True) # n * dim
    text_features /= text_features.norm(dim=-1, keepdim=True)# m * dim
    similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T # m * n
    return similarity

def evaluate_comment_match(data):
    examples = data['examples']
    score_list = []
    for idx, e in enumerate(examples):
        if idx % 100 ==0:
            print(idx)
        pgn = e['input']
        text_choice = [key for key in e['target_scores'].keys()]
        similarities = generate_similarity([pgn], text_choice) # m * 1
        score = e['target_scores'][text_choice[similarities.argmax(axis=0)[0]]]
        score_list.append(score)
    value = mean(score_list)
    stderr_value = mean_stderr(score_list)
    print(f"{value}, {stderr_value}")

def evaluate_pgn_match(data):
    examples = data['examples']
    score_list = []
    for idx, e in enumerate(examples):
        if idx % 100 ==0:
            print(idx)
        text = e['input']
        pgn_choice = [key for key in e['target_scores'].keys()]
        similarities = generate_similarity(pgn_choice, [text]) # 1 * n
        score = e['target_scores'][pgn_choice[similarities.argmax(axis=1)[0]]]
        score_list.append(score)
    value = mean(score_list)
    stderr_value = mean_stderr(score_list)
    print(f"{value}, {stderr_value}")

data = json.load(open('../eval_task/chess_opening_generation.json'))
# ../eval_task/chess_opening_generation.json'
# ../eval_task/chess_state_value_multi_choice_2_nob.json'
# ../eval_task/chess_annotation_nob.json'
# ../eval_task/chess_opening_mc.json'
evaluate_pgn_match(data)