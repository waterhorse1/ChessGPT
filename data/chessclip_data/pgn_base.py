"""chess dataset builder."""

import dataclasses
import logging
import os
import re

import chess
import chess.pgn
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.core import beam_utils

from chess_ai import feature_converter

_ANNOTATION_PATTERN = re.compile(
    r"""
    (?P<prefix>\s?)
    \[%(?:evp|cal|eval|clk|csl|emt|tqu)\s.*\]
    (?P<suffix>\s?)
    """,
    re.VERBOSE,
)

class LanguageIdentification:
    def __init__(self, model_path):
        pretrained_lang_model = model_path
        import fasttext

        self.model = fasttext.load_model(pretrained_lang_model)

    def predict_lang(self, text, k=3):
        text = text.replace("\n", " ")
        predictions = self.model.predict(
            text, k=k
        )  # returns top-k matching languages
        output = []
        for lang, _ in zip(*predictions):
            output.append(lang.replace("__label__", ""))
        return set(output)

try:
    lang_detector = LanguageIdentification('/nvme2/filter_models/lid.176.bin')
except:
    print('error loading language detector, skip')

def is_english(comment, k=2):
    def remove_all_pgn_moves(pgn_string):
        pattern = r"([PNBRQK]?[a-h]?[1-8]?x?[a-h][1-8](?:=[NBRQ])?[+#]?|[O-]{3,5})"
        return re.sub(pattern, '', pgn_string)
    lang_ids = lang_detector.predict_lang(remove_all_pgn_moves(comment), k)
    if "en" not in lang_ids:
        return False
    else:
        return True

def _mirror_move(move: chess.Move) -> chess.Move:
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        move.promotion,
    )


def _get_probs(board: chess.Board, move: str):
    if board.turn == chess.WHITE:
        move = board.parse_uci(move)
        probs = [-1] * len(feature_converter.LC0_POLICY_INDEX)
        for legal_move in board.legal_moves:
            probs[feature_converter.as_lc0_policy_index(legal_move.uci())] = 0
        probs[feature_converter.as_lc0_policy_index(move.uci())] = 1
    else:
        move = _mirror_move(board.parse_uci(move))
        board = board.copy(stack=False)
        board.apply_mirror()
        probs = [-1] * len(feature_converter.LC0_POLICY_INDEX)
        for legal_move in board.legal_moves:
            probs[feature_converter.as_lc0_policy_index(legal_move.uci())] = 0
        probs[feature_converter.as_lc0_policy_index(move.uci())] = 1
    return np.asarray(probs, np.float32)


def generate_examples_from_game(game: chess.pgn.Game, filter_en=False):
    game_result = game.headers["Result"].strip()
    if game_result in ("*", "1/2-1/2", "1/2"):
        winner = None
    elif game_result == "1-0":
        winner = chess.WHITE
    elif game_result == "0-1":
        winner = chess.BLACK
    else:
        winner = None#raise ValueError(f"Unknown result {game_result}")

    history = []
    while True:
        board = game.board()
        # Add last board to history
        history.append(feature_converter.get_board_features(board))
        # Advance to the next game state
        next_game = game.next()
        if next_game is None:
            break
        else:
            # game.move is the move that leads to the game state
            move = next_game.move.uci()
            comment = next_game.comment.strip()
            # Filter annotation pattern
            comment = _ANNOTATION_PATTERN.sub("", comment)
            if filter_en:
                if not comment or not is_english(comment, k=2):
                    game = next_game
                    continue
            else:
                if not comment:
                    game = next_game
                    continue
            # Add last board to history
            features = feature_converter.stack_observations(history, history_size=8)
            features["probs"] = _get_probs(board, move)
            features["comment"] = comment
            if winner is None:
                result = 0
            elif board.turn == chess.WHITE:
                result = 1 if winner == chess.WHITE else -1
            else:
                result = 1 if winner == chess.BLACK else -1
            features["result"] = result
            yield features
            game = next_game

def generate_examples_from_game_no_comment(game: chess.pgn.Game):
    game_result = game.headers["Result"].strip()
    if game_result in ("*", "1/2-1/2", "1/2"):
        winner = None
    elif game_result == "1-0":
        winner = chess.WHITE
    elif game_result == "0-1":
        winner = chess.BLACK
    else:
        winner = None#raise ValueError(f"Unknown result {game_result}")

    history = []
    while True:
        board = game.board()
        # Add last board to history
        history.append(feature_converter.get_board_features(board))
        # Advance to the next game state
        next_game = game.next()
        if next_game is None:
            break
        else:
            # game.move is the move that leads to the game state
            move = next_game.move.uci()
            comment = next_game.comment.strip()
            # Add last board to history
            features = feature_converter.stack_observations(history, history_size=8)
            features["probs"] = _get_probs(board, move)
            features["comment"] = comment
            if winner is None:
                result = 0
            elif board.turn == chess.WHITE:
                result = 1 if winner == chess.WHITE else -1
            else:
                result = 1 if winner == chess.BLACK else -1
            features["result"] = result
            yield features
            game = next_game


def _iter_games(filename, encoding="utf-8"):
    key_prefix = os.path.basename(filename)
    counter = 0
    with open(filename, mode="rt", encoding=encoding, errors='ignore') as handle:
        game = chess.pgn.read_game(handle)
        yield f"{key_prefix}/{counter}", game
        while game is not None:
            counter += 1
            try:
                game = chess.pgn.read_game(handle)
                if game.errors:
                    raise game.errors[0]
            except Exception:
                beam_utils.inc_counter("read:failed", 1)
                continue
            else:
                yield f"{key_prefix}/{counter}", game


def generate_beam_examples(paths, encoding="utf-8", filter_en=False):
    beam = tfds.core.lazy_imports.apache_beam

    if not isinstance(paths, (tuple, list)):
        paths = (paths,)

    filenames = []
    for path in paths:
        filenames.extend(list(path.glob("*.pgn")))

    filenames = [str(f) for f in filenames]

    def process_example(filename):
        for key, game in _iter_games(filename, encoding=encoding):
            try:
                if game.errors:
                    raise game.errors[0]
                for i, example in enumerate(generate_examples_from_game(game, filter_en=filter_en)):
                    yield f"{key}/{i}", example
                logging.debug("Finished processing %s", filename)
                beam_utils.inc_counter("parsing:success", 1)
            except Exception:  # pylint: disable=broad-except
                logging.exception("Unable to process %s", filename)
                beam_utils.inc_counter("parsing:failed", 1)

    return beam.Create(filenames) | "ParsePGN" >> beam.FlatMap(process_example)


@dataclasses.dataclass
class DatasetConfig:
    description: str = ""
    citation: str = ""
    homepage: str = ""


def build_info(builder, config: DatasetConfig):
    features = tfds.features.FeaturesDict(
        {
            # TFDS does not support uint64 in the serialized format so we have to
            # use uint8
            "planes": tfds.features.Tensor(
                shape=(13 * 8 * 8,),
                dtype=tf.uint8,
                encoding=tfds.features.Encoding.ZLIB,
            ),
            "probs": tfds.features.Tensor(
                shape=(len(feature_converter.LC0_POLICY_INDEX),),
                dtype=tf.float32,
                encoding=tfds.features.Encoding.ZLIB,
            ),
            "us_black": tfds.features.Tensor(shape=(), dtype=tf.bool),
            "move_count": tfds.features.Tensor(shape=(), dtype=tf.uint32),
            "rule50_counter": tfds.features.Tensor(shape=(), dtype=tf.uint32),
            "castling_us_ooo": tfds.features.Tensor(shape=(), dtype=tf.bool),
            "castling_us_oo": tfds.features.Tensor(shape=(), dtype=tf.bool),
            "castling_them_ooo": tfds.features.Tensor(shape=(), dtype=tf.bool),
            "castling_them_oo": tfds.features.Tensor(shape=(), dtype=tf.bool),
            "comment": tfds.features.Tensor(shape=(), dtype=tf.string),
            "result": tfds.features.Tensor(shape=(), dtype=tf.int32),
        }
    )

    return tfds.core.DatasetInfo(
        builder=builder,
        description=config.description,
        features=features,
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        # supervised_keys=("image", "label"),  # Set to `None` to disable
        supervised_keys=None,
        homepage=config.homepage,
        citation=config.citation,
    )


class PGNDatasetBuilder(tfds.core.GeneratorBasedBuilder, skip_registration=True):
    """DatasetBuilder for chess_crawl dataset."""

    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    Preparation of the chess_crawl dataset requires manual download.
    Download the crawled dataset and place. Place the `game_pgn.zip`
    file in the `manual_dir/`. For more information, please refer to

    https://www.tensorflow.org/datasets/add_dataset#manual_download_and_extraction

    """

    def _process_pipeline_result(self, pipeline_result):
        # TODO(yl): Maybe store the metrics somewhere
        logging.info("Metrics %r", pipeline_result.metrics().query())
