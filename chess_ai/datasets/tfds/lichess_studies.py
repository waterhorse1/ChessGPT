import logging
import os

import chess
import chess.pgn
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.core import beam_utils

from chess_ai import feature_converter
from chess_ai.datasets.tfds import pgn_base


# Forked from pgn base with some modification
# TODO(yl): merge this back
# Removed result
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


def generate_examples_from_game(game: chess.pgn.Game):
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
            comment = next_game.comment
            # Add last board to history
            features = feature_converter.stack_observations(history, history_size=8)
            features["probs"] = _get_probs(board, move)
            features["comment"] = comment
            yield features
            game = next_game


def _iter_games(filename, encoding="utf-8"):
    encoding = "utf-8"
    key_prefix = os.path.basename(filename)
    counter = 0
    with open(filename, mode="rt", encoding=encoding) as handle:
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


def generate_beam_examples(path, encoding="utf-8"):
    beam = tfds.core.lazy_imports.apache_beam

    filenames = list(path.glob("*.pgn"))
    filenames = [str(f) for f in filenames]

    def process_example(filename):
        for key, game in _iter_games(filename, encoding=encoding):
            try:
                if game.errors:
                    raise game.errors[0]
                for i, example in enumerate(generate_examples_from_game(game)):
                    yield f"{key}/{i}", example
                logging.debug("Finished processing %s", filename)
                beam_utils.inc_counter("parsing:success", 1)
            except Exception:  # pylint: disable=broad-except
                logging.exception("Unable to process %s", filename)
                beam_utils.inc_counter("parsing:failed", 1)

    return beam.Create(filenames) | "ParsePGN" >> beam.FlatMap(process_example)


class LichessStudies(pgn_base.PGNDatasetBuilder):
    """DatasetBuilder for chess_crawl dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    Preparation of the chess_crawl dataset requires manual download.
    Download the crawled dataset and place. Place the `game_pgn.zip`
    file in the `manual_dir/`. For more information, please refer to

    https://www.tensorflow.org/datasets/add_dataset#manual_download_and_extraction

    """

    def _info(self):
        return pgn_base.build_info(
            self,
            pgn_base.DatasetConfig(
                homepage="https://lichess.org/study"
            ),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""

        # TODO(yl): This requires the raw data archive to have a fixed name.
        # Consider configuring this via a `BuilderConfig`.
        data_dir = dl_manager.manual_dir / "lichess_studies"
        return {
            "train": self._generate_examples(data_dir),
        }
        
    def _generate_examples(self, pgn_dir):
        """Yields examples."""
        return pgn_base.generate_beam_examples(pgn_dir, encoding="utf-8")
