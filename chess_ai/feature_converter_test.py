import io
import os

import chess
import chess.pgn
import numpy as np
import tensorflow as tf
from absl.testing import absltest

from chess_ai import feature_converter
from chess_ai.datasets.chess_crawl import chess_crawl

_TEST_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../tests/testdata")
)


def _assert_input_planes_equal(board, input_planes):
    assert input_planes.shape == (112, 8, 8)
    piece_order = [
        chess.PAWN,
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
        chess.KING,
    ]
    t = len(board.move_stack)
    for i, piece_type in enumerate(piece_order):
        expected_our_piece_plane = board.pieces(piece_type, color=board.turn)
        expected_their_piece_plane = board.pieces(piece_type, color=not board.turn)
        if board.turn == chess.BLACK:
            expected_our_piece_plane = expected_our_piece_plane.mirror()
            expected_their_piece_plane = expected_their_piece_plane.mirror()
        expected_our_piece_plane = np.array(
            expected_our_piece_plane.tolist(), dtype=np.float32
        ).reshape((8, 8))
        expected_their_piece_plane = np.array(
            expected_their_piece_plane.tolist(), dtype=np.float32
        ).reshape((8, 8))
        np.testing.assert_allclose(input_planes[i], expected_our_piece_plane)
        np.testing.assert_allclose(input_planes[i + 6], expected_their_piece_plane)
    # Skip history checking for now
    # Check padding history planes
    np.testing.assert_allclose(input_planes[13 * (t + 1) : 104, :, :], 0)
    # castling plane
    # our ooo (queenside)
    np.testing.assert_allclose(
        input_planes[104], board.has_queenside_castling_rights(board.turn)
    )
    # our oo (kingside)
    np.testing.assert_allclose(
        input_planes[105], board.has_kingside_castling_rights(board.turn)
    )
    # their ooo (queenside)
    np.testing.assert_allclose(
        input_planes[106], board.has_queenside_castling_rights(not board.turn)
    )
    # their oo (kingside)
    np.testing.assert_allclose(
        input_planes[107], board.has_kingside_castling_rights(not board.turn)
    )
    is_black = board.turn == chess.BLACK
    # color plane (0 for white and 1 for black)
    np.testing.assert_allclose(input_planes[108], float(is_black))
    # rule50 plane
    np.testing.assert_allclose(input_planes[109], board.halfmove_clock / 99.0)
    np.testing.assert_allclose(input_planes[110], 0.0)
    np.testing.assert_allclose(input_planes[111], 1.0)


class FeatureConverterTest(absltest.TestCase):
    def test_get_board_features(self):
        with open(os.path.join(_TEST_DATA_DIR, "game_44.pgn"), "rt") as f:
            pgn = f.read()
        game = chess.pgn.read_game(io.StringIO(pgn))
        board = game.board()
        moves = game.mainline_moves()
        states = []
        for t, move in enumerate(moves):
            states.append(feature_converter.get_board_features(board))
            # Verify that observations can be encoded correctly
            input_planes = feature_converter.get_lc0_input_planes(
                feature_converter.stack_observations(states[: t + 1], history_size=8)
            )
            _assert_input_planes_equal(board, input_planes)
            board.push(move)

    def test_get_lc0_input_planes_tfds(self):
        with open(os.path.join(_TEST_DATA_DIR, "game_44.pgn"), "rt") as f:
            pgn = f.read()
        examples = list(
            chess_crawl.generate_examples(chess.pgn.read_game(io.StringIO(pgn)))
        )
        dataset = tf.data.Dataset.from_tensor_slices(
            tf.nest.map_structure(
                lambda *x: tf.stack([tf.convert_to_tensor(xx) for xx in x]), *examples
            )
        )
        dataset = dataset.map(feature_converter.get_lc0_input_planes_tf)
        game = chess.pgn.read_game(io.StringIO(pgn))
        board = game.board()
        moves = game.mainline_moves()
        for move, input_planes in zip(moves, dataset.as_numpy_iterator()):
            # Verify that observations can be encoded correctly
            _assert_input_planes_equal(board, input_planes)
            board.push(move)


if __name__ == "__main__":
    absltest.main()
