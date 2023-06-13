import bz2
import gzip
from typing import Optional, TextIO

import chess
import zstandard
from etils import epath


class PGNReader:
    class _BufferedTextReader:
        """Adaptor for tracking lines read from a file."""

        def __init__(self, handle) -> None:
            self._handle = handle
            self._accumulator = []

        def readline(self) -> str:
            """Read and accumulate a line from the file."""
            line = self._handle.readline()
            self._accumulator.append(line)
            return line

        def read_result(self) -> str:
            result = "".join(self._accumulator)
            self._reset()
            return result

        def _reset(self) -> None:
            del self._accumulator[:]

    def __init__(self, handle: TextIO):
        self._reader = self._BufferedTextReader(handle)

    def read_game(self) -> Optional[str]:
        """Read a single PGN record from file."""
        # Use skip_game to real from the underlying file
        # and accumulates the result
        # NOTE: We reply on the fact that chess.pgn only
        # uses `readline` from the file handle, which we intercepts
        # and accumulates the read values. This allows us to
        # reuse the parsing done in chess.pgn which is likely
        # more robust than a custom implementation.
        game_found_and_skipped = chess.pgn.skip_game(self._reader)
        # Get a single PGN record as str
        result = self._reader.read_result()
        # Reset the accumulator
        if game_found_and_skipped:
            return result
        else:
            return None


def _open_file(filename: str, mode=None, encoding=None):
    if filename.endswith(".zst"):
        return zstandard.open(filename, mode=mode, encoding=encoding)
    elif filename.endswith(".bz2"):
        return bz2.open(filename, mode=mode, encoding=encoding)
    elif filename.endswith(".gz"):
        return gzip.open(filename, mode=mode, encoding=encoding)
    else:
        return epath.Path(filename).open(encoding=encoding)


def split_pgns_from_file(filename: str):
    with _open_file(filename, mode="rt", encoding="utf-8") as handle:
        reader = PGNReader(handle)
        game = reader.read_game()
        while game is not None:
            yield filename, game
            game = reader.read_game()
