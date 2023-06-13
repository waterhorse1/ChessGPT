"""chess_crawl dataset."""
'''
tfds build --imports chess_ai.datasets.tfds --overwrite pathtochessmastery --manual_dir /nvme2/clip_annotated_pgn/annotated_pgn --register_checksums '--beam_pipeline_options=runner=DirectRunner,direct_num_workers=8,direct_running_mode=multi_processing'
'''
import os

import chess
import chess.pgn
import tensorflow_datasets as tfds

from chess_ai.datasets.tfds import pgn_base


class Pathtochessmastery(pgn_base.PGNDatasetBuilder):
    """DatasetBuilder for chess_crawl dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self):
        return pgn_base.build_info(self, pgn_base.DatasetConfig(
            homepage="https://www.pathtochessmastery.com/"
        ))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""

        data_dir = dl_manager.manual_dir / "pathtochessmastery"

        return {
            "train": self._generate_examples(data_dir),
        }

    def _iter_games(self, filename):
        encoding = "utf-8"
        key_prefix = os.path.basename(filename)
        counter = 0
        with open(filename, mode="rt", encoding=encoding) as handle:
            game = chess.pgn.read_game(handle)
            while game is not None:
                yield f"{key_prefix}/{counter}", game
                game = chess.pgn.read_game(handle)
                counter += 1

    def _generate_examples(self, pgn_dir):
        """Yields examples."""
        return pgn_base.generate_beam_examples(pgn_dir, "utf-8")
