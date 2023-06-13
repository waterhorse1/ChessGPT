# BROKEN at the moment, do not use.
"""pgnlib dataset."""

import io
import logging
import zipfile

import chardet
import chess
import chess.pgn
import tensorflow_datasets as tfds
from tensorflow_datasets.core import beam_utils

from chess_ai.datasets.tfds import pgn_base


class Pgnlib(pgn_base.PGNDatasetBuilder):
    """DatasetBuilder for chess_crawl dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self):
        return pgn_base.build_info(
            self,
            pgn_base.DatasetConfig(
                homepage="https://www.chesspublishing.com/subscribe.html"
            ),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""

        data_dir = dl_manager.manual_dir / "pgnlib"
        return {
            "train": self._generate_examples(data_dir),
        }
        
    def _generate_examples(self, pgn_dir):
        """Yields examples."""
        return pgn_base.generate_beam_examples(pgn_dir, encoding="utf-8", filter_en=True)