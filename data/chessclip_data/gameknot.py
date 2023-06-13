"""chess_crawl dataset."""

import tensorflow_datasets as tfds

from chess_ai.datasets.tfds import pgn_base


class Gameknot(pgn_base.PGNDatasetBuilder):
    """DatasetBuilder for chess_crawl dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self):
        return pgn_base.build_info(
            self,
            pgn_base.DatasetConfig(
                homepage="https://www.angelfire.com/games3/smartbridge/"
            ),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""

        # TODO(yl): This requires the raw data archive to have a fixed name.
        # Consider configuring this via a `BuilderConfig`.
        data_dir = dl_manager.manual_dir / "gameknot_translated"

        return {
            "train": self._generate_examples(data_dir),
        }

    def _generate_examples(self, pgn_dir):
        """Yields examples."""
        # pylint: disable=import-outside-toplevel
        return pgn_base.generate_beam_examples(pgn_dir, "utf-8")
