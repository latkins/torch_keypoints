import logging
import tempfile
import zipfile
from enum import Enum
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat

from . import download
from .enums import Split

logger = logging.getLogger(__name__)


class MPII:
    FOLDER_NAME = "mpii_human_pose_v1"
    IMG_URL = "https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz"
    ANNOT_URL = "https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1_u12_2.zip"

    def __init__(self, data_dir: Path = Path("/tmp"), split: Split = Split.TRAIN):

        assert isinstance(split, Split)
        if (
            not (data_dir / self.FOLDER_NAME / "mpii_human_pose_v1.tar.gz").exists()
            or not (data_dir / self.FOLDER_NAME / "mpii_human_pose_v1_u12_2.zip").exists()
        ):
            self._download(data_dir)
        pass

    def _download(self, data_dir: Path):
        logger.warning("This is a large (~13Gb) download")
        root = data_dir / self.FOLDER_NAME
        root.mkdirs(exists_ok=True)

        logger.info("Downloading annotations.")
        download.stream(self.ANNOT_URL, root)

        logger.info("Downloading images.")
        download.stream(self.IMAGE_URL, root)
