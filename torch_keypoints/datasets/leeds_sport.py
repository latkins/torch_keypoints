import logging
import tempfile
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat

from . import download
from .enums import Split

logger = logging.getLogger(__name__)


class LeedsSportBase:
    FOLDER_NAME = None
    DATA_URL = None

    def __init__(self, data_dir: Path = Path("/tmp/"), split: Split = Split.TRAIN, transforms=None):
        """
        Loads dataset if it is preseint in `data_dir`.
        Downloads and loads if not.

        :param data_dir: The directory in which to put data.
        """
        assert isinstance(split, Split)

        if not (data_dir / self.FOLDER_NAME).exists():
            self._download(data_dir)

        self.root = data_dir / self.FOLDER_NAME

        joints = loadmat(self.root / "joints.mat")["joints"]
        joints = np.moveaxis(joints, -1, 0)
        self.joints = np.moveaxis(joints, 1, 2)

        self.image_paths = list(
            sorted((self.root / "images").glob("*.jpg"), key=lambda p: int(p.stem[2:]))
        )

        self.transforms = transforms

    def _download(self, data_dir: Path):
        with tempfile.NamedTemporaryFile() as temp:
            download.stream(self.DATA_URL, temp)
            with zipfile.ZipFile(temp) as temp_zipped:
                temp_zipped.extractall(data_dir / self.FOLDER_NAME)

    def __getitem__(self, key: int):
        with self.image_paths[key].open("rb") as f:
            img = Image.open(f).convert("RGB")

        # This dataset only has a single person per image, but others may have more
        # Therefore, wrap keypoints in list.
        targets = OrderedDict()
        targets["keypoints"] = [self.joints[key]]

        if self.transforms:
            img, targets = self.transforms(img, targets)

        return img, targets

    def __len__(self):
        return self.joints.shape[0]


class LeedsSport(LeedsSportBase):
    FOLDER_NAME = "lsp_dataset_original"
    DATA_URL = "https://sam.johnson.io/research/lsp_dataset_original.zip"

    def __init__(self, data_dir: Path = Path("/tmp/"), split: Split = Split.TRAIN):
        """
        Loads dataset if it is preseint in `data_dir`.
        Downloads and loads if not.

        :param data_dir: The directory in which to put data.
        """
        super().__init__(data_dir, split)
        assert split is not Split.VAL, "This dataset does not have a canonical validation split."
        if split is Split.TRAIN:
            self.joints = self.joints[:1000]
            self.image_paths = self.image_paths[:1000]
        elif split is Split.TEST:
            self.joints = self.joints[1000:]
            self.image_paths = self.image_paths[1000:]

        self.split = split


class LeedsSportExtended(LeedsSportBase):
    FOLDER_NAME = "lsp_dataset_extended"
    DATA_URL = "https://sam.johnson.io/research/lspet_dataset.zip"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.joints = np.moveaxis(self.joints, 1, 2)


if __name__ == "__main__":
    ds = LeedsSport(split=Split.TEST)
    print(ds[0][1].shape)
    ds = LeedsSportExtended()
    print(ds[0][1].shape)
