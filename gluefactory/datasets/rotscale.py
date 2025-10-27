"""
Simply load images from a folder or nested folders (does not have any split).
"""

import argparse
import logging
import tarfile
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from ..settings import DATA_PATH
from ..utils.image import ImagePreprocessor, load_image
from ..utils.tools import fork_rng
from ..visualization.viz2d import plot_image_grid
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


def read_homography(path):
    with open(path) as f:
        result = []
        for line in f.readlines():
            while "  " in line:  # Remove double spaces
                line = line.replace("  ", " ")
            line = line.replace(" \n", "").replace("\n", "")
            # Split and discard empty strings
            elements = list(filter(lambda s: s, line.split(" ")))
            if elements:
                result.append(elements)
        return np.array(result).astype(float)


class RotScale(BaseDataset, torch.utils.data.Dataset):
    default_conf = {
        "preprocessing": ImagePreprocessor.default_conf,
        "data_dir": "rotscale",
        "subset": None,
        "grayscale": False,
    }


    url1 = "https://raw.githubusercontent.com/YoniChechik/AI_is_Math/master/c_08_features/left.jpg"
    url2 = "https://raw.githubusercontent.com/YoniChechik/AI_is_Math/master/c_08_features/right.jpg"

    def _init(self, conf):
        assert conf.batch_size == 1
        self.preprocessor = ImagePreprocessor(conf.preprocessing)

        self.root = DATA_PATH / conf.data_dir
        if not self.root.exists():
            logger.info("Downloading the HPatches dataset.")
            self.download()
        self.pair = [x.name for x in self.root.iterdir()]

        self.items = []  # (seq, q_idx, is_illu)
        self.items.append(self.pair)

    def download(self):
        data_dir = Path(os.path.join(self.root.parent, 'rotscale'))
        data_dir.mkdir(exist_ok=True, parents=True)
       
        torch.hub.download_url_to_file(self.url1, data_dir.joinpath('left.jpg').__str__())
        torch.hub.download_url_to_file(self.url2, data_dir.joinpath('right.jpg').__str__())

    def get_dataset(self, split):
        assert split in ["val", "test"]
        return self

    def _read_image(self, seq: int, idx: int) -> dict:
        img_file = self.root / f"{self.items[seq][idx]}"
        img = load_image(img_file, self.conf.grayscale)
        return self.preprocessor(img)

    def __getitem__(self, idx):
        data0 = self._read_image(idx, 0)
        data1 = self._read_image(idx, 1)
        H = np.eye(3, dtype=np.float32)
        H = data1["transform"] @ H @ np.linalg.inv(data0["transform"])
        file1 = (self.root / f"{self.items[idx][0]}").__str__()
        file2 = (self.root / f"{self.items[idx][1]}").__str__()
        return {
            "H_0to1": H.astype(np.float32),
            "idx": idx,
            "name": f"rotscale_{idx}",
            "name1": file1,
            "name2": file2,
            "view0": data0,
            "view1": data1,
        }

    def __len__(self):
        return len(self.items)


def visualize(args):
    conf = {
        "batch_size": 1,
        "num_workers": 8,
        "prefetch_factor": 1,
    }
    conf = OmegaConf.merge(conf, OmegaConf.from_cli(args.dotlist))
    dataset = RotScale(conf)
    loader = dataset.get_data_loader("test")
    logger.info("The dataset has %d elements.", len(loader))

    with fork_rng(seed=dataset.conf.seed):
        images = []
        for _, data in zip(range(args.num_items), loader):
            images.append(
                [data[f"view{i}"]["image"][0].permute(1, 2, 0) for i in range(2)]
            )
    plot_image_grid(images, dpi=args.dpi)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from .. import logger  # overwrite the logger

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_items", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("dotlist", nargs="*")
    args = parser.parse_intermixed_args()
    visualize(args)
