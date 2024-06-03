import pickle
from dataclasses import dataclass
from typing import List

from koregraph.config.environment import DATASET_PATH


@dataclass
class Posture:
    timestamp: int
    posture: List[float]


dataset_folder = DATASET_PATH
keypoints_3D_folder = dataset_folder / "keypoints3d"
keypoints_2D_folder = dataset_folder / "keypoints2d"


def main():
    with open(keypoints_2D_folder / "gBR_sBM_cAll_d04_mBR0_ch01.pkl", "rb") as f:
        data = pickle.load(f)

    posture = Posture(data["timestamps"][0], data["keypoints2d"][0])

    print(posture)


if __name__ == "__main__":
    main()
