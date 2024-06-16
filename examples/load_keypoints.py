import pickle
from dataclasses import dataclass
from typing import List

from koregraph.config.params import DATA_PATH
from koregraph.tools.visualization_3d import visualize_posture_sequence


@dataclass
class Posture:
    timestamp: int
    posture: List[float]


dataset_folder = DATA_PATH
keypoints_3D_folder = dataset_folder / "keypoints3d"
keypoints_2D_folder = dataset_folder / "keypoints2d"


def main():
    with open(keypoints_3D_folder / "gBR_sBM_cAll_d04_mBR0_ch01.pkl", "rb") as f:
        data = pickle.load(f)

    # posture = Posture(data["timestamps"][0], data["keypoints3d"][0])

    # print(data["keypoints3d_optim"][0])
    print(data.keys())
    keypoints = data["keypoints3d_optim"]
    print(keypoints.shape)


if __name__ == "__main__":
    main()
