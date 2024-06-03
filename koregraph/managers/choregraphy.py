from datetime import timedelta
import pickle
from pathlib import Path
from os import environ
from typing import Dict

from koregraph.models.choregraphy import Choregraphy


KEYPOINTS_DIRECTORY: Path = Path(
    environ.get(
        "KEYPOINTS_DIRECTORY",
        "data/keypoints2d",
    )
)


def load_choregraphy(name: str) -> Choregraphy:
    """Load and return a choregraphy from a pickle file.

    Args:
        name (str): The choregraphy file's name.

    Returns:
        Choregraphy: The loaded Choregraphy.
    """

    with open(KEYPOINTS_DIRECTORY / f"{name}.pkl", "rb") as keypoints_file:
        choregraphy_raw: Dict = pickle.load(keypoints_file)
    loaded_choregraphy = Choregraphy(
        name,
        choregraphy_raw["keypoints2d"][
            0, :, :, :2
        ],  # Take the first view (among nine), all postures, all keypoints, only the x and y coordinates
        choregraphy_raw["timestamps"],
    )

    assert len(loaded_choregraphy.keypoints2d) == len(
        loaded_choregraphy.timestamps
    ), f"In loaded choregraphy {name}, not the same number of postures and timestamps"

    return loaded_choregraphy

def save_choregaphy_chunk(chore: Choregraphy, path: Path) -> None:
    """Saves a choregraphy chunk to a pickle file.

    Args:
        data (Choregraphy): The choregraphy chunk
        path (Path): The path to where we'll save the choregraphy chunk.

    Returns:
        Nothing
    """
    with open(path / f"{chore.name}.pkl", "wb") as handle:
        pickle.dump(
            {
            'keypoints2d': chore.keypoints2d,
            'timestamps': chore.timestamps
            },
            handle, pickle.HIGHEST_PROTOCOL
        )


if __name__ == "__main__":
    choregraphy = load_choregraphy("gWA_sBM_cAll_d26_mWA4_ch07")
    print(
        timedelta(
            microseconds=int(choregraphy.timestamps[-5] - choregraphy.timestamps[-6])
        )
    )
