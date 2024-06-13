import aniposelib
import os
import json


def load_camera_group(camera_dir, env_name):
    """Load a set of cameras in the environment."""
    file_path = os.path.join(camera_dir, f"{env_name}.json")
    assert os.path.exists(file_path), f"File {file_path} does not exist!"
    with open(file_path, "r") as f:
        params = json.load(f)
    cameras = []
    for param_dict in params:
        camera = aniposelib.cameras.Camera(
            name=param_dict["name"],
            size=param_dict["size"],
            matrix=param_dict["matrix"],
            rvec=param_dict["rotation"],
            tvec=param_dict["translation"],
            dist=param_dict["distortions"],
        )
        cameras.append(camera)
    camera_group = aniposelib.cameras.CameraGroup(cameras)
    return camera_group
