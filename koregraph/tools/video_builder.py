from imageio import get_writer
from pathlib import Path
from os import environ

from numpy import asarray, isnan
from PIL.Image import Image, new as new_pillow_image
from PIL.ImageDraw import Draw
from moviepy.editor import AudioFileClip, VideoFileClip, VideoClip

from koregraph.managers.choregraphy import load_choregraphy
from koregraph.models.aist_file import AISTFile
from koregraph.models.choregraphy import Choregraphy
from koregraph.models.posture import Posture


KEYPOINTS_BUILDER_TEMP_DIRECTORY = Path(
    environ.get("KEYPOINTS_BUILDER_TEMP_DIRECTORY", "temp")
)
KEYPOINTS_BUILDER_TEMP_DIRECTORY.mkdir(parents=True, exist_ok=True)

COLORS = [
    [255, 0, 0],
    [255, 85, 0],
    [255, 170, 0],
    [255, 255, 0],
    [170, 255, 0],
    [85, 255, 0],
    [0, 255, 0],
    [0, 255, 85],
    [0, 255, 170],
    [0, 255, 255],
    [0, 170, 255],
    [0, 85, 255],
    [0, 0, 255],
    [85, 0, 255],
    [170, 0, 255],
    [255, 0, 255],
    [255, 0, 170],
    [255, 0, 85],
]


def draw_keypoints(posture: Posture, frame_format=(1920, 1080), radius=5) -> Image:
    """Draw the 2D posture onto a new image and return the image.

    Args:
        posture (Posture): The posture to draw.
        frame_format (tuple, optional): The frame dimension. Defaults to (720, 720).

    Returns:
        Image: The image with the drawn keypoints.
    """

    new_frame = new_pillow_image("RGB", frame_format)
    draw = Draw(new_frame)
    for index, ((x, y), ((x1, y1), (x2, y2))) in enumerate(
        zip(posture.keypoints, posture.bones())
    ):
        if isnan(x) or isnan(y) or x < 0 or y < 0:
            continue
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius), fill=tuple(COLORS[index])
        )
        draw.line((x1, y1, x2, y2), fill=tuple(COLORS[index]))
    return new_frame


def export_choregraphy_keypoints(
    choregraphy: Choregraphy, export_name: str = None
) -> Path:
    """Builds the choregraphy video without sound.

    Args:
        choregraphy_name (str): The choregraphy to build the video from.
        export_name (str, optional): The name of the temp export. Defaults to None (choregraphy_name will be used).

    Returns:
        Path: The Path where the temp video was exported.
    """

    if export_name is None:
        export_name = choregraphy.name
    export_path = Path(f"temp/{export_name}_soundless.mp4")

    with get_writer(export_path, mode="I", fps=60) as video_writer:
        [
            video_writer.append_data(asarray(draw_keypoints(Posture(keypoints))))
            for keypoints in choregraphy.keypoints2d
        ]

    return export_path


def keypoints_video_audio_builder(choregraphy_name: str):
    """Export a video with audio and drawn keypoints.

    Args:
        choregraphy_name (str): The choregraphy to build the video from.
    """

    return keypoints_video_audio_builder_from_choreography(
        load_choregraphy(choregraphy_name)
    )


def keypoints_video_audio_builder_from_choreography(choregraphy: Choregraphy):
    """Export a video with audio and drawn keypoints.

    Args:
        choregraphy (Choregraphy): The choregraphy to build the video from.
    """

    aist_file = AISTFile(choregraphy.name)

    print(f"Choregraphy '{choregraphy.name}' : {len(choregraphy.keypoints2d)} postures")

    video_path = export_choregraphy_keypoints(choregraphy)

    video = VideoFileClip(str(video_path.absolute()))
    audio = AudioFileClip(str(aist_file.music.absolute()))

    final_filename = Path(f"temp/{choregraphy.name}.mp4")
    final_filename.unlink(missing_ok=True)

    final_video: VideoClip = video.set_audio(audio)
    final_video.write_videofile(
        fps=60, codec="libx264", filename=str(final_filename.absolute())
    )
    video_path.unlink()


if __name__ == "__main__":
    keypoints_video_audio_builder("gWA_sBM_cAll_d26_mWA4_ch07")
