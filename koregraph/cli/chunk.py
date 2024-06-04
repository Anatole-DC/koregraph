from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
import os
import shutil

from koregraph.tools.choregraphy_to_chunks import split_sequence, split_audio
from koregraph.managers.choregraphy import save_choregaphy_chunk
from koregraph.managers.audio import save_audio_chunk
from koregraph.params import GENERATED_AUDIO_DIRECTORY, GENERATED_KEYPOINTS_DIRECTORY


parser = ArgumentParser(
    "Koregraph chunk",
    description="Use this create keypoints + audio chunks",
)

parser.add_argument(
    "-c", "--choregraphy", dest="choregraphy", required=True, help="Choregraphy name"
)
parser.add_argument(
    "-s", "--size", dest="chunk_size", required=True, help="Size of chunks in seconds"
)

parser.add_argument(
    "--reload-music",
    dest="reload_music",
    required=False,
    action=BooleanOptionalAction,
    help="The music will be split again for this chunk size if it already exists",
)


def main():
    def clean_path(path, reload=True):
        if os.path.exists(path) and not os.path.isfile(path) and reload:
            try:
                shutil.rmtree(path)
            except Exception as e:
                print(f"Failed to delete {path}. Reason: {e}")
        path.mkdir(parents=True, exist_ok=True)

    # Create folders where we will put the chunks
    GENERATED_KEYPOINTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    GENERATED_AUDIO_DIRECTORY.mkdir(parents=True, exist_ok=True)

    arguments = parser.parse_args()

    choregraphy_name = Path(arguments.choregraphy).stem
    chunk_size = arguments.chunk_size
    reload_music = arguments.reload_music

    # Clean previous chunks out if needed
    chore_path = GENERATED_KEYPOINTS_DIRECTORY / choregraphy_name / chunk_size
    clean_path(chore_path)
    _, _, _, _, music_name, _ = choregraphy_name.split("_")
    music_path = GENERATED_AUDIO_DIRECTORY / music_name / chunk_size
    clean_path(music_path, arguments.reload_music)

    # Get and save chunks
    chores = split_sequence(choregraphy_name, int(chunk_size))
    for chore in chores:
        save_choregaphy_chunk(chore, chore_path)

    if reload_music or len(os.listdir(music_path)) == 0:
        musics, sr = split_audio(choregraphy_name, chunk_size_sec=int(chunk_size))
        for music, chunk_id in musics:
            path = music_path / (music_name + f"_{chunk_id}.mp3")
            save_audio_chunk(music, sr, path)


if __name__ == "__main__":
    main()
