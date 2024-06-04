import os
import shutil

from koregraph.tools.choregraphy_to_chunks import split_sequence, split_audio
from koregraph.managers.choregraphy import save_choregaphy_chunk
from koregraph.managers.audio import save_audio_chunk
from koregraph.params import GENERATED_AUDIO_DIRECTORY, GENERATED_KEYPOINTS_DIRECTORY, CHUNK_SIZE

def reset_chunks(path, reload=True):
    if os.path.exists(path) and not os.path.isfile(path) and reload:
        try:
            shutil.rmtree(path)
        except Exception as e:
            print(f"Failed to delete {path}. Reason: {e}")
    path.mkdir(parents=True, exist_ok=True)


def generate_chunk(choregraphy_name: str, chunk_size: int = CHUNK_SIZE, reload_music: bool = False):
    # Clean previous chunks out if needed
    chore_path = GENERATED_KEYPOINTS_DIRECTORY / choregraphy_name / chunk_size
    reset_chunks(chore_path)
    _, _, _, _, music_name, _ = choregraphy_name.split("_")
    music_path = GENERATED_AUDIO_DIRECTORY / music_name / chunk_size
    reset_chunks(music_path, reload_music)

    # Get and save chunks
    chores = split_sequence(choregraphy_name, int(chunk_size))
    for chore in chores:
        save_choregaphy_chunk(chore, chore_path)

    if reload_music or len(os.listdir(music_path)) == 0:
        musics, sr = split_audio(choregraphy_name, chunk_size_sec=int(chunk_size))
        for music, chunk_id in musics:
            path = music_path / (music_name + f"_{chunk_id}.mp3")
            save_audio_chunk(music, sr, path)
