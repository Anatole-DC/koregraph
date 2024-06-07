import os
from pydub import AudioSegment
from koregraph.config.params import AUDIO_DIRECTORY, GENERATED_AUDIO_SILENCE_DIRECTORY


def add_silence_to_audio_files():
    """Add one second of silence to the beginning and end of each audio file in the input directory.
    The new audio files are saved in the output directory.
    """
    input_directory = AUDIO_DIRECTORY
    output_directory = GENERATED_AUDIO_SILENCE_DIRECTORY

    # Vérifier si le répertoire de sortie existe, sinon le créer
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created output directory: {output_directory}")

    # Parcourir tous les fichiers dans le répertoire d'entrée
    for filename in os.listdir(input_directory):
        if filename.endswith(".mp3"):
            input_audio_path = os.path.join(input_directory, filename)
            output_audio_path = os.path.join(output_directory, f"silence_{filename}")

            print(f"Processing file: {input_audio_path}")

            # Charger le fichier audio
            try:
                audio = AudioSegment.from_file(input_audio_path)
            except Exception as e:
                print(f"Failed to load {input_audio_path}: {e}")
                continue

            # Créer une seconde de silence
            one_second_silence = AudioSegment.silent(
                duration=1000
            )  # 1000 ms = 1 seconde

            # Ajouter la seconde de silence au début et à la fin
            audio_with_silence = one_second_silence + audio + one_second_silence

            # Exporter le nouveau fichier audio
            try:
                audio_with_silence.export(output_audio_path, format="mp3")
                print(f"Exported file to: {output_audio_path}")
            except Exception as e:
                print(f"Failed to export {output_audio_path}: {e}")
