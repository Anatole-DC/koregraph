import numpy as np
from koregraph.api.preprocessing.posture_proc import (
    generate_posture_array,
    upscale_posture_pred,
)
from koregraph.config.params import PREDICTION_OUTPUT_DIRECTORY
from koregraph.tools.video_builder import (
    keypoints_video_audio_builder_from_choreography,
)
from koregraph.models.choregraphy import Choregraphy
from koregraph.utils.choregraphies import save_choregaphy_chunk


def generate_and_export_choreography(posture_file_2):
    """Generate and export a choreography with interpolated posture arrays.
    Args:
        posture_file_2 (str): The name of the posture file where you want to interpolate the posture arrays.
        Returns:
        np.array: The final array of postures."""

    posture_file_1 = "gPO_sBM_cAll_d10_mPO1_ch01.pkl"

    # Définition des arrays de posture
    posture_array_1 = generate_posture_array(posture_file_1)
    posture_array_2 = generate_posture_array(posture_file_2)

    # Nombre de lignes pour chaque partie de l'array final
    n_rows_part1 = 60
    n_rows_part2 = len(posture_array_2)
    n_rows_part3 = n_rows_part1

    # Initialisation de l'array final
    final_array = np.zeros(
        (n_rows_part1 + n_rows_part2 + n_rows_part3, posture_array_1.shape[1])
    )

    # Ajout de la première partie (transition de posture_array_1 à posture_array_2)
    for i in range(n_rows_part1):
        final_array[i] = posture_array_1[0] + (
            posture_array_2[0] - posture_array_1[0]
        ) * (i / n_rows_part1)

    # Ajout de la deuxième partie (toutes les lignes de posture_array_2)
    final_array[n_rows_part1 : n_rows_part1 + n_rows_part2] = posture_array_2

    # Ajout de la troisième partie (transition de la dernière posture de posture_array_2 à la première posture de posture_array_1)
    for i in range(n_rows_part3):
        final_array[n_rows_part1 + n_rows_part2 + i] = posture_array_2[-1] + (
            posture_array_1[0] - posture_array_2[-1]
        ) * (i / n_rows_part3)

    upscaled_posture = upscale_posture_pred(final_array)

    export_name = f"interpolated_chore_{posture_file_2}"

    # Create Choregraphy object
    chore = Choregraphy(
        export_name, final_array.reshape(-1, 17, 2), np.ones(upscaled_posture.shape[0])
    )

    return chore.keypoints2d.reshape(-1, 34)


# Utilisation de la fonction
if __name__ == "__main__":
    generate_and_export_choreography()
