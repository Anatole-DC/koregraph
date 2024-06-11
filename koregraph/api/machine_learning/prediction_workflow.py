from numpy import ones as np_ones
from tensorflow.keras.models import load_model
from scipy.signal import savgol_filter

from koregraph.models.choregraphy import Choregraphy
from koregraph.utils.controllers.choregraphies import save_choregaphy_chunk
from koregraph.api.preprocessing.audio_preprocessing import music_to_numpy
from koregraph.api.preprocessing.posture_preprocessing import (
    upscale_posture_pred,
    posture_array_to_keypoints,
)
from koregraph.tools.video_builder import (
    keypoints_video_audio_builder_from_choreography,
)
from koregraph.config.params import (
    AUDIO_DIRECTORY,
    MODEL_OUTPUT_DIRECTORY,
    PREDICTION_OUTPUT_DIRECTORY,
)
from koregraph.api.preprocessing.audio_preprocessing import scale_audio

def smooth_predictions(predictions, window_length=50, polyorder=2):
    """Smooth predictions using Savitzky-Golay filter."""
    smoothed_predictions = savgol_filter(predictions, window_length, polyorder, axis=0)
    return smoothed_predictions


def predict(audio_name: str = "mBR0", model_name: str = "model", backup: bool = False, smooth = True):
    # model_path = MODEL_OUTPUT_DIRECTORY / (model_name + ".pkl")
    # model = load_pickle_object(model_path)

    model_path = (
        MODEL_OUTPUT_DIRECTORY
        / model_name
        / f"{model_name}{'_backup' if backup else ''}.keras"
    )
    model = load_model(model_path, compile=False)

    audio_filepath = AUDIO_DIRECTORY / (audio_name + ".mp3")
    input = music_to_numpy(audio_filepath)

    # TODO remove this step when reshape is done in preprocessing workflow
    # input = scale_audio(input)
    input = input.reshape(-1, 1, input.shape[1])
    prediction = model.predict(input)
    prediction = upscale_posture_pred(posture_array_to_keypoints(prediction))

    if smooth:
        prediction = smooth_predictions(prediction)

    prediction_name = (
        (model_name + "_" + audio_name) if audio_name not in model_name else model_name
    )

    chore = Choregraphy(prediction_name, prediction.reshape(-1, 17, 2))
    # Save prediction to pkl
    save_choregaphy_chunk(chore, PREDICTION_OUTPUT_DIRECTORY)

    # Create video
    keypoints_video_audio_builder_from_choreography(chore, audio_name)

    print("Happy viewing!")


if __name__ == "__main__":
    predict(audio_name="mBR2", chore_id="02")
