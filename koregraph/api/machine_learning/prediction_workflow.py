from numpy import ones as np_ones
from koregraph.models.choregraphy import Choregraphy
from koregraph.managers.choregraphy import save_choregaphy_chunk
from koregraph.utils.pickle import load_pickle_object
from koregraph.api.music_to_numpy import music_to_numpy
from koregraph.api.posture_proc import upscale_posture_pred
from koregraph.tools.video_builder import (
    keypoints_video_audio_builder_from_choreography,
)
from koregraph.params import (
    AUDIO_DIRECTORY,
    MODEL_OUTPUT_DIRECTORY,
    PREDICTION_OUTPUT_DIRECTORY,
)
#from koregraph.api.audio_proc import scale_audio


def predict(audio_name: str = "mBR0", model_name: str = "model", chore_id: str = "01"):
    model_path = MODEL_OUTPUT_DIRECTORY / (model_name + ".pkl")
    model = load_pickle_object(model_path)

    audio_filepath = AUDIO_DIRECTORY / (audio_name + ".mp3")
    input = music_to_numpy(audio_filepath)

    # TODO remove this step when reshape is done in preprocessing workflow
    #input = scale_audio(input)
    input = input.reshape(-1, 1, input.shape[1])
    prediction = model.predict(input)
    prediction = upscale_posture_pred(prediction)

    print(prediction.shape)

    prediction_name = (
        model_name.replace("_", "") + "_sBM_cAll_d00_" + audio_name + f"_ch{chore_id}"
    )

    chore = Choregraphy(
        prediction_name, prediction.reshape(-1, 17, 2), np_ones(prediction.shape[0])
    )
    # Save prediction to pkl
    save_choregaphy_chunk(chore, PREDICTION_OUTPUT_DIRECTORY)

    # Create video
    keypoints_video_audio_builder_from_choreography(chore)

    print("Happy viewing!")


if __name__ == "__main__":
    predict(audio_name="mBR2", chore_id="02")
