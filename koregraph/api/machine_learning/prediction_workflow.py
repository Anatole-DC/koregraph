from numpy import ones as np_ones, concatenate, append
from koregraph.models.choregraphy import Choregraphy
from koregraph.managers.choregraphy import save_choregaphy_chunk
from koregraph.utils.pickle import load_pickle_object
from koregraph.api.music_to_numpy import music_to_numpy
from koregraph.api.posture_proc import upscale_posture_pred, scale_posture_pred
from koregraph.tools.video_builder import (
    keypoints_video_audio_builder_from_choreography,
)
from koregraph.params import (
    AUDIO_DIRECTORY,
    MODEL_OUTPUT_DIRECTORY,
    PREDICTION_OUTPUT_DIRECTORY,
    PERCENTAGE_CUT,
    GENERATED_KEYPOINTS_DIRECTORY,
    GENERATED_AUDIO_DIRECTORY,
    CHUNK_SIZE,
    FRAME_FORMAT,
)
from koregraph.utils.preproc import cut_percentage
from koregraph.api.audio_proc import scale_audio


def predict(audio_name: str = "mBR0", model_name: str = "model", chunk: bool = False):
    model_path = MODEL_OUTPUT_DIRECTORY / (model_name + ".pkl")
    model = load_pickle_object(model_path)

    audio_filepath = AUDIO_DIRECTORY / (audio_name + ".mp3")
    input = music_to_numpy(audio_filepath)

    if chunk:
        print("input shape", input.shape)
        input = input.reshape(1, input.shape[0], input.shape[1])
    else:
        # TODO remove this step when reshape is done in preprocessing workflow
        # input = scale_audio(input)
        input = input.reshape(-1, 1, input.shape[1])
        print("input min:", input.min())
        print("input max:", input.max())

    print(input.shape)
    prediction = model.predict(input)
    # prediction = upscale_posture_pred(prediction)

    print("Prediction shape:", prediction.shape)
    print("predciton min", prediction.min())
    print("predciton max", prediction.max())
    prediction_name = (
        model_name.replace("_", "-") + "_sBM_cAll_d00_" + audio_name + "_ch01"
    )

    chore = Choregraphy(
        prediction_name, prediction.reshape(-1, 17, 2), np_ones(prediction.shape[0])
    )
    # Save prediction to pkl
    save_choregaphy_chunk(chore, PREDICTION_OUTPUT_DIRECTORY)

    # Create video
    keypoints_video_audio_builder_from_choreography(chore)

    print("Happy viewing!")


def predict_next_move(
    audio_name: str = "mBR0",
    model_name: str = "model",
    chore_chunk_name: str = "gBR_sFM_cAll_d04_mBR0_ch01",
    chunk_id: int = 0,
    perc_cut: float = PERCENTAGE_CUT,
):
    def build_next_input(first_frames, prediction):
        prediction_frame_size = int((CHUNK_SIZE * perc_cut) * 60)
        first_frames = first_frames.reshape(
            -1, int((CHUNK_SIZE * (1 - perc_cut)) * 60), 17, 2
        )
        prediction = prediction.reshape(-1, prediction_frame_size, 17, 2)

        print(first_frames.shape)
        print(prediction.shape)
        return concatenate(
            (first_frames[0, prediction_frame_size:, :, :], prediction[0, :, :, :]),
            axis=0,
        )

    model_path = MODEL_OUTPUT_DIRECTORY / (model_name + ".pkl")
    model = load_pickle_object(model_path)

    audio_filepath = (
        GENERATED_AUDIO_DIRECTORY
        / audio_name
        / str(CHUNK_SIZE)
        / (f"{audio_name}_{chunk_id}.mp3")
    )
    audio = music_to_numpy(audio_filepath)

    # cut input, take 8 sec
    print("Before cut", audio.shape)
    audio, _ = cut_percentage(audio, perc_cut)
    print("After cut", audio.shape)
    print("min audio: ", audio.min())
    print("max audio: ", audio.max())
    # add beginning of chore
    chore = load_pickle_object(
        GENERATED_KEYPOINTS_DIRECTORY
        / chore_chunk_name
        / str(CHUNK_SIZE)
        / (f"{chore_chunk_name}_{chunk_id}.pkl")
    )
    input = chore["keypoints2d"]

    input = scale_posture_pred(input)
    input = input.reshape(-1, 34)
    input, _ = cut_percentage(input.reshape(-1, 34), perc_cut)

    print("min input: ", input.min())
    print("max input: ", input.max())

    input = input.reshape(-1, int((CHUNK_SIZE * (1 - perc_cut)) * 60), 17, 2)

    print(input.shape)
    all_output = input

    # Predict first frame
    print("Prediction 0")
    prediction = model.predict(input)
    print("prediction shape", prediction.shape)
    print("prediction min", prediction.min())
    print("prediction max", prediction.max())
    all_output = concatenate(
        (all_output[0].reshape(-1, 17, 2), prediction.reshape(-1, 17, 2)), axis=0
    )

    # Predict next frame
    for i in range(4):
        print(f"Prediction {i+1}")
        input = build_next_input(input, prediction)
        input = input.reshape(-1, int((CHUNK_SIZE * (1 - perc_cut)) * 60), 17, 2)
        print(f"Next input shape {input.shape}")
        prediction = model.predict(input)
        all_output = concatenate((all_output, prediction.reshape(-1, 17, 2)), axis=0)
        print("prediction shape", prediction.shape)
        print("prediction min", prediction.min())
        print("prediction max", prediction.max())

    prediction = upscale_posture_pred(prediction)
    all_output = upscale_posture_pred(all_output)

    print("Prediction shape:", prediction.shape)
    print("All shape:", all_output.shape)
    prediction_name = (
        model_name.replace("_", "-") + "_sBM_cAll_d00_" + audio_name + "_ch01"
    )
    output_name = model_name.replace("_", "-") + "_sBM_cAll_d00_" + audio_name + "_ch02"

    # print('chore')
    # print(chore.shape)
    # print(chore)
    output = append(upscale_posture_pred(input), prediction)

    chore = Choregraphy(
        prediction_name, output.reshape(-1, 17, 2), np_ones(output.shape[0])
    )
    chore_all = Choregraphy(
        output_name, all_output.reshape(-1, 17, 2), np_ones(all_output.shape[0])
    )
    # Save prediction to pkl
    save_choregaphy_chunk(chore, PREDICTION_OUTPUT_DIRECTORY)
    save_choregaphy_chunk(chore_all, PREDICTION_OUTPUT_DIRECTORY)

    # Create video
    keypoints_video_audio_builder_from_choreography(chore)
    # keypoints_video_audio_builder_from_choreography(chore_all)

    print("Happy viewing!")


if __name__ == "__main__":
    predict(audio_name="mBR2", chore_id="02")
