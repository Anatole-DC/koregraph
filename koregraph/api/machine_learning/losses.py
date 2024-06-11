from numpy import ndarray, zeros, sqrt
from tensorflow import reduce_mean, square, norm, abs

from koregraph.api.preprocessing.audio_preprocessing import music_to_numpy
from koregraph.config.params import (
    AUDIO_DIRECTORY,
    GENERATED_PICKLE_DIRECTORY,
    MODEL_OUTPUT_DIRECTORY,
)
from koregraph.utils.controllers.pickles import load_pickle_object


def mse_loss(y_true: ndarray, y_pred: ndarray) -> float:
    return reduce_mean(square(y_true - y_pred))


def temporal_loss(y_true: ndarray, y_pred: ndarray) -> float:
    if y_pred.shape[0] is None or y_pred.shape[0] < 2:
        return 0.1
    diff = y_pred[:, 1:, :] - y_pred[:, :-1, :]
    return 1 / reduce_mean(square(diff))


def pose_smoothness_loss(y_true: ndarray, y_pred: ndarray) -> float:
    if y_pred.shape[0] is None or y_pred.shape[0] < 2:
        return 0
    diff = y_pred[:, 2:, :] - 2 * y_pred[:, 1:-1, :] + y_pred[:, :-2, :]
    return reduce_mean(square(diff))


def diversity_loss(y_true: ndarray, y_pred: ndarray) -> float:
    variance = reduce_mean(
        square(y_pred - reduce_mean(y_pred, axis=1, keepdims=True)), axis=1
    )
    return -reduce_mean(variance)


# Example bone pairs and lengths (this should be adapted to your specific skeleton)
BONE_PAIRS = [(0, 1), (1, 2), (2, 3)]  # Define your bone pairs
BONE_LENGTH = [1.0, 1.0, 1.0]  # Define the actual lengths of these bones


def kinematic_loss(y_true: ndarray, y_pred: ndarray) -> float:
    loss = 0
    for (i, j), l in zip(BONE_PAIRS, BONE_LENGTH):
        diff = norm(
            y_pred[:, :, i * 2 : i * 2 + 2] - y_pred[:, :, j * 2 : j * 2 + 2], axis=-1
        )
        loss += reduce_mean(square(diff - l))
    return loss


def combined_loss(
    mse_weight: float = 1.0,
    temporal_weight: float = 1.0,
    smooth_weight: float = 1.0,
    diversity_weight: float = 1.0,
    epsilon: float = 1.0,
    zeta: float = 1.0,
):
    def loss(y_true: ndarray, y_pred: ndarray) -> float:
        print(y_pred.shape)
        mse = mse_loss(y_true, y_pred)
        # temp_loss = temporal_loss(y_true, y_pred)
        smooth_loss = pose_smoothness_loss(y_true, y_pred)
        # div_loss = diversity_loss(y_true, y_pred)
        # perc_loss = perceptual_loss(y_true, y_pred)
        # kin_loss = kinematic_loss(y_pred)

        total_loss = (
            mse_weight * mse
            # + temporal_weight * temp_loss
            + smooth_weight * smooth_loss
            # + diversity_weight * div_loss
            # + epsilon * perc_loss
            # + zeta * kin_loss
        )
        return total_loss

    return loss


def prediction_distances(points: ndarray):
    distances = zeros((y_true.shape[0] - 1, 17))  # 16 distances for 17 points

    # print(sqrt(sum(diff(points, n=1, axis=0) ** 2, axis=1)))

    for i in range(0, 17, 2):
        x1 = y_pred[:-1, i]  # x coordinates of point i for all postures except the last
        y1 = y_pred[
            :-1, i + 1
        ]  # y coordinates of point i for all postures except the last
        x2 = y_pred[1:, i]  # x coordinates of point i for all postures except the first
        y2 = y_pred[
            1:, i + 1
        ]  # y coordinates of point i for all postures except the first

        distances[:, i] = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Step 3: Compute the mean square error of these distances
    mse = distances

    return mse


def danse_loss(y_true: ndarray, y_pred: ndarray) -> float:
    return reduce_mean(abs(y_true - y_pred)) + temporal_loss(y_true, y_pred)


if __name__ == "__main__":
    # y_true
    y_true, _ = load_pickle_object(
        GENERATED_PICKLE_DIRECTORY / "generated_gBR_sFM_cAll_d04_mBR0_ch01.pkl"
    )

    # Load y_pred
    model_path = MODEL_OUTPUT_DIRECTORY / ("test_full_train" + ".pkl")
    model = load_pickle_object(model_path)

    audio_filepath = AUDIO_DIRECTORY / ("mBR0" + ".mp3")
    input = music_to_numpy(audio_filepath)

    # TODO remove this step when reshape is done in preprocessing workflow
    # input = scale_audio(input)
    input = input.reshape(-1, 1, input.shape[1])
    y_pred = model.predict(input)
    y_pred = y_pred[: len(y_true)]

    assert (
        y_true.shape == y_pred.shape
    ), f"Shapes don't match {y_true.shape=} {y_pred.shape=}"

    # Test loss function
    danse_loss(y_true, y_pred)
