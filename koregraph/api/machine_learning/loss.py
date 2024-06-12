from tensorflow import reduce_mean, square, expand_dims
from keras.saving import register_keras_serializable


def distance_frame_to_frame(frame1, frame2):
    distance = 0
    for i in range(0, 34, 2):
        distance += (frame1[:, i] - frame2[:, i]) ** 2 + (
            frame1[:, (i + 1)] - frame2[:, (i + 1)]
        ) ** 2

    return distance


@register_keras_serializable()
def my_mse(y_true, y_pred):
    distances = distance_frame_to_frame(y_true[::, :], y_pred[::, :])

    return reduce_mean(square(distances))


@register_keras_serializable()
def my_mse_maximise_movement(y_true, y_pred):
    first_frame = expand_dims(y_pred[0, :], 0)
    last_frame = expand_dims(y_pred[-1, :], 0)

    print(first_frame.shape)
    print(last_frame.shape)
    mvt = distance_frame_to_frame(first_frame, last_frame)
    reg = 1/(mvt + 0.000001)
    distances = distance_frame_to_frame(
        y_true[::, :], y_pred[::, :]
    ) + reg

    return reduce_mean(square(distances))
