import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def visualize_posture_sequence(posture_sequence, fps=60):
    """
    Visualize a sequence of 3D postures using matplotlib.

    Parameters:
    - posture_sequence: numpy array of shape (n, 17, 3), where n is the number of frames
    - fps: frames per second for the animation
    """
    n, num_keypoints, _ = posture_sequence.shape

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set the limits for the axes
    ax.set_xlim(np.min(posture_sequence[:, :, 0]), np.max(posture_sequence[:, :, 0]))
    ax.set_ylim(np.min(posture_sequence[:, :, 1]), np.max(posture_sequence[:, :, 1]))
    ax.set_zlim(np.min(posture_sequence[:, :, 2]), np.max(posture_sequence[:, :, 2]))

    # Create a scatter plot for the keypoints
    scatter = ax.scatter([], [], [])

    def init():
        scatter._offsets3d = ([], [], [])
        return scatter,

    def update(frame):
        x = posture_sequence[frame, :, 0]
        y = posture_sequence[frame, :, 1]
        z = posture_sequence[frame, :, 2]
        scatter._offsets3d = (x, y, z)
        return scatter,

    # Create the animation
    ani = FuncAnimation(fig, update, frames=n, init_func=init, blit=True, interval=100)

    plt.show()
