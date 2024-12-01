
import numpy as np


class TransformGesture:
    def __call__(self, gesture):
        x = []
        y = []
        z = []
        for frame in gesture:
            for pos in frame:
                x.append(pos[0])
                y.append(pos[1])
                z.append(pos[2])

        x_mean = np.mean(x)
        y_mean = np.mean(y)
        z_mean = np.mean(z)
        x_range = max(x) - min(x)
        y_range = max(y) - min(y)
        z_range = max(z) - min(z)

        hand_size = max(x_range, y_range, z_range)
        if hand_size == 0:
                hand_size = 1e-6
        transformed_gesture = []
        for frame in gesture:
            transformed_gesture.append([])
            for pos in frame:
                transformed_gesture[-1].append([
                    (pos[0] - x_mean) / hand_size / 3 + 0.2,
                    (pos[1] - y_mean) / hand_size / 3 + 0.2,
                    (pos[2] - z_mean) / hand_size / 3 + 0.2,
                ])
        return transformed_gesture