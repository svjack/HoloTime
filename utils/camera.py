import numpy as np

class CameraParams:
    def __init__(self, H: int = 512, W: int = 512, fov_h: float = np.pi/2, fov_w: float = np.pi/2):
        self.H = H
        self.W = W
        self.cx = W / 2
        self.cy = H / 2
        self.fov = (fov_h, fov_w)
        self.focal = (W / (2*np.tan(fov_w/2)), H / (2*np.tan(fov_h/2)))
        self.K = np.array([
            [self.focal[0], 0., self.W/2],
            [0., self.focal[1], self.H/2],
            [0.,            0.,       1.],
        ]).astype(np.float32)