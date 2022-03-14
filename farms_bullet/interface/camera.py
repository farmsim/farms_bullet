"""Camera"""

import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pybullet
import farms_pylog as pylog


class Camera:
    """Camera"""

    def __init__(self, timestep: float, target_identity: int = None, **kwargs):
        super().__init__()
        self.target = target_identity
        self.timestep = timestep
        self.yaw = kwargs.pop('yaw')
        self.pitch = kwargs.pop('pitch')
        self.distance = kwargs.pop('distance')
        self.yaw_speed = kwargs.pop('yaw_speed', 0)
        self.motion_filter = kwargs.pop('motion_filter', None)
        if self.motion_filter is None:
            self.motion_filter = 10*timestep
        assert not kwargs, kwargs

    @staticmethod
    def get_camera():
        """Get camera information"""
        return pybullet.getDebugVisualizerCamera()

    def update_yaw(self):
        """Update yaw"""
        self.yaw += self.yaw_speed*self.timestep


class CameraTarget(Camera):
    """Camera with target following"""

    def __init__(self, target_identity: int, **kwargs):
        super().__init__(**kwargs)
        self.target = target_identity
        self.target_pos = kwargs.pop(
            'target_pos',
            np.array(pybullet.getBasePositionAndOrientation(self.target)[0])
            if self.target is not None
            else np.array(self.get_camera()[11])
        )

    def update_target_pos(self):
        """Update target position"""
        self.target_pos = (
            (1-self.motion_filter)*self.target_pos
            + self.motion_filter*np.array(
                pybullet.getBasePositionAndOrientation(self.target)[0]
            )
        )


class UserCamera(CameraTarget):
    """UserCamera"""

    def __init__(self, target_identity: int, **kwargs):
        super().__init__(target_identity, **kwargs)
        self.update(use_camera=False)

    def set_zoom(self, value):
        """Set zoom"""
        self.distance = value
        pybullet.resetDebugVisualizerCamera(
            cameraDistance=self.distance,
            cameraYaw=self.yaw,
            cameraPitch=self.pitch,
            cameraTargetPosition=self.target_pos
        )

    def update(self, use_camera: bool = True):
        """Camera view"""
        if use_camera:
            (
                self.yaw,
                self.pitch,
                self.distance,
                target_pos
            ) = self.get_camera()[8:12]
            self.target_pos = np.array(target_pos)
        self.update_yaw()
        if self.target is not None:
            self.update_target_pos()
        pybullet.resetDebugVisualizerCamera(
            cameraDistance=self.distance,
            cameraYaw=self.yaw,
            cameraPitch=self.pitch,
            cameraTargetPosition=self.target_pos
        )


class CameraRecord(CameraTarget):
    """Camera recording"""

    def __init__(
            self,
            timestep: float,
            target_identity: int,
            n_iterations: int,
            fps: float,
            **kwargs,
    ):
        super().__init__(
            target_identity=target_identity,
            timestep=timestep,
            **kwargs,
        )
        self.width = kwargs.pop('width', 1280)
        self.height = kwargs.pop('height', 720)
        self.skips = kwargs.pop('skips', max(0, int(1//(timestep*fps))-1))
        self.fps = 1/(self.timestep*(self.skips+1))
        self.data = np.zeros(
            [n_iterations//(self.skips+1)+1, self.height, self.width, 3],
            dtype=np.uint8
        )

    def record(self, iteration: int):
        """Record camera"""
        if not iteration % (self.skips+1):
            sample = iteration if not self.skips else iteration//(self.skips+1)
            self.update_yaw()
            self.update_target_pos()
            self.data[sample, :, :, :] = pybullet.getCameraImage(
                width=self.width,
                height=self.height,
                viewMatrix=pybullet.computeViewMatrixFromYawPitchRoll(
                    cameraTargetPosition=self.target_pos,
                    distance=self.distance,
                    yaw=self.yaw,
                    pitch=self.pitch,
                    roll=0,
                    upAxisIndex=2
                ),
                projectionMatrix=pybullet.computeProjectionMatrixFOV(
                    fov=60,
                    aspect=self.width/self.height,
                    nearVal=0.1,
                    farVal=10
                ),
                renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
                flags=pybullet.ER_NO_SEGMENTATION_MASK
            )[2][:, :, :3]

    def save(
            self,
            filename: str = 'video.avi',
            iteration: int = None,
            writer: str = 'ffmpeg',
    ):
        """Save recording"""
        data = (
            self.data[:iteration//(self.skips+1)]
            if iteration is not None
            else self.data
        )
        ffmpegwriter = manimation.writers[writer]
        pylog.debug(
            'Recording video to %s with %s (fps=%s, skips=%s)',
            filename,
            writer,
            self.fps,
            self.skips,
        )
        metadata = dict(
            title='FARMS simulation',
            artist='FARMS',
            comment='FARMS simulation'
        )
        writer = ffmpegwriter(fps=self.fps, metadata=metadata)
        size = 10
        fig = plt.figure(
            'Recording',
            figsize=(size, size*self.height/self.width)
        )
        fig_ax = plt.gca()
        ims = None
        with writer.saving(fig, filename, dpi=self.width/size):
            for frame in tqdm(data):
                ims = render_matplotlib_image(fig_ax, frame, ims=ims)
                writer.grab_frame()
        plt.close(fig)


def render_matplotlib_image(fig_ax, img, ims=None, cbar_label='', clim=None):
    """Render matplotlib image"""
    if ims is None:
        ims = plt.imshow(img)
        fig_ax.get_xaxis().set_visible(False)
        fig_ax.get_yaxis().set_visible(False)
        fig_ax.set_aspect(aspect=1)
        if cbar_label:
            divider = make_axes_locatable(fig_ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = plt.colorbar(ims, cax=cax)
            cbar.set_label(cbar_label, rotation=90)
        if clim:
            plt.clim(clim)
        plt.tight_layout()
    else:
        ims.set_data(img)
    return ims
