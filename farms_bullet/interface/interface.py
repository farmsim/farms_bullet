"""Interface"""

import pybullet
from farms_core import pylog
from farms_core.simulation.options import SimulationOptions
from .camera import UserCamera, CameraRecord


class DebugParameter:
    """DebugParameter"""

    def __init__(
            self,
            name: str,
            val: float,
            val_min: float,
            val_max: float,
    ):
        super().__init__()
        self.name = name
        self.value = val
        self.val_min = val_min
        self.val_max = val_max
        self._handler = None
        self.changed = False

    def init(self):
        """Initialise"""
        self.add(self.value)

    def add(self, value: float):
        """Add parameter"""
        if self._handler is None:
            self._handler = pybullet.addUserDebugParameter(
                paramName=self.name,
                rangeMin=self.val_min,
                rangeMax=self.val_max,
                startValue=value
            )
        else:
            raise Exception(
                f'Handler for parameter \'{self.name}\' is already used'
            )

    def remove(self):
        """Remove parameter"""
        pybullet.removeUserDebugItem(self._handler)

    def get_value(self):
        """Current value"""
        return pybullet.readUserDebugParameter(self._handler)

    def update(self):
        """Update"""
        previous = self.value
        self.value = self.get_value()
        self.changed = abs(self.value - previous) > 1e-2
        if self.changed:
            pylog.debug(
                '%s changed (%s -> %s)',
                self.name,
                previous,
                self.value,
            )


class ParameterPlay(DebugParameter):
    """Play/pause parameter"""

    def __init__(self, initial_value: bool = True):
        self.previous_value = initial_value
        super().__init__(
            name='Play/Pause',
            val=initial_value,
            val_min=0,
            val_max=-1,
        )

    def update(self):
        """Update"""
        value = self.get_value()
        if value != self.previous_value:
            self.value = not self.value
            self.previous_value = value


class UserParameters(dict):
    """Parameters control"""

    def __init__(self, options: SimulationOptions):
        super().__init__()
        self['play'] = ParameterPlay(initial_value=options.play)
        self['rtl'] = DebugParameter('Real-time limiter', options.rtl, 1e-3, 3)
        self['zoom'] = DebugParameter('Zoom', options.zoom, 0, 3)

    def init(self):
        """Intialise"""
        for parameter in self:
            self[parameter].init()

    def update(self):
        """Update parameters"""
        for parameter in self:
            self[parameter].update()

    def play(self):
        """Play"""
        return self['play']

    def rtl(self):
        """Real-time limiter"""
        return self['rtl']

    def zoom(self):
        """Camera zoom"""
        return self['zoom']


class Interfaces:
    """Interfaces (GUI, camera, video)"""

    def __init__(
            self,
            camera: UserCamera = None,
            user_params: UserParameters = None,
            video: CameraRecord = None,
            camera_skips: int = 1,
    ):
        super().__init__()
        self.camera = camera
        self.user_params = user_params
        self.video = video
        self.camera_skips = camera_skips

    def init_camera(self, target_identity, timestep, **kwargs):
        """Initialise camera"""
        self.camera = UserCamera(
            target_identity=target_identity,
            yaw_speed=(
                360/10*self.camera_skips
                if kwargs.pop('rotating_camera', False)
                else 0
            ),
            timestep=timestep,
            **kwargs,
        )

    def init_video(
            self,
            target_identity: int,
            simulation_options: SimulationOptions,
            **kwargs,
    ):
        """Initialise video recording"""
        self.video = CameraRecord(
            timestep=simulation_options.timestep,
            target_identity=target_identity,
            n_iterations=simulation_options.n_iterations,
            fps=simulation_options.fps,
            pitch=kwargs.pop('pitch', simulation_options.video_pitch),
            yaw=kwargs.pop('yaw', simulation_options.video_yaw),
            yaw_speed=(
                1000
                if kwargs.pop('rotating_camera', False)
                else 0
            ),
            motion_filter=kwargs.pop(
                'motion_filter',
                simulation_options.video_filter,
            ),
            distance=simulation_options.video_distance,
        )
        assert not kwargs, kwargs

    def init_debug(self, simulation_options: SimulationOptions):
        """Initialise debug"""
        # User parameters
        if self.user_params is None:
            self.user_params = UserParameters(simulation_options)
        self.user_params.init()
