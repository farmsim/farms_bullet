"""Simulator"""

import time
import numpy as np
import pybullet
from farms_core import pylog
from farms_core.array.types import NDARRAY_3
from ..utils.output import redirect_output


def init_engine(headless: bool = False, opengl2: bool = False):
    """Initialise engine"""
    pylog.debug('Pybullet version: %s', pybullet.getAPIVersion())
    background_color = 0.9*np.ones(3)

    options = ''
    if not headless:
        options += (
            f'--background_color_red={background_color[0]}'
            f' --background_color_green={background_color[1]}'
            f' --background_color_blue={background_color[2]}'
        )
    elif opengl2:
        options += ' --opengl2'
    # options += ' --enable_experimental_opencl'
    kwargs_options = {'options': options} if options else {}
    with redirect_output(pylog.debug):
        pybullet.connect(
            pybullet.DIRECT if headless else pybullet.GUI,  # pybullet.DIRECT
            # options='--enable_experimental_opencl'
            # options='--opengl2'  #  --minGraphicsUpdateTimeMs=32000
            **kwargs_options,
        )
    # pybullet_path = pybullet_data.getDataPath()
    # pylog.debug('Adding pybullet data path {}'.format(pybullet_path))
    # pybullet.setAdditionalSearchPath(pybullet_path)


def real_time_handing(
        timestep: float,
        tic_rt: NDARRAY_3,
        rtl: float = 1.0,
        verbose: bool = False,
        **kwargs,
):
    """Real-time handling"""
    tic_rt[1] = time.time()
    tic_rt[2] += timestep/rtl - (tic_rt[1] - tic_rt[0])
    rtf = timestep / (tic_rt[1] - tic_rt[0])
    if tic_rt[2] > 1e-2:
        time.sleep(tic_rt[2])
        tic_rt[2] = 0
    elif tic_rt[2] < 0:
        tic_rt[2] = 0
    tic_rt[0] = time.time()
    if rtf < 0.1 and verbose:
        pylog.debug('Significantly slower than real-time: %s%%', 100*rtf)
        time_plugin = kwargs.pop('time_plugin', False)
        time_control = kwargs.pop('time_control', False)
        time_sim = kwargs.pop('time_sim', False)
        if time_plugin:
            pylog.debug('  Time in py_plugins: %s [ms]', time_plugin)
        if time_control:
            pylog.debug('    Time in control: %s [ms]', time_control)
        if time_sim:
            pylog.debug('  Time in simulation: %s [ms]', time_sim)
