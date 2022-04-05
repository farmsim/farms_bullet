"""Simulation model"""

import os
from typing import List, Dict

import numpy as np
from nptyping import NDArray
import pybullet

import farms_pylog as pylog
from farms_core.model.options import SpawnLoader
from ..utils.sdf import load_sdf, load_sdf_pybullet
from ..utils.output import redirect_output


class SimulationModel:
    """SimulationModel"""

    def __init__(self, identity: int = None):
        super().__init__()
        self._identity = identity
        self.joint_list = None
        self.controller = None

    def identity(self):
        """Model identity"""
        return self._identity

    def links_identities(self):
        """Joints"""
        return np.arange(-1, pybullet.getNumJoints(self._identity), dtype=int)

    def joints_identities(self):
        """Joints"""
        return np.arange(pybullet.getNumJoints(self._identity), dtype=int)

    def n_joints(self):
        """Get number of joints"""
        return pybullet.getNumJoints(self._identity)

    @staticmethod
    def get_parent_links_info(identity, base_link='base_link'):
        """Get links (parent of joint)"""
        links = {base_link: -1}
        links.update({
            info[12].decode('UTF-8'): info[16] + 1
            for info in [
                pybullet.getJointInfo(identity, j)
                for j in range(pybullet.getNumJoints(identity))
            ]
        })
        return links

    @staticmethod
    def get_joints_info(identity):
        """Get joints"""
        joints = {
            info[1].decode('UTF-8'): info[0]
            for info in [
                pybullet.getJointInfo(identity, j)
                for j in range(pybullet.getNumJoints(identity))
            ]
        }
        return joints

    def spawn(self):
        """Spawn"""

    def step(self):
        """Step"""

    def log(self):
        """Log"""

    def save_logs(self):
        """Save logs"""

    def plot(self):
        """Plot"""

    def reset(self):
        """Reset"""

    def delete(self):
        """Delete"""

    @staticmethod
    def from_sdf(sdf, **kwargs):
        """Model from SDF"""
        sdf = os.path.expandvars(sdf)
        assert os.path.isfile(sdf), f'{sdf} does not exist'
        spawn_loader = kwargs.pop('spawn_loader', SpawnLoader.FARMS)
        pylog.debug('Loading %s with %s', sdf, spawn_loader)
        if spawn_loader == SpawnLoader.PYBULLET:
            model = load_sdf_pybullet(sdf, **kwargs)[0]
        else:
            model = load_sdf(sdf, force_concave=True, **kwargs)[0]
        return model

    @staticmethod
    def from_urdf(urdf, **kwargs):
        """Model from SDF"""
        assert os.path.isfile(urdf), f'{urdf} does not exist'
        with redirect_output(pylog.warning):
            model = pybullet.loadURDF(urdf, **kwargs)
        return model


class GroundModel(SimulationModel):
    """Ground model"""

    def __init__(
            self,
            position: NDArray[(3,), float] = None,
            orientation: NDArray[(4,), float] = None,
    ):
        super().__init__()
        self.position = position
        self.orientation = orientation
        self.plane = None

    def spawn(self):
        """Spawn"""
        self.plane = pybullet.createCollisionShape(pybullet.GEOM_PLANE)
        options = {}
        if self.position is not None:
            options['basePosition'] = self.position
        if self.orientation is not None:
            options['baseOrientation'] = self.orientation
        self._identity = pybullet.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=self.plane,
            **options,
        )


class DescriptionFormatModel(SimulationModel):
    """DescriptionFormatModel"""

    def __init__(
            self,
            path: str,
            load_options: Dict = None,
            spawn_options: Dict = None,
            visual_options: Dict = None,
    ):
        super().__init__()
        self.path = path
        self.load_options = (
            load_options
            if load_options is not None
            else {}
        )
        self.spawn_options = (
            spawn_options
            if spawn_options is not None
            else {'posObj': [0, 0, 0], 'ornObj': [0, 0, 0, 1]}
        )
        self.visual_options = (
            visual_options
            if visual_options is not None
            else {}
        )

    def spawn(self):
        """Spawn"""
        extension = os.path.splitext(self.path)[1]
        if extension == '.sdf':
            self._identity = self.from_sdf(self.path, **self.load_options)
        elif extension == '.urdf':
            self._identity = self.from_urdf(self.path, **self.load_options)
        else:
            raise Exception(
                f'Unknown description format extension .{extension}'
            )

        # Spawn options
        if self.spawn_options:
            pos = pybullet.getBasePositionAndOrientation(
                bodyUniqueId=self._identity
            )[0]
            pos_obj = self.spawn_options.get('posObj')
            orn_obj = self.spawn_options.get('ornObj')
            pybullet.resetBasePositionAndOrientation(
                bodyUniqueId=self._identity,
                posObj=np.array(pos) + np.array(pos_obj),
                ornObj=np.array(orn_obj),
            )

        # Visual options
        _visual_options = self.visual_options.copy()
        if _visual_options:
            path = _visual_options.pop('path', None)
            if path is not None:
                texture = pybullet.loadTexture(
                    os.path.join(os.path.dirname(self.path), path)
                )
                _visual_options['textureUniqueId'] = texture
            rgba_color = _visual_options.pop('rgbaColor')
            specular_color = _visual_options.pop('specularColor')
            for info in pybullet.getVisualShapeData(self._identity):
                # for _ in range(pybullet.getNumJoints(self._identity)+1):
                pybullet.changeVisualShape(
                    objectUniqueId=info[0],
                    linkIndex=info[1],
                    shapeIndex=-1,
                    rgbaColor=rgba_color,
                    specularColor=specular_color,
                    **_visual_options,
                )


class SimulationModels(SimulationModel):
    """Simulation models"""

    def __init__(self, models: List[SimulationModel]):
        super().__init__()
        self._models = models

    def __iter__(self):
        return iter(self._models)

    def __getitem__(self, key):
        assert key < len(self._models)
        return self._models[key]

    def spawn(self):
        """Spawn"""
        for model in self:
            model.spawn()

    def step(self):
        """Step"""
        for model in self:
            model.step()

    def log(self):
        """Log"""
        for model in self:
            model.log()
