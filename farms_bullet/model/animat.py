"""Animat"""

import os
from typing import List

import numpy as np
import pybullet

import farms_pylog as pylog
from farms_data.units import SimulationUnitScaling
from farms_data.amphibious.animat_data import ModelData
from farms_data.model.options import (
    SpawnLoader,
    ModelOptions,
    SpawnOptions,
    JointOptions,
)

from ..utils.sdf import load_sdf, load_sdf_pybullet
from ..sensors.sensors import (
    Sensors,
    LinksStatesSensor,
    JointsStatesSensor,
    ContactsSensors,
)

from .control import reset_controllers
from .model import SimulationModel


def joint_type_str(joint_type: int):
    """Return joint type as str"""
    return (
        'Revolute' if joint_type == pybullet.JOINT_REVOLUTE
        else 'Prismatic' if joint_type == pybullet.JOINT_PRISMATIC
        else 'Spherical' if joint_type == pybullet.JOINT_SPHERICAL
        else 'Planar' if joint_type == pybullet.JOINT_PLANAR
        else 'Fixed' if joint_type == pybullet.JOINT_FIXED
        else 'Unknown'
    )


def initial_pose(
        identity: int,
        joints: List[int],
        joints_options: JointOptions,
        spawn_options: SpawnOptions,
        units: SimulationUnitScaling,
):
    """Initial pose"""
    spawn_orientation = pybullet.getQuaternionFromEuler(
        spawn_options.orientation
    )
    com_pos, com_ori = pybullet.getDynamicsInfo(identity, -1)[3:5]
    pos_offset = np.array(pybullet.multiplyTransforms(
        [0, 0, 0],
        spawn_orientation,
        com_pos,
        com_ori,
    )[0])
    pybullet.resetBasePositionAndOrientation(
        identity,
        np.array(spawn_options.position)*units.meters+pos_offset,
        pybullet.multiplyTransforms(
            [0, 0, 0], spawn_orientation,
            [0, 0, 0], com_ori,
        )[1],
    )
    pybullet.resetBaseVelocity(
        objectUniqueId=identity,
        linearVelocity=np.array(spawn_options.velocity_lin)*units.velocity,
        angularVelocity=np.array(spawn_options.velocity_ang)/units.seconds
    )
    for joint, info in zip(joints, joints_options):
        pybullet.resetJointState(
            bodyUniqueId=identity,
            jointIndex=joint,
            targetValue=info.initial_position,
            targetVelocity=info.initial_velocity/units.seconds,
        )


class Animat(SimulationModel):
    """Animat"""

    def __init__(
            self,
            identity: int = None,
            options: ModelOptions = None,
            data: ModelData = None,
            units: SimulationUnitScaling = None,
    ):
        super().__init__(identity=identity)
        self.options = options
        self.links_map = {}
        self.joints_map = {}
        self.masses = {}
        self.sensors = Sensors()
        self.data = data
        self.units = units

    def links_identities(self):
        """Links"""
        return [
            self.links_map[link]
            for link in self.options.morphology.links_names()
        ]

    def joints_identities(self):
        """Joints"""
        names = self.options.morphology.joints_names()
        for joint in names:
            assert joint in self.joints_map, (
                f'Joint {joint} not in {self.joints_map.keys()}'
            )
        return [self.joints_map[joint] for joint in names]

    def spawn(self):
        """Spawn amphibious"""
        # Spawn
        use_pybullet_loader = self.options.spawn.loader == SpawnLoader.PYBULLET
        self.spawn_sdf(original=use_pybullet_loader)

        # Sensors
        if self.data:
            self.add_sensors()

        # Body properties
        self.set_body_properties()

        # Reset controllers
        reset_controllers(self._identity)

    def spawn_sdf(self, verbose: bool = True, original: bool = False):
        """Spawn sdf"""
        sdf_path = self.options.sdf_path
        if verbose:
            pylog.debug(
                'Spawning %s using %s',
                sdf_path,
                'Pybullet' if original else 'FARMS',
            )
        sdf_path = os.path.expandvars(sdf_path)
        assert os.path.isfile(sdf_path), f'{sdf_path} is not a file'
        if original:
            self._identity, self.links_map, self.joints_map = load_sdf_pybullet(
                sdf_path=sdf_path,
                morphology_links=self.options.morphology.links_names(),
                units=self.units,
            )
        else:
            self._identity, self.links_map, self.joints_map = load_sdf(
                sdf_path=sdf_path,
                force_concave=False,
                reset_control=True,
                verbose=verbose,
                links_options=self.options.morphology.links,
                use_self_collision=self.options.morphology.self_collisions,
                units=self.units,
            )
        if verbose:
            pylog.debug(
                '%s\n\n%s\n%s',
                f'Spawned model (Identity={self._identity})',
                'Model properties from pybullet (scaled units):',
                '\n'.join([
                    (
                        f'- {link_name+":": <20}:'
                        ' - Mass: {:.3e}'
                        ' - Inertias: {}'
                        ' - COM: {}'
                        ' - Orientation: {}'
                    ).format(
                        *[
                            float(value)
                            if val_i == 0  # Mass
                            else str([f'{val:.3e}' for val in value])
                            if val_i == 1  # Inertias
                            else str([f'{val:+.3e}' for val in value])
                            for val_i, value in enumerate(np.array(
                                    pybullet.getDynamicsInfo(
                                        self._identity,
                                        link,
                                    ),
                                    dtype=object,
                            )[[0, 2, 3, 4]])
                        ]
                    )
                    for link_name, link in self.links_map.items()
                ]),
            )
        initial_pose(
            identity=self._identity,
            joints=self.joints_identities(),
            joints_options=self.options.morphology.joints,
            spawn_options=self.options.spawn,
            units=self.units,
        )
        if verbose:
            self.print_information()

    def add_sensors(self):
        """Add sensors"""
        # Links
        if self.options.control.sensors.links:
            for link in self.options.control.sensors.links:
                assert link in self.links_map, (
                    f'link {link} not in {self.links_map}'
                )
            self.sensors.add({
                'links': LinksStatesSensor(
                    array=self.data.sensors.links.array,
                    model_id=self.identity(),
                    links=[
                        self.links_map[link]
                        for link in self.options.control.sensors.links
                    ],
                    units=self.units
                )
            })

        # Joints
        if self.options.control.sensors.joints:
            self.sensors.add({
                'joints': JointsStatesSensor(
                    array=self.data.sensors.joints.array,
                    model_id=self._identity,
                    joints=[
                        self.joints_map[joint]
                        for joint in self.options.control.sensors.joints
                    ],
                    units=self.units,
                    enable_ft=True
                )
            })

        # Contacts
        if self.options.control.sensors.contacts:
            self.sensors.add({
                'contacts': ContactsSensors(
                    array=self.data.sensors.contacts.array,
                    model_ids=[
                        self._identity
                        for _ in self.options.control.sensors.contacts
                    ],
                    model_links=[
                        self.links_map[foot]
                        for foot in self.options.control.sensors.contacts
                    ],
                    meters=self.units.meters,
                    newtons=self.units.newtons,
                )
            })

    def set_body_properties(self, verbose: bool = False):
        """Set body properties"""
        # Masses
        for link in self.options.morphology.links:
            self.masses[link.name] = pybullet.getDynamicsInfo(
                self.identity(),
                self.links_map[link.name],
            )[0]
        if self.data is not None:
            links = self.data.sensors.links
            links.masses = [0 for _ in links.names]
            for link in self.options.morphology.links:
                if link.name in links.names:
                    index = links.names.index(link.name)
                    links.masses[index] = self.masses[link.name]
        if verbose:
            pylog.debug('Body mass: %s [kg]', np.sum(self.masses.values()))

        # Deactivate collisions
        self.set_collisions(
            [
                link.name
                for link in self.options.morphology.links
                if not link.collisions
            ],
            group=0,
            mask=0,
        )
        if self.options.morphology.self_collisions:
            # Remove self collisions
            pairs = [
                f'{link0},{link1}'
                for link0, link1 in self.options.morphology.self_collisions
            ] + [
                f'{link1},{link0}'
                for link0, link1 in self.options.morphology.self_collisions
            ]
            for link0 in self.options.morphology.links:
                for link1 in self.options.morphology.links:
                    if (
                            f'{link0.name},{link1.name}' not in pairs
                            or f'{link1.name},{link0.name}' not in pairs
                    ):
                        pybullet.setCollisionFilterPair(
                            bodyUniqueIdA=self.identity(),
                            bodyUniqueIdB=self.identity(),
                            linkIndexA=self.links_map[link0.name],
                            linkIndexB=self.links_map[link1.name],
                            enableCollision=0,
                        )

        # Default dynamics
        for link in self.links_map:

            # Default friction
            self.set_link_dynamics(
                link,
                lateralFriction=0,
                spinningFriction=0,
                rollingFriction=0,
            )

            # Default damping
            self.set_link_dynamics(
                link,
                linearDamping=0,
                angularDamping=0,
                jointDamping=0,
            )

        # Model options dynamics
        pylog.debug(
            'Setting link dynamic properties:\n  - %s',
            '\n  - '.join([
                f'{link.name+":":<20} {link.pybullet_dynamics}'
                for link in self.options.morphology.links
            ])
        )
        for link in self.options.morphology.links:
            self.set_link_dynamics(
                link.name,
                **link.pybullet_dynamics,
            )
        pylog.debug(
            'Setting joint dynamic properties:\n  - %s',
            '\n  - '.join([
                f'{joint.name+":":<20} {joint.pybullet_dynamics}'
                for joint in self.options.morphology.joints
            ])
        )
        for joint in self.options.morphology.joints:
            self.set_joint_dynamics(
                joint.name,
                **joint.pybullet_dynamics,
            )

    def print_information(self):
        """Print information"""
        pylog.debug(
            'Links ids:\n%s',
            '\n'.join([
                f'  {name}: {identity}'
                for name, identity in self.links_map.items()
            ])
        )
        pylog.debug(
            'Joints ids:\n%s',
            '\n'.join([
                f'  {name}: {identity} (type: {joint_type})'
                for name, identity in self.joints_map.items()
                for joint_type in [joint_type_str(pybullet.getJointInfo(
                        self.identity(),
                        identity
                )[2])]
            ])
        )

    def print_dynamics_info(self, links: List = None):
        """Print dynamics info"""
        links = links if links is not None else self.links_map
        pylog.debug('Dynamics:')
        for link in links:
            dynamics_msg = (
                '\n      mass: {}'
                '\n      lateral_friction: {}'
                '\n      local inertia diagonal: {}'
                '\n      local inertial pos: {}'
                '\n      local inertial orn: {}'
                '\n      restitution: {}'
                '\n      rolling friction: {}'
                '\n      spinning friction: {}'
                '\n      contact damping: {}'
                '\n      contact stiffness: {}'
            )
            pylog.debug(
                '  - %s:%s',
                link,
                dynamics_msg.format(*pybullet.getDynamicsInfo(
                    self.identity(),
                    self.links_map[link]
                ))
            )
        pylog.debug('Model mass: %s [kg]', self.total_mass())

    def total_mass(self):
        """Print dynamics"""
        return np.sum([
            pybullet.getDynamicsInfo(self.identity(), link)[0]
            for link in self.links_map.values()
        ])

    def set_collisions(self, links: List, group: int = 0, mask: int = 0):
        """Activate/Deactivate leg collisions"""
        for link in links:
            pybullet.setCollisionFilterGroupMask(
                bodyUniqueId=self._identity,
                linkIndexA=self.links_map[link],
                collisionFilterGroup=group,
                collisionFilterMask=mask
            )

    def set_link_dynamics(self, link: str, **kwargs):
        """Set link dynamic properties"""
        for key, value in kwargs.items():
            dynamics_kwargs = (
                {
                    'contactStiffness': kwargs['contactStiffness'],
                    'contactDamping': kwargs['contactDamping'],
                }
                if key in ('contactStiffness', 'contactDamping')
                else {key: value}
            )
            pybullet.changeDynamics(
                bodyUniqueId=self.identity(),
                linkIndex=self.links_map[link],
                **dynamics_kwargs
            )

    def set_joint_dynamics(self, joint: str, **kwargs):
        """Set joint dynamic properties"""
        for key, value in kwargs.items():
            dynamics_kwargs = (
                {
                    'jointLowerLimit': kwargs['jointLowerLimit'],
                    'jointUpperLimit': kwargs['jointUpperLimit'],
                }
                if key in ('jointLowerLimit', 'jointUpperLimit')
                else {key: value}
            )
            pybullet.changeDynamics(
                bodyUniqueId=self.identity(),
                linkIndex=self.joints_map[joint],
                **dynamics_kwargs
            )
