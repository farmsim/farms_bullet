"""Control"""

import pybullet
import numpy as np

from farms_core.model.control import ControlType
from farms_core.units import SimulationUnitScaling
from .model import SimulationModels


def reset_controllers(identity: int):
    """Reset controllers"""
    n_joints = pybullet.getNumJoints(identity)
    joints = np.arange(n_joints)
    zeros = np.zeros_like(joints)
    pybullet.setJointMotorControlArray(
        identity,
        joints,
        pybullet.POSITION_CONTROL,
        targetPositions=zeros,
        targetVelocities=zeros,
        forces=zeros
    )
    pybullet.setJointMotorControlArray(
        identity,
        joints,
        pybullet.VELOCITY_CONTROL,
        targetVelocities=zeros,
        forces=zeros,
    )
    pybullet.setJointMotorControlArray(
        identity,
        joints,
        pybullet.TORQUE_CONTROL,
        forces=zeros
    )


def control_models(
        iteration: int,
        time: float,
        timestep: float,
        models: SimulationModels,
        units: SimulationUnitScaling,
):
    """Control"""
    torques = units.torques
    iseconds = 1/units.seconds
    for model in models:
        if model.controller is None:
            continue
        controller = model.controller
        joints_map = model.joints_map
        if controller.joints_names[ControlType.POSITION]:
            kwargs = {}
            if controller.position_args is not None:
                kwargs['positionGains'] = controller.position_args[0]
                kwargs['velocityGains'] = controller.position_args[1]
                kwargs['targetVelocities'] = controller.position_args[2]
            joints_positions = controller.positions(iteration, time, timestep)
            pybullet.setJointMotorControlArray(
                bodyUniqueId=model.identity(),
                jointIndices=[
                    joints_map[joint]
                    for joint in controller.joints_names[ControlType.POSITION]
                ],
                controlMode=pybullet.POSITION_CONTROL,
                targetPositions=[
                    joints_positions[joint]
                    for joint in controller.joints_names[ControlType.POSITION]
                ],
                forces=controller.max_torques[ControlType.POSITION]*torques,
                **kwargs,
            )
        if controller.joints_names[ControlType.VELOCITY]:
            kwargs = {}
            if controller.velocity_args is not None:
                kwargs['positionGains'] = controller.velocity_args[0]
                kwargs['velocityGains'] = controller.velocity_args[1]
            joints_velocities = controller.velocities(iteration, time, timestep)
            pybullet.setJointMotorControlArray(
                bodyUniqueId=model.identity(),
                jointIndices=[
                    joints_map[joint]
                    for joint in controller.joints_names[ControlType.VELOCITY]
                ],
                controlMode=pybullet.VELOCITY_CONTROL,
                targetVelocities=[
                    joints_velocities[joint]*iseconds
                    for joint in controller.joints_names[ControlType.VELOCITY]
                ],
                forces=controller.max_torques[ControlType.VELOCITY]*torques,
                **kwargs,
            )
        if controller.joints_names[ControlType.TORQUE]:
            joints_torques = controller.torques(iteration, time, timestep)
            pybullet.setJointMotorControlArray(
                bodyUniqueId=model.identity(),
                jointIndices=[
                    joints_map[joint]
                    for joint in controller.joints_names[ControlType.TORQUE]
                ],
                controlMode=pybullet.TORQUE_CONTROL,
                forces=[
                    joints_torques[joint]*torques
                    for joint in controller.joints_names[ControlType.TORQUE]
                ],
            )
