"""Control"""

from ...controllers.control import AnimatController
from .network import SalamanderNetworkODE


class SalamanderController(AnimatController):
    """AnimatController"""

    @classmethod
    def from_data(cls, model, animat_data, timestep):
        """Salamander controller from options"""
        return cls(
            model=model,
            network=SalamanderNetworkODE(animat_data, timestep)
        )