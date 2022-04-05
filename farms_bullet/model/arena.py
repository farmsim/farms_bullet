"""Arena"""

from scipy.spatial.transform import Rotation
from farms_core.model.options import ArenaOptions
from farms_core.simulation.options import SimulationOptions
from farms_bullet.model.model import SimulationModels, DescriptionFormatModel


def get_arena(
        arena_options: ArenaOptions,
        simulation_options: SimulationOptions,
) -> SimulationModels:
    """Get arena from options"""

    # Options
    meters = simulation_options.units.meters
    orientation = Rotation.from_euler(
        seq='xyz',
        angles=arena_options.orientation,
        degrees=False,
    ).as_quat()

    # Main arena
    arena = DescriptionFormatModel(
        path=arena_options.sdf,
        spawn_options={
            'posObj': [pos*meters for pos in arena_options.position],
            'ornObj': orientation,
        },
        load_options={'units': simulation_options.units},
    )

    # Ground
    if arena_options.ground_height is not None:
        arena.spawn_options['posObj'][2] += (
            arena_options.ground_height*meters
        )

    return arena
