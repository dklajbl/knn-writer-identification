from src.patchers.patcher_config import PatcherConfig
from src.patchers.base_patcher import BasePatcher
from src.patchers.grid_patcher import GridPatcher
from src.patchers.random_patcher import RandomPatcher
from src.patchers.sift_patcher import SIFTPatcher


def make_patcher(config: PatcherConfig) -> BasePatcher:

    """
    Factory function that creates a patcher instance from configuration.

    Parameters:
        config (PatcherConfig): configuration object describing which patching strategy to use.

    Returns:
        - GridPatcher instance: an initialized grid patcher object
        - RandomPatcher instance: an initialized random patcher object
        - ... (other instances of different patching methods)

    Raises:
        NotImplementedError: if the requested patching method exists conceptually but has not been implemented yet
        ValueError: if the method patcher name is unknown
    """

    if config.method == "grid":
        return GridPatcher(config)

    if config.method == "random":
        return RandomPatcher(config)

    if config.method == "sift":
        return SIFTPatcher(config)

    raise ValueError(f"Unknown patch method: {config.method}")
