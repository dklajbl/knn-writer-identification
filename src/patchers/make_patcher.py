from src.patchers.grid_patcher import GridPatcher
from src.patchers.patcher_config import PatcherConfig


def make_patcher(config: PatcherConfig) -> GridPatcher:

    """
    Factory function that creates a patcher instance from configuration.

    Parameters:
        config (PatcherConfig): configuration object describing which patching strategy to use.

    Returns:
        - GridPatcher instance: an initialized grid patcher object
        - ... (other instances of different patching methods)

    Raises:
        NotImplementedError: if the requested patching method exists conceptually but has not been implemented yet
        ValueError: if the method patcher name is unknown
    """

    if config.method == "grid":
        return GridPatcher(config)

    if config.method == "random":
        raise NotImplementedError("Random patcher not implemented yet.")

    if config.method == "algorithmic":
        raise NotImplementedError("Algorithmic patcher not implemented yet.")

    raise ValueError(f"Unknown patch method: {config.method}")
