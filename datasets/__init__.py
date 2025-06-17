# datasets/__init__.py
from datasets.NuScenes.NuScenes import SDaIGNuScenesDataset


def build_dataset_from_cfg(cfg):
    """Build a dataset from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".

    Returns:
        object: The constructed object.
    """
    args = dict(cfg)
    dataset_type = args.pop("type")
    if dataset_type == "SDaIGNuScenesDataset":
        return SDaIGNuScenesDataset(**args)
    else:
        raise KeyError(f"{dataset_type} is not a valid dataset type")
