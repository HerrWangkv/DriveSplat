# data/__init__.py
from data.NuScenes.NuScenes import (
    SDaIGNuScenesTrainDataset,
    SDaIGNuScenesFinetuneDataset,
    SDaIGNuScenesTestDataset,
)


def build_dataset_from_cfg(cfg):
    """Build a dataset from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".

    Returns:
        object: The constructed object.
    """
    args = dict(cfg)
    dataset_type = args.pop("type")
    if dataset_type == "SDaIGNuScenesTrainDataset":
        return SDaIGNuScenesTrainDataset(**args)
    elif dataset_type == "SDaIGNuScenesFinetuneDataset":
        return SDaIGNuScenesFinetuneDataset(**args)
    elif dataset_type == "SDaIGNuScenesTestDataset":
        return SDaIGNuScenesTestDataset(**args)
    else:
        raise KeyError(f"{dataset_type} is not a valid dataset type")
