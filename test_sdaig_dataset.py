from omegaconf import OmegaConf
from data import build_dataset_from_cfg

def main():
    dataset_cfg = OmegaConf.load("data/NuScenes/sdaig.yaml")

    dataset = build_dataset_from_cfg(dataset_cfg.data.val)
    dataset.vis(4200)
    breakpoint()


if __name__ == "__main__":
    main()
