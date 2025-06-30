from omegaconf import OmegaConf
from data import build_dataset_from_cfg

def main():
    dataset_cfg = OmegaConf.load("data/NuScenes/sdaig.yaml")

    dataset = build_dataset_from_cfg(dataset_cfg.data.inference)
    dataset.vis(100)
    breakpoint()


if __name__ == "__main__":
    main()
