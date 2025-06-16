import hydra

@hydra.main(
    version_base=None,
    config_path="configs/dataset",
    config_name="NuScenes",
)
def main(cfg):
    from datasets import build_dataset_from_cfg

    dataset = build_dataset_from_cfg(cfg.data.inference.nuscenes)
    dataset.vis(0)
    breakpoint()


if __name__ == "__main__":
    main()