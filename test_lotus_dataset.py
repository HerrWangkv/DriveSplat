import hydra


@hydra.main(
    version_base=None,
    config_path="datasets/NuScenes",
    config_name="sdaig",
)
def main(cfg):
    from datasets import build_dataset_from_cfg

    dataset = build_dataset_from_cfg(cfg.data.trainval)
    dataset.vis(0)
    breakpoint()


if __name__ == "__main__":
    main()
