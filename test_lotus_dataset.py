import hydra


@hydra.main(
    version_base=None,
    config_path="data/NuScenes",
    config_name="sdaig",
)
def main(cfg):
    from data import build_dataset_from_cfg

    dataset = build_dataset_from_cfg(cfg.data.inference)
    dataset.vis(0)
    breakpoint()


if __name__ == "__main__":
    main()
