from hydra import compose, initialize_config_dir
from hydra.utils import to_absolute_path


def load_hydra_conf(config_dir="conf", config_name="config"):
    abs_config_dir = to_absolute_path(config_dir)
    with initialize_config_dir(
        config_dir=abs_config_dir, job_name="supersegment", version_base=None
    ):
        cfg = compose(config_name=config_name)
    return cfg
