import yaml


def get_config_info(config_path):
    with open(config_path, 'r') as f:
        config_info = yaml.safe_load(f)
    return config_info
