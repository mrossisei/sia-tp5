import yaml


def load_yaml(path):
    """Carga un YAML y devuelve un dict. Nombre canónico (convención TP4)."""
    with open(path) as f:
        return yaml.safe_load(f)


# Alias de compatibilidad con TP3 (que usaba `load_config`).
load_config = load_yaml
