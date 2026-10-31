"""Shared YAML loading helper.

Phonopy YAML files are large (the user's mesh.yaml is 41 MB). The C-based
loader is dramatically faster than the pure-Python fallback.
"""

import warnings

import yaml


def get_fast_loader():
    """Return the fastest available SafeLoader.

    Prefers libyaml's C-based CSafeLoader. Falls back to pure-Python SafeLoader
    with a warning, since this can be 10-20× slower on multi-MB files.
    """
    try:
        from yaml import CSafeLoader
        return CSafeLoader
    except ImportError:
        warnings.warn(
            "yaml.CSafeLoader unavailable; falling back to pure-Python SafeLoader. "
            "Parsing large phonopy YAML files will be slow. Install libyaml + "
            "rebuild pyyaml with C extensions for ~10× speedup.",
            stacklevel=2,
        )
        return yaml.SafeLoader


def load_yaml(filepath: str) -> dict:
    """Load a YAML file using the fastest available loader."""
    Loader = get_fast_loader()
    with open(filepath, "r") as f:
        return yaml.load(f, Loader=Loader)
