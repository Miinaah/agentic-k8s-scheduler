
"""Utilities for converting Kubernetes resource quantities."""


def cpu_to_cores(value):
    """Convert a Kubernetes CPU quantity to CPU cores."""
    value = str(value)
    if value.endswith("m"):
        return float(value[:-1]) / 1000
    return float(value)


def memory_to_gb(value):
    """Convert a Kubernetes memory quantity to decimal GB."""
    value = str(value)

    units = {
        "Ki": 1024,
        "Mi": 1024**2,
        "Gi": 1024**3,
        "Ti": 1024**4,
        "K": 1000,
        "M": 1000**2,
        "G": 1000**3,
        "T": 1000**4,
    }

    for suffix in ("Ki", "Mi", "Gi", "Ti", "K", "M", "G", "T"):
        if value.endswith(suffix):
            return float(value[:-len(suffix)]) * units[suffix] / 1_000_000_000

    return float(value) / 1_000_000_000
