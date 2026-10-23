
import unittest


def cpu_to_cores(value):
    """Convert Kubernetes CPU quantities to CPU cores."""
    value = str(value)
    if value.endswith("m"):
        return float(value[:-1]) / 1000
    return float(value)


def memory_to_gb(value):
    """Convert Kubernetes memory quantities to decimal GB."""
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


class TestResourceConversion(unittest.TestCase):
    def test_cpu_millicores(self):
        self.assertAlmostEqual(cpu_to_cores("250m"), 0.25)

    def test_cpu_cores(self):
        self.assertAlmostEqual(cpu_to_cores("2"), 2.0)

    def test_memory_mib(self):
        self.assertAlmostEqual(memory_to_gb("512Mi"), 512 * 1024**2 / 1_000_000_000)

    def test_memory_gib(self):
        self.assertAlmostEqual(memory_to_gb("1Gi"), 1024**3 / 1_000_000_000)


if __name__ == "__main__":
    unittest.main()
