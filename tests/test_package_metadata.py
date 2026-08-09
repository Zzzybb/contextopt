from __future__ import annotations

import importlib.metadata
import unittest

from contextopt import __version__


class PackageMetadataTests(unittest.TestCase):
    def test_runtime_version_matches_distribution_metadata(self) -> None:
        self.assertEqual(__version__, importlib.metadata.version("contextopt"))
        self.assertEqual(__version__, "0.9.0a1")
