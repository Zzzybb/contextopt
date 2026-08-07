from __future__ import annotations

import unittest

from settings import merge_settings


class MergeSettingsTests(unittest.TestCase):
    def test_false_override_is_explicit(self) -> None:
        self.assertEqual(
            merge_settings({"enabled": True}, {"enabled": False}),
            {"enabled": False},
        )

    def test_none_inherits_default(self) -> None:
        self.assertEqual(
            merge_settings({"region": "cn"}, {"region": None}),
            {"region": "cn"},
        )


if __name__ == "__main__":
    unittest.main()
