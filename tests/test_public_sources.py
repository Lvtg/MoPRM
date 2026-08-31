import unittest

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moprm.datasets.public_sources import extract_gsm8k_answer, normalize_choice


class PublicSourcesTest(unittest.TestCase):
    def test_extract_gsm8k_answer(self) -> None:
        self.assertEqual(extract_gsm8k_answer("work\n#### 1,234"), "1,234")

    def test_normalize_choice(self) -> None:
        self.assertEqual(normalize_choice("(C)"), "C")
        self.assertEqual(normalize_choice("answer is b"), "B")


if __name__ == "__main__":
    unittest.main()

