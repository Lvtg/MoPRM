from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moprm.answer_checking import check_answer, normalize_answer, strip_boxed


class AnswerCheckingTest(unittest.TestCase):
    def test_boxed_answer(self) -> None:
        self.assertEqual(strip_boxed(r"The value is \boxed{42}."), "42")
        self.assertTrue(check_answer(r"\boxed{3/4}", "0.75"))

    def test_choice_answer(self) -> None:
        self.assertTrue(check_answer("Final answer: C", "C", domain="logic"))
        self.assertFalse(check_answer("Final answer: B", "C", domain="logic"))

    def test_normalize_answer(self) -> None:
        self.assertEqual(normalize_answer(" The answer is 1,000. "), "1000")


if __name__ == "__main__":
    unittest.main()
