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

    def test_latex_fraction(self) -> None:
        self.assertTrue(check_answer(r"\frac{14}{3}", "14/3"))
        self.assertTrue(check_answer(r"\boxed{\frac{3}{56}}", "3/56"))

    def test_latex_text_and_degree(self) -> None:
        self.assertTrue(check_answer(r"\text{Evelyn}", "Evelyn"))
        self.assertTrue(check_answer(r"90^\circ", "90"))

    def test_latex_coordinate_normalization(self) -> None:
        self.assertEqual(
            normalize_answer(r"\left( 3, \frac{\pi}{2} \right)"),
            "(3,pi/2)",
        )


if __name__ == "__main__":
    unittest.main()
