from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moprm.answer_checking import (
    check_answer,
    extract_final_answer,
    has_explicit_final_answer,
    normalize_answer,
    strip_boxed,
)


class AnswerCheckingTest(unittest.TestCase):
    def test_boxed_answer(self) -> None:
        self.assertEqual(strip_boxed(r"The value is \boxed{42}."), "42")
        self.assertTrue(check_answer(r"\boxed{3/4}", "0.75"))

    def test_choice_answer(self) -> None:
        self.assertTrue(check_answer("Final answer: C", "C", domain="logic"))
        self.assertFalse(check_answer("Final answer: B", "C", domain="logic"))

    def test_extracts_last_final_answer_line(self) -> None:
        text = (
            "The answer is (E) because the green book is second.\n\n"
            "Final answer: E"
        )
        self.assertEqual(extract_final_answer(text), "E")

    def test_detects_missing_explicit_final_answer(self) -> None:
        self.assertTrue(has_explicit_final_answer("Work.\nFinal answer: 42"))
        self.assertTrue(has_explicit_final_answer(r"Work. \boxed{42}"))
        self.assertFalse(has_explicit_final_answer("Work trails off without a final line"))

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

    def test_inline_math_delimiters(self) -> None:
        self.assertTrue(check_answer(r"\(-125\)", "-125"))
        self.assertTrue(check_answer(r"\(\frac{2}{3}\)", r"\frac{2}{3}"))
        self.assertTrue(check_answer(r"\((1, -16, -4, 43)\)", "(1,-16,-4,43)"))

    def test_numeric_unit_suffix(self) -> None:
        self.assertTrue(check_answer("36 seconds", "36"))
        self.assertFalse(check_answer("37 seconds", "36"))
        self.assertFalse(check_answer("36%", "36"))


if __name__ == "__main__":
    unittest.main()
