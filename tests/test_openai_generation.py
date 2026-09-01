import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moprm.candidates.openai_generator import (  # noqa: E402
    OpenAICandidateConfig,
    build_generation_prompt,
    generate_openai_candidates,
)
from moprm.openai_responses import (  # noqa: E402
    extract_output_text,
    extract_usage,
    load_env_file,
)
from moprm.schema import ProblemRecord  # noqa: E402


class FakeResponsesClient:
    def __init__(self) -> None:
        self.inputs: list[str] = []

    def create_response(self, **kwargs):
        self.inputs.append(kwargs["input_text"])
        return {
            "id": "resp_test",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Reason briefly.\nFinal answer: 42",
                        }
                    ],
                }
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 6,
                "total_tokens": 16,
            },
        }


class OpenAIGenerationTest(unittest.TestCase):
    def test_load_env_file_without_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(
                "\n".join(
                    [
                        "# comment",
                        "OPENAI_API_KEY='sk-test'",
                        'OPENAI_MODEL="gpt-test"',
                        "export OTHER=value",
                    ]
                ),
                encoding="utf-8",
            )
            values = load_env_file(path)
        self.assertEqual(values["OPENAI_API_KEY"], "sk-test")
        self.assertEqual(values["OPENAI_MODEL"], "gpt-test")
        self.assertEqual(values["OTHER"], "value")

    def test_extract_response_text_and_usage(self) -> None:
        payload = {
            "output": [
                {
                    "content": [
                        {"type": "output_text", "text": "Part one."},
                        {"type": "output_text", "text": "Part two."},
                    ]
                }
            ],
            "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
        }
        self.assertEqual(extract_output_text(payload), "Part one.\nPart two.")
        self.assertEqual(extract_usage(payload)["total_tokens"], 3)

    def test_generation_prompt_does_not_include_gold_answer(self) -> None:
        record = ProblemRecord.from_dict(
            {
                "problem_id": "p1",
                "domain": "math",
                "problem": "What is 20 + 22?",
                "answer": "SECRET_GOLD",
                "candidates": [],
            }
        )
        prompt = build_generation_prompt(record, 0)
        self.assertIn("What is 20 + 22?", prompt)
        self.assertNotIn("SECRET_GOLD", prompt)

    def test_generate_openai_candidates_marks_no_gold_usage(self) -> None:
        record = ProblemRecord.from_dict(
            {
                "problem_id": "p1",
                "domain": "math",
                "problem": "What is 20 + 22?",
                "answer": "42",
                "candidates": [],
            }
        )
        generated = generate_openai_candidates(
            record,
            FakeResponsesClient(),
            OpenAICandidateConfig(model="gpt-test", num_candidates=2),
        )
        self.assertEqual(len(generated.candidates), 2)
        self.assertEqual(generated.candidates[0].final_answer, "42")
        self.assertFalse(generated.candidates[0].metadata["uses_gold_answer"])
        self.assertTrue(generated.candidates[0].metadata["has_explicit_final_answer"])
        self.assertEqual(generated.candidates[0].metadata["usage"]["total_tokens"], 16)


if __name__ == "__main__":
    unittest.main()
