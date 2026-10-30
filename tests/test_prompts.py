import unittest

from sat2aerial.prompts import build_prompt, parse_variants


class PromptTest(unittest.TestCase):
    def test_reference_tokens(self):
        prompt = build_prompt(view="oblique30", style_refs=2, altitude_m=80)
        self.assertIn("<ref_image_0>", prompt)
        self.assertIn("<ref_image_1>", prompt)
        self.assertIn("<ref_image_2>", prompt)
        self.assertIn("80", prompt)
        self.assertIn("30 degrees", prompt)

    def test_no_style_tokens(self):
        prompt = build_prompt(view="nadir", style_refs=0)
        self.assertIn("<ref_image_0>", prompt)
        self.assertNotIn("<ref_image_1>", prompt)

    def test_variants(self):
        variants = parse_variants("nadir,oblique30,nadir_style,oblique45+style2")
        self.assertEqual(variants[0], ("nadir", "nadir", 0))
        self.assertEqual(variants[2], ("nadir_style", "nadir", 1))
        self.assertEqual(variants[3], ("oblique45+style2", "oblique45", 2))

    def test_bad_variant(self):
        with self.assertRaises(ValueError):
            parse_variants("nope")


if __name__ == "__main__":
    unittest.main()
