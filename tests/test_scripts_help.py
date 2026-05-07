import subprocess
import sys
import unittest


class ScriptHelpTests(unittest.TestCase):
    def test_root_scripts_expose_help(self) -> None:
        for script in [
            "draw_tracks.py",
            "train_som.py",
            "train_rl.py",
            "run_inference.py",
            "evaluate_models.py",
            "plot_analysis.py",
        ]:
            with self.subTest(script=script):
                result = subprocess.run(
                    [sys.executable, script, "--help"],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)


if __name__ == "__main__":
    unittest.main()
