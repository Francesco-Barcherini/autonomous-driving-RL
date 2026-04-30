import importlib.util
import os
import unittest

from src.config.settings import load_config
from src.environment.track import Track
from src.graphics.renderer import Renderer


HAS_PYGAME = importlib.util.find_spec("pygame") is not None
HAS_SHAPELY = importlib.util.find_spec("shapely") is not None


@unittest.skipUnless(HAS_PYGAME and HAS_SHAPELY, "pygame and shapely are required")
class RendererTests(unittest.TestCase):
    def test_headless_renderer_smoke(self) -> None:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        config = load_config("config.toml")
        renderer = Renderer(config, "smoke")
        track = Track(
            "smoke",
            80.0,
            [(20.0, 100.0), (300.0, 100.0)],
            [(20.0, 60.0), (20.0, 140.0)],
            [(300.0, 60.0), (300.0, 140.0)],
            [],
        )
        renderer.draw(track, lines=["smoke"])
        renderer.quit()


if __name__ == "__main__":
    unittest.main()
