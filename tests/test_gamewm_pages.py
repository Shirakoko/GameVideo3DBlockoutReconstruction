"""Regression checks for the static GameWM reading-note page."""

from html.parser import HTMLParser
from pathlib import Path
import re
import unittest
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "gamewm" / "index.html"


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


class GameWMPagesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = PAGE.read_text(encoding="utf-8")
        cls.parser = PageParser()
        cls.parser.feed(cls.html)

    def test_all_eight_images_are_portable_and_present(self):
        images = [attrs["src"] for tag, attrs in self.parser.tags if tag == "img"]
        self.assertEqual(len(images), 8)
        self.assertEqual(len(set(images)), 8)
        for src in images:
            with self.subTest(src=src):
                self.assertTrue(src.startswith("assets/"))
                self.assertFalse(urlsplit(src).scheme)
                self.assertNotIn("\\", src)
                image = (PAGE.parent / src).resolve()
                self.assertTrue(image.is_relative_to(PAGE.parent.resolve()))
                self.assertTrue(image.is_file())
                self.assertGreater(image.stat().st_size, 0)
                self.assertRegex(image.name, r"^figure-\d{2}\.(png|jpeg)$")
        self.assertEqual(
            len(list((PAGE.parent / "assets").iterdir())),
            8,
            "Do not accidentally publish unrelated assets.",
        )

    def test_no_local_paths_or_unrelated_media(self):
        self.assertIsNone(re.search(r"\b[A-Za-z]:[\\/]|file://", self.html))
        self.assertFalse(
            {"video", "audio", "iframe", "script"}
            & {tag for tag, _ in self.parser.tags}
        )
        self.assertNotIn("fonts.googleapis.com", self.html)

    def test_language_and_mobile_viewport(self):
        html_attrs = next(attrs for tag, attrs in self.parser.tags if tag == "html")
        self.assertEqual(html_attrs["lang"], "zh-CN")
        viewport = next(
            attrs["content"]
            for tag, attrs in self.parser.tags
            if tag == "meta" and attrs.get("name") == "viewport"
        )
        self.assertEqual(viewport, "width=device-width, initial-scale=1")

    def test_navigation_and_entry_points(self):
        links = [attrs.get("href") for tag, attrs in self.parser.tags if tag == "a"]
        self.assertIn("../", links)
        self.assertIn("https://arxiv.org/abs/2611.19247", links)
        home = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn('href="gamewm/"', home)
        self.assertIn(
            "https://shirakoko.github.io/GameVideo3DBlockoutReconstruction/gamewm/",
            readme,
        )


if __name__ == "__main__":
    unittest.main()
