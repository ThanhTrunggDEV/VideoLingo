import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core import _1_ytdlp

MIRROR = "upos-sz-mirrorcos.bilivideo.com"


class BilibiliCdnFallbackTests(unittest.TestCase):
    def _run(self, url, failures=1):
        calls = []

        class FakeYDL:
            def __init__(self, opts):
                self.opts = opts

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def process_info(self, info_dict):
                return True

            def download(self, urls):
                calls.append(self)
                if len(calls) <= failures:
                    raise RuntimeError("Got error: 36 bytes read, 10247976 more expected")

        with tempfile.TemporaryDirectory() as temp_dir, \
                patch.object(_1_ytdlp, "load_key", return_value={}), \
                patch.object(_1_ytdlp, "update_ytdlp", return_value=FakeYDL), \
                patch.object(_1_ytdlp, "find_video_files", return_value="synthetic.mp4"), \
                patch.object(_1_ytdlp, "write_input_manifest", return_value=None):
            _1_ytdlp.download_video_ytdlp(url, temp_dir)
        return calls, FakeYDL

    def test_stalled_round_switches_to_a_mirror_and_rewrites_the_format_urls(self):
        calls, FakeYDL = self._run("https://www.bilibili.com/video/BV1xx411c7mD/")

        self.assertEqual(len(calls), 2)
        self.assertIs(type(calls[0]), FakeYDL)
        self.assertTrue(issubclass(type(calls[1]), FakeYDL))
        self.assertIsNot(type(calls[1]), FakeYDL)
        info = {"requested_formats": [
            {"url": "https://upos-hz-mirrorakam.akamaized.net/upgcxcode/1/2/3.m4s?e=x"},
            {"url": "https://upos-hz-mirrorakam.akamaized.net/upgcxcode/4/5/6.m4s?e=y"},
        ]}
        self.assertTrue(calls[1].process_info(info))
        self.assertEqual(info["requested_formats"][0]["url"],
                         f"https://{MIRROR}/upgcxcode/1/2/3.m4s?e=x")
        self.assertEqual(info["requested_formats"][1]["url"],
                         f"https://{MIRROR}/upgcxcode/4/5/6.m4s?e=y")

    def test_single_format_url_is_rewritten_too_and_other_urls_are_kept(self):
        calls, _ = self._run("https://www.bilibili.com/video/BV1xx411c7mD/")

        info = {"url": "https://upos-hz-mirrorakam.akamaized.net/upgcxcode/1/2/3.m4s?e=x"}
        calls[1].process_info(info)
        self.assertEqual(info["url"], f"https://{MIRROR}/upgcxcode/1/2/3.m4s?e=x")

        untouched = {"url": "https://i0.hdslb.com/bfs/archive/cover.jpg"}
        calls[1].process_info(untouched)
        self.assertEqual(untouched["url"], "https://i0.hdslb.com/bfs/archive/cover.jpg")

    def test_non_bilibili_urls_never_switch_hosts(self):
        calls, FakeYDL = self._run("https://video.example.com/item", failures=0)

        self.assertEqual(len(calls), 1)
        self.assertIs(type(calls[0]), FakeYDL)

    def test_giving_up_after_a_full_sweep_of_stalled_cdns(self):
        with patch("time.sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "Bilibili stopped responding"):
                self._run("https://www.bilibili.com/video/BV1xx411c7mD/", failures=8)
        sleep.assert_called_once_with(120)


class LongFilenameTests(unittest.TestCase):
    TITLE = ("前辈，求你别装了！一座清幽小院，一场看似寻常的隐居生活。景浦只想安安静静种田、饮茶、"
             "做些凡人日子里的小事，却不知自己随手留下的一物一言，在修仙者眼中皆藏着难以揣")

    def test_long_titles_stay_within_the_255_byte_filename_limit(self):
        from yt_dlp import YoutubeDL

        calls, _ = BilibiliCdnFallbackTests()._run(
            "https://www.bilibili.com/video/BV1xx411c7mD/", failures=0)
        opts = calls[0].opts
        for title in (self.TITLE, "😀" * 80):
            with self.subTest(first_char=title[0]):
                with YoutubeDL({**opts, "outtmpl": "output/%(title)s.%(ext)s", "quiet": True},
                               auto_init=False) as ydl:
                    name = ydl.prepare_filename(
                        {"title": title, "ext": "mp4", "format_id": "30066", "id": "BV1"})
                # yt-dlp writes the video stream as <name>.f<id>.mp4.part first:
                # the suffix that overflowed the 255-byte limit before trimming.
                part = name[:-len(".mp4")] + ".f30066.mp4.part"
                self.assertLessEqual(len(part.encode()), 255)
                self.assertIn(title[0], name)


class FindVideoFilesTests(unittest.TestCase):
    @patch.object(_1_ytdlp, "load_key", return_value=["mp4"])
    def test_keeps_output_prefixed_inputs_and_ignores_generated_videos(self, _):
        for input_name in ("output.mp4", "output tutorial.mp4", "OUTPUT.MP4", "output_sub tutorial.mp4"):
            with self.subTest(input_name=input_name), tempfile.TemporaryDirectory() as temp_dir:
                previous_cwd = os.getcwd()
                try:
                    os.chdir(temp_dir)
                    output_dir = Path("output")
                    output_dir.mkdir()
                    for filename in (input_name, "output_sub.mp4", "output_dub.mp4"):
                        (output_dir / filename).touch()

                    result = _1_ytdlp.find_video_files()

                    self.assertEqual(Path(result).name, input_name)
                finally:
                    os.chdir(previous_cwd)

    @patch.object(_1_ytdlp, "load_key", return_value=["mp4"])
    def test_absolute_directory_and_case_insensitive_generated_names(self, _):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "custom output"
            output.mkdir()
            for filename in ("output tutorial.MP4", "OUTPUT_SUB.MP4", "Output_Dub.mp4"):
                (output / filename).touch()
            (output / "folder.mp4").mkdir()
            self.assertEqual(Path(_1_ytdlp.find_video_files(str(output))).name, "output tutorial.MP4")

    @patch.object(_1_ytdlp, "load_key", return_value=["mp4"])
    def test_multiple_real_inputs_remain_ambiguous(self, _):
        with tempfile.TemporaryDirectory() as temp_dir:
            for filename in ("output.mp4", "other.mp4"):
                (Path(temp_dir) / filename).touch()
            with self.assertRaisesRegex(ValueError, "found 2"):
                _1_ytdlp.find_video_files(temp_dir)

    @patch.object(_1_ytdlp, "load_key", return_value=["mp4"])
    def test_generated_videos_are_not_treated_as_inputs(self, _):
        with tempfile.TemporaryDirectory() as temp_dir:
            previous_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                output_dir = Path("output")
                output_dir.mkdir()
                (output_dir / "output_sub.mp4").touch()
                (output_dir / "output_dub.mp4").touch()

                with self.assertRaisesRegex(ValueError, "found 0"):
                    _1_ytdlp.find_video_files()
            finally:
                os.chdir(previous_cwd)


if __name__ == "__main__":
    unittest.main()
