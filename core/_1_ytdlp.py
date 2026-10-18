import os,sys
import glob
import json
import re
import subprocess
from core.utils import *

OUTPUT_DIR = "output"
INPUT_MANIFEST = "input_manifest.json"
GENERATED_AUDIO_NAMES = {"dub.mp3", "normalized_dub.wav"}
GENERATED_VIDEO_NAMES = {"output_sub.mp4", "output_dub.mp4"}

def sanitize_filename(filename):
    # Remove or replace illegal characters
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    # Ensure filename doesn't start or end with a dot or space
    filename = filename.strip('. ')
    # Use default name if filename is empty
    return filename if filename else 'video'

def update_ytdlp():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "yt-dlp"])
        if 'yt_dlp' in sys.modules:
            del sys.modules['yt_dlp']
        rprint("[green]yt-dlp updated[/green]")
    except subprocess.CalledProcessError as e:
        rprint("[yellow]Warning: Failed to update yt-dlp: {e}[/yellow]")
    from yt_dlp import YoutubeDL
    return YoutubeDL

def needs_cookies(error):
    """True when the site refused the download until the user is signed in."""
    message = str(error).lower()
    return any(hint in message for hint in ("sign in to confirm", "--cookies", "login required"))

def download_video_ytdlp(url, save_path='output', resolution='1080'):
    """Download a video with yt-dlp (YouTube, Bilibili/b23.tv and other supported sites).

    Bilibili videos that require sign-in need a cookies.txt path (Download Settings)."""
    os.makedirs(save_path, exist_ok=True)
    # NOTE: keep this self-contained (no helper calls, only `os` + the names
    # the proxy test injects): tests/test_download_proxy.py execs this
    # function in isolation.
    lowered_url = (url or "").lower()
    is_bilibili = "bilibili" in lowered_url or "b23.tv" in lowered_url
    if not is_bilibili:
        format_sel = 'bestvideo+bestaudio/best' if resolution == 'best' else f'bestvideo[height<={resolution}]+bestaudio/best[height<={resolution}]'
        bilibili_sort = None
    elif resolution == 'best' or (str(resolution).isdigit() and int(resolution) >= 720):
        # 1080p AVC (30080) is refused past ~390MB on this CDN while 1080p
        # HEVC (30077) serves all ranges: prefer HEVC, fall back to AVC.
        # (AV01/1000xx dies at ~1.8MB, excluded by both filters.)
        if resolution == 'best':
            format_sel = 'bestvideo[vcodec^=hvc1]+bestaudio/bestvideo[vcodec^=avc1]+bestaudio/bestvideo+bestaudio/best'
        else:
            format_sel = (
                f'bestvideo[vcodec^=hvc1][height<={resolution}]+bestaudio/'
                f'bestvideo[vcodec^=avc1][height<={resolution}]+bestaudio/'
                f'bestvideo[height<={resolution}]+bestaudio/'
                f'best[height<={resolution}]/best'
            )
        bilibili_sort = ['vcodec:hvc1', 'vcodec:avc1']
    else:
        # 360p AVC (30016) is proven to download cleanly: keep AVC first.
        format_sel = (
            f'bestvideo[vcodec^=avc1][height<={resolution}]+bestaudio/'
            f'bestvideo[height<={resolution}]+bestaudio/'
            f'best[height<={resolution}]/best'
        )
        bilibili_sort = ['vcodec:avc1']
    ydl_opts = {
        'format': format_sel,
        'outtmpl': f'{save_path}/%(title)s.%(ext)s',
        # Linux caps filenames at 255 bytes and CJK characters take 3-4 bytes
        # each: a long Bilibili title passes at `<title>.jpg` but the video
        # stream is fetched as `<title>.f<id>.mp4.part` and the extra suffix
        # overflows ("File name too long"), failing every retry round.
        # trim_file_name counts the full path in characters (4 bytes worst
        # case), so 55 keeps `output/` + title <= 220 bytes and every suffix
        # yt-dlp appends afterwards (`.f30066.mp4.part`) still fits.
        'trim_file_name': 55,
        'noplaylist': True,
        'writethumbnail': True,
        'merge_output_format': 'mp4',
        'postprocessors': [{'key': 'FFmpegThumbnailsConvertor', 'format': 'jpg'}],
        # Large files often die mid-download ("X bytes read, Y more expected")
        # when the server throttles or resets the connection: resume the
        # partial file and retry with backoff instead of starting over.
        'continuedl': True,
        'retries': 20,
        'fragment_retries': 20,
        'retry_sleep': 'exp=1:30',
        'socket_timeout': 30,
    }
    if is_bilibili:
        # Double safety alongside the format string: when the fallback chain
        # reaches a generic `best`, still rank preferred codecs above AV01.
        ydl_opts['format_sort'] = bilibili_sort
        # Bilibili CDN cuts long-lived connections (after ~100-200MB the
        # speed collapses to tens of KiB/s, then "X bytes read, Y more
        # expected" until retries run out). Downloading in 10MB chunks opens
        # a fresh request per chunk, each finishing before the cut: 233MB in
        # 45s with zero errors vs. stuck at 23% after 20 retries without it.
        ydl_opts['http_chunk_size'] = 10 * 1024 * 1024
        # This network resolves the Bilibili CDN (upos-*.akamaized.net) to
        # IPv6 first, but the IPv6 path stalls: every request dies after
        # ~1KB ("1022 bytes read, ... more expected" x20, zero progress).
        # Forcing IPv4 downloads cleanly (294MB in 50s, zero errors).
        # Same as yt-dlp `--force-ipv4`.
        ydl_opts['source_address'] = '0.0.0.0'

    # None/missing inherits yt-dlp's system/environment proxy discovery.
    # An empty string explicitly disables proxies; a URL overrides discovery.
    youtube = load_key("youtube")
    proxy = youtube.get("proxy")
    if proxy is not None:
        if not isinstance(proxy, str):
            raise ValueError('youtube.proxy must be null, an empty string, or a proxy URL')
        ydl_opts['proxy'] = proxy.strip()

    # Read Youtube Cookie File
    cookies_path = str(youtube.get("cookies_path") or "").strip().strip('"')
    if cookies_path:
        if not os.path.isfile(cookies_path):
            raise ValueError(f"Cookies file not found: {cookies_path}")
        ydl_opts["cookiefile"] = cookies_path

    # Mirror CDN hosts for Bilibili media. The primary Akamai host
    # (upos-hz-mirrorakam.akamaized.net) sometimes keeps a truncated copy of
    # a file: it serves the first N MiB, then answers every range request at
    # the cut-off with 36 bytes / an empty reply, while the mirrors serve the
    # very same signed URL (byte-identical by md5). Rotating hosts lets a
    # stalled download resume from the partial file on another edge.
    bilibili_mirrors = [
        'upos-sz-mirrorcos.bilivideo.com',
        'upos-sz-mirrorhw.bilivideo.com',
        'upos-sz-mirror08c.bilivideo.com',
    ]

    # Get YoutubeDL class after updating (once: re-upgrading pip every retry
    # round would be slow, so reuse the class across rounds below).
    YoutubeDL = update_ytdlp()

    def make_downloader(mirror):
        """YoutubeDL variant that serves Bilibili media from `mirror`."""
        if mirror is None:
            return YoutubeDL

        class MirrorYoutubeDL(YoutubeDL):
            def process_info(self, info_dict):
                for fmt in (info_dict.get('requested_formats') or (info_dict,)):
                    media_url = fmt.get('url') or ''
                    if '/upgcxcode/' in media_url:
                        scheme, _, rest = media_url.partition('://')
                        _, _, path = rest.partition('/')
                        fmt['url'] = f'{scheme}://{mirror}/{path}'
                return super().process_info(info_dict)

        return MirrorYoutubeDL

    # yt-dlp gives up after `retries` failures *in total*, even when every
    # retry made progress (e.g. 23% -> 29% across 20 retries, then "Giving
    # up"). For throttled hosts like Bilibili that is wrong: reset the budget
    # whenever the download grew, and only give up after consecutive rounds
    # with zero progress. NOTE: os-only on purpose (see NOTE above).
    last_error = None
    stagnant_rounds = 0
    cdn_plan = [None] + (bilibili_mirrors if is_bilibili else [])
    cdn_index = 0
    for round_no in range(1, 51):
        mirror = cdn_plan[cdn_index % len(cdn_plan)]
        before = 0
        try:
            for _name in os.listdir(save_path):
                try:
                    before += os.path.getsize(os.path.join(save_path, _name))
                except OSError:
                    pass
        except OSError:
            pass
        via = f" via {mirror}" if mirror else ""
        print(f"[download] Round {round_no}/50: resuming at {before / 1048576:.1f} MiB{via}...")
        try:
            with make_downloader(mirror)(ydl_opts) as ydl:
                ydl.download([url])
            last_error = None
            break
        except Exception as e:
            last_error = e
            after = 0
            try:
                for _name in os.listdir(save_path):
                    try:
                        after += os.path.getsize(os.path.join(save_path, _name))
                    except OSError:
                        pass
            except OSError:
                pass
            if after > before:
                stagnant_rounds = 0
                print(f"[download] Round {round_no} failed but grew +{(after - before) / 1048576:.1f} MiB "
                      f"(total {after / 1048576:.1f} MiB) - retry budget reset, continuing...")
            else:
                stagnant_rounds += 1
                # The next round switches to the next CDN host: a truncated
                # copy on one edge says nothing about the mirrors.
                cdn_index += 1
                if stagnant_rounds >= 8:
                    break
                # Only pause once every host stalled a full sweep without
                # progress: HTTP 503 / refused connections on a throttled CDN
                # mean its rolling quota window is exhausted, while a fresh
                # host may resume exactly where this one refused.
                # NOTE: __import__ keeps this working where only `os` plus
                # the injected test names exist (see NOTE above).
                if cdn_index % len(cdn_plan) == 0:
                    _cool = min(120 * (stagnant_rounds // len(cdn_plan)), 600)
                    print(f"[download] All CDNs refused every retry - cooling down {_cool}s "
                          f"(stagnant {stagnant_rounds}/8)...")
                    __import__("time").sleep(_cool)
    if last_error is not None:
        if is_bilibili:
            done = 0
            try:
                for _name in os.listdir(save_path):
                    try:
                        done += os.path.getsize(os.path.join(save_path, _name))
                    except OSError:
                        pass
            except OSError:
                pass
            raise RuntimeError(
                f"Bilibili stopped responding at {done / 1048576:.1f} MiB: the server closed every "
                f"connection ({last_error}). Its CDN throttles large files - try again later to "
                f"resume, or pick 360p (small enough to finish in one go), or export cookies.txt "
                f"from a logged-in browser and set it under Download Settings."
            ) from last_error
        raise last_error

    # A previous failed attempt (e.g. Bilibili AV01 format 100026 cut off at
    # ~1.8MB) can leave stale `.part` files behind. The successful download
    # already removed its own `.part`, so anything left is garbage.
    # NOTE: os-only on purpose (see NOTE above).
    for leftover_name in os.listdir(save_path):
        if leftover_name.endswith(".part") or leftover_name.endswith(".ytdl"):
            try:
                os.remove(os.path.join(save_path, leftover_name))
            except OSError:
                pass
    
    # Check and rename files after download
    for file in os.listdir(save_path):
        if os.path.isfile(os.path.join(save_path, file)):
            filename, ext = os.path.splitext(file)
            new_filename = sanitize_filename(filename)
            if new_filename != filename:
                os.rename(os.path.join(save_path, file), os.path.join(save_path, new_filename + ext))
    media_file = find_video_files(save_path)
    write_input_manifest(media_file, "video", save_path)

def write_input_manifest(media_file: str, media_type: str, save_path='output', subtitle_file=None):
    os.makedirs(save_path, exist_ok=True)
    manifest_path = os.path.join(save_path, INPUT_MANIFEST)
    media_path = media_file.replace("\\", "/") if sys.platform.startswith('win') else media_file
    manifest = {"path": media_path, "type": media_type}
    if subtitle_file:
        manifest["subtitle"] = subtitle_file.replace("\\", "/")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

def _read_input_manifest(save_path='output'):
    manifest_path = os.path.join(save_path, INPUT_MANIFEST)
    if not os.path.exists(manifest_path):
        return None
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    media_file = data.get("path")
    media_type = data.get("type")
    if media_type not in {"video", "audio", "subtitle"} or not media_file or not os.path.exists(media_file):
        return None
    return media_file.replace("\\", "/") if sys.platform.startswith('win') else media_file, media_type

def find_subtitle_file(save_path='output'):
    """The subtitles of the user that take the place of the recognition, or None."""
    manifest_path = os.path.join(save_path, INPUT_MANIFEST)
    if not os.path.exists(manifest_path):
        return None
    with open(manifest_path, "r", encoding="utf-8") as f:
        subtitle_file = json.load(f).get("subtitle")
    return subtitle_file if subtitle_file and os.path.exists(subtitle_file) else None

def find_video_files(save_path='output'):
    video_files = [file for file in glob.glob(save_path + "/*") if os.path.isfile(file) and os.path.splitext(file)[1][1:].lower() in load_key("allowed_video_formats")]
    # change \\ to /, this happen on windows
    if sys.platform.startswith('win'):
        video_files = [file.replace("\\", "/") for file in video_files]
    video_files = [
        file
        for file in video_files
        if os.path.basename(file).lower() not in GENERATED_VIDEO_NAMES
    ]
    if len(video_files) != 1:
        raise ValueError(f"Number of videos found {len(video_files)} is not unique. Please check.")
    return video_files[0]

def find_audio_files(save_path='output'):
    audio_files = [file for file in glob.glob(save_path + "/*") if os.path.splitext(file)[1][1:].lower() in load_key("allowed_audio_formats")]
    if sys.platform.startswith('win'):
        audio_files = [file.replace("\\", "/") for file in audio_files]
    audio_files = [file for file in audio_files if os.path.basename(file) not in GENERATED_AUDIO_NAMES]
    if len(audio_files) != 1:
        raise ValueError(f"Number of audio files found {len(audio_files)} is not unique. Please check.")
    return audio_files[0]

def _safe_find_video_file(save_path='output'):
    try:
        return find_video_files(save_path)
    except ValueError as e:
        if "found 0" in str(e):
            return None
        raise

def _safe_find_audio_file(save_path='output'):
    try:
        return find_audio_files(save_path)
    except ValueError as e:
        if "found 0" in str(e):
            return None
        raise

def find_media_file(save_path='output'):
    manifest = _read_input_manifest(save_path)
    if manifest:
        return manifest
    video_file = _safe_find_video_file(save_path)
    if video_file:
        return video_file, "video"
    audio_file = _safe_find_audio_file(save_path)
    if audio_file:
        return audio_file, "audio"
    raise ValueError("No media file found. Please download or upload a media file first.")

def is_audio_only_input(save_path='output'):
    # True when the input has no video: a standalone audio file, or subtitles only.
    # In this case VideoLingo only produces subtitle files; no video output.
    try:
        _, media_type = find_media_file(save_path)
        return media_type in ("audio", "subtitle")
    except Exception:
        return False

if __name__ == '__main__':
    # Example usage
    url = input('Please enter the URL of the video you want to download: ')
    resolution = input('Please enter the desired resolution (360/480/720/1080, default 1080): ')
    resolution = int(resolution) if resolution.isdigit() else 1080
    download_video_ytdlp(url, resolution=resolution)
    print(f"🎥 Video has been downloaded to {find_video_files()}")
