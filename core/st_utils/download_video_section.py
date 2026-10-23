import os
import re
import shutil
from time import sleep

import streamlit as st
from core._1_ytdlp import download_video_ytdlp, find_media_file, find_subtitle_file, needs_cookies, write_input_manifest
from core._2_import_subtitles import add_input_subtitles, read_cues
from core.utils.models import _2_CLEANED_CHUNKS
from core.utils import *
from translations.translations import translate as t

OUTPUT_DIR = "output"


def _css_text(value):
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _inject_file_uploader_i18n():
    # Streamlit does not expose official i18n for file_uploader internals.
    # Streamlit 1.49 DOM:
    #   div[data-testid="stFileUploaderDropzoneInstructions"]
    #     > span    (cloud icon, must keep)
    #     > div     (column flex)
    #         > span  (1st: "Drag and drop ... here")
    #         > span  (2nd: "Limit ... · MP4, MOV ...")
    # Streamlit 1.66 DOM: the inner div holds a single native line with the
    # limit and the accepted formats, and the browse button is labelled
    # "Upload". The :has() guard keeps the legacy replacements from hiding
    # that native line, and the selectors are scoped to the two uploaders of
    # this section so the other uploaders keep their own text.
    drag_text = _css_text(t("Drag and drop file here"))
    limit_text = _css_text(t("Limit 4GB per file · MP4, MOV, AVI, MKV, FLV, WMV, WEBM, WAV, MP3, FLAC, M4A"))
    browse_text = _css_text(t("Browse files"))
    st.markdown(
        f"""
        <style>
        /* Legacy two-line dropzone: translate both lines */
        .st-key-media_upload div[data-testid="stFileUploaderDropzoneInstructions"] > div:has(> span:nth-of-type(2)) > span:nth-of-type(1),
        .st-key-subtitle_upload div[data-testid="stFileUploaderDropzoneInstructions"] > div:has(> span:nth-of-type(2)) > span:nth-of-type(1) {{
            font-size: 0 !important;
            line-height: 1.4;
        }}
        .st-key-media_upload div[data-testid="stFileUploaderDropzoneInstructions"] > div:has(> span:nth-of-type(2)) > span:nth-of-type(1)::before,
        .st-key-subtitle_upload div[data-testid="stFileUploaderDropzoneInstructions"] > div:has(> span:nth-of-type(2)) > span:nth-of-type(1)::before {{
            content: "{drag_text}";
            font-size: 1rem;
        }}
        /* Legacy sub line (limit + accepted formats) */
        .st-key-media_upload div[data-testid="stFileUploaderDropzoneInstructions"] > div > span:nth-of-type(2),
        .st-key-subtitle_upload div[data-testid="stFileUploaderDropzoneInstructions"] > div > span:nth-of-type(2) {{
            font-size: 0 !important;
            line-height: 1.4;
        }}
        .st-key-media_upload div[data-testid="stFileUploaderDropzoneInstructions"] > div > span:nth-of-type(2)::before {{
            content: "{limit_text}";
            font-size: 0.8rem;
        }}
        /* The uploader of the subtitles takes SRT files only */
        .st-key-subtitle_upload div[data-testid="stFileUploaderDropzoneInstructions"] > div > span:nth-of-type(2)::before {{
            content: "SRT";
            font-size: 0.8rem;
        }}
        /* New single-line dropzone: translate the native limit line */
        .st-key-media_upload div[data-testid="stFileUploaderDropzoneInstructions"] > div:not(:has(> span:nth-of-type(2))) > span:nth-of-type(1),
        .st-key-subtitle_upload div[data-testid="stFileUploaderDropzoneInstructions"] > div:not(:has(> span:nth-of-type(2))) > span:nth-of-type(1) {{
            font-size: 0 !important;
            line-height: 1.4;
        }}
        .st-key-media_upload div[data-testid="stFileUploaderDropzoneInstructions"] > div:not(:has(> span:nth-of-type(2))) > span:nth-of-type(1)::before {{
            content: "{limit_text}";
            font-size: 0.8rem;
        }}
        .st-key-subtitle_upload div[data-testid="stFileUploaderDropzoneInstructions"] > div:not(:has(> span:nth-of-type(2))) > span:nth-of-type(1)::before {{
            content: "SRT";
            font-size: 0.8rem;
        }}
        /* Browse files button: replace the label, keep the icon */
        .st-key-media_upload div[data-testid="stFileUploader"] button[kind="secondary"],
        .st-key-subtitle_upload div[data-testid="stFileUploader"] button[kind="secondary"] {{
            font-size: 0 !important;
        }}
        .st-key-media_upload div[data-testid="stFileUploader"] button[kind="secondary"] [data-testid="stMarkdownContainer"],
        .st-key-subtitle_upload div[data-testid="stFileUploader"] button[kind="secondary"] [data-testid="stMarkdownContainer"] {{
            display: none !important;
        }}
        .st-key-media_upload div[data-testid="stFileUploader"] button[kind="secondary"]::before,
        .st-key-subtitle_upload div[data-testid="stFileUploader"] button[kind="secondary"]::before {{
            content: "{browse_text}";
            font-size: 0.875rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _uploaded_name(filename):
    """A safe output name for an uploaded media file, never a hidden one."""
    name = filename.replace(' ', '_')
    stem, dot, extension = name.rpartition('.')
    if dot and extension:
        clean_ext = "." + extension.lower()
        media_ext = extension.lower()
    else:
        stem, clean_ext, media_ext = name, "", ""
    base = re.sub(r'[^\w\-\.]', '', stem).strip('.') or "media"
    return base + clean_ext, media_ext


def _save_uploaded_media(uploaded_file, upload_id):
    """Write the uploaded media file to output/ and remember it as the input."""
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    clean_name, media_ext = _uploaded_name(uploaded_file.name)
    media_path = os.path.join(OUTPUT_DIR, clean_name)
    with open(media_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    media_type = "video" if media_ext in load_key("allowed_video_formats") else "audio"
    write_input_manifest(media_path, media_type)
    st.session_state["_processed_upload_id"] = upload_id
    return True


def _subtitle_uploader(label, hint):
    with st.container(key="subtitle_upload"):
        return st.file_uploader(label, type=["srt"], help=hint)


def _add_uploaded_subtitles(uploaded_file, new_input=False):
    """Save the uploaded subtitles. True when they were added, the page has to run again then."""
    if not uploaded_file:
        return False
    try:
        read_cues(uploaded_file.getvalue())
    except ValueError as e:
        st.error(t("These subtitles can not be used: {error}").replace("{error}", str(e)))
        return False
    if new_input and os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    add_input_subtitles(uploaded_file.name.replace(' ', '_'), uploaded_file.getvalue())
    return True


def upload_subtitles(label, hint, new_input=False):
    """Subtitles of the user for the input. True when they were added, the page has to run again then."""
    return _add_uploaded_subtitles(_subtitle_uploader(label, hint), new_input)


def input_subtitles_section(media_type):
    """The subtitles that come with the input, or the place to add them before the recognition."""
    subtitle_file = find_subtitle_file()
    if subtitle_file:
        text = "Subtitles `{file}` are translated as they are, no video is generated." if media_type == "subtitle" \
            else "Subtitles `{file}` are used instead of the recognition, with their own lines and times."
        st.info(t(text).replace("{file}", os.path.basename(subtitle_file)))
        if load_key("whisper.language") == "auto":
            st.warning(t("The language of subtitles can not be detected. Select their language as `Recog Lang` in the settings."))
    elif not os.path.exists(_2_CLEANED_CHUNKS):
        with st.expander(t("Use existing subtitles")):
            _inject_file_uploader_i18n()
            if upload_subtitles(t("Subtitles (SRT)"), t("They are used instead of the recognition, with their own lines and times")):
                st.rerun()


def download_video_section():
    st.header(t("a. Download or Upload Video"))
    with st.container(border=True):
        try:
            media_file, media_type = find_media_file()
            if media_type == "video":
                st.video(media_file)
            elif media_type == "audio":
                st.audio(media_file)
            input_subtitles_section(media_type)
            if st.button(t("Delete and Reselect"), key="delete_video_button"):
                os.remove(media_file)
                if os.path.exists(OUTPUT_DIR):
                    shutil.rmtree(OUTPUT_DIR)
                st.session_state.pop("_processed_upload_id", None)
                sleep(1)
                st.rerun()
            return True
        except ValueError as e:
            if "No media file found" not in str(e):
                st.error(t("Media file detection failed: {error}").replace("{error}", str(e)))
                if st.button(t("Clear output and reselect"), key="clear_output_button"):
                    if os.path.exists(OUTPUT_DIR):
                        shutil.rmtree(OUTPUT_DIR)
                    st.session_state.pop("_processed_upload_id", None)
                    st.rerun()
                return False
        except Exception:
            pass

        col1, col2 = st.columns([3, 1])
        with col1:
            url = st.text_input(t("Enter video link (YouTube/Bilibili/...):"))
        with col2:
            res_dict = {
                "360p": "360",
                "1080p": "1080",
                t("Best"): "best"
            }
            target_res = load_key("ytb_resolution")
            res_options = list(res_dict.keys())
            default_idx = list(res_dict.values()).index(target_res) if target_res in res_dict.values() else 0
            res_display = st.selectbox(t("Resolution"), options=res_options, index=default_idx)
            res = res_dict[res_display]
        with st.expander(t("Download Settings")):
            cookies_path = st.text_input(
                t("Cookies Path"),
                value=load_key("youtube.cookies_path") or "",
                placeholder="cookies.txt",
                help=t("Path of a cookies.txt file exported from your browser, for videos that require sign-in"),
            )
            if cookies_path != (load_key("youtube.cookies_path") or ""):
                update_key("youtube.cookies_path", cookies_path)
        if st.button(t("Download Video"), key="download_button", width="stretch"):
            if url:
                try:
                    with st.spinner(t("Downloading video...")):
                        download_video_ytdlp(url, resolution=res)
                except Exception as e:
                    st.error(f"{t('Download failed')}: {e}")
                    if needs_cookies(e):
                        st.info(t("This video requires sign-in. Export cookies.txt from your browser and fill in its path under Download Settings."))
                    return False
                st.rerun()

        _inject_file_uploader_i18n()
        with st.container(key="media_upload"):
            uploaded_file = st.file_uploader(t("Upload local media file"), type=load_key("allowed_video_formats") + load_key("allowed_audio_formats"))
        uploaded_subtitles = _subtitle_uploader(
            t("Or upload subtitles without a video (SRT)"),
            t("The subtitles are translated with their own lines and times. For a video with these subtitles, upload the video first and add the subtitles below it"),
        )

        media_added = False
        media_present = False
        if uploaded_file:
            upload_id = f"{uploaded_file.name}:{uploaded_file.size}"
            if st.session_state.get("_processed_upload_id") == upload_id:
                try:
                    find_media_file()
                    media_present = True
                    st.warning(t("Upload was already processed. Delete and reselect to upload again."))
                except Exception:
                    st.session_state.pop("_processed_upload_id", None)
                    media_added = _save_uploaded_media(uploaded_file, upload_id)
            else:
                media_added = _save_uploaded_media(uploaded_file, upload_id)

        subtitles_added = _add_uploaded_subtitles(uploaded_subtitles, new_input=not (media_added or media_present))
        if media_added or subtitles_added:
            st.rerun()
        return False
