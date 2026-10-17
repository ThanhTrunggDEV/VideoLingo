import importlib.util
import time
import streamlit as st
import requests
from translations.translations import translate as t
from core.utils import *
from core.utils.ask_gpt import is_local_endpoint, normalize_base_url


def config_input(label, key, help=None, placeholder=None):
    """Generic config input handler"""
    val = st.text_input(label, value=load_key(key), help=help, placeholder=placeholder)
    if val != load_key(key):
        update_key(key, val)
    return val


def burned_subtitles_settings():
    """Which subtitles are burned into the subtitle video and into the dubbed video."""
    from core.utils.subtitle_style import BURN_MODES, BURN_MODE_KEYS, get_burn_mode
    labels = {"bilingual": t("Bilingual"), "translation": t("Translation only"), "source": t("Source only")}
    columns = st.columns(2)
    for column, video, label in zip(columns, ("subtitle", "dubbed"), (t("Subtitle video"), t("Dubbed video"))):
        key = BURN_MODE_KEYS[video][0]
        mode = get_burn_mode(video)
        with column:
            selected = st.selectbox(
                label,
                options=list(BURN_MODES),
                index=BURN_MODES.index(mode),
                format_func=labels.get,
                help=t("Which subtitles are burned into this video"),
                key=f"burn_mode_{video}",
            )
        if selected != mode:
            update_key(key, selected, add_missing=True)
            st.rerun()


def subtitle_style_settings():
    """Font, size and color of the burned-in subtitles; the other values of subtitle.style are edited in config.yaml."""
    from core.utils.subtitle_style import default_font, get_subtitle_style, save_subtitle_style, to_ass_color, to_hex_color
    with st.popover(t("Subtitle Style"), use_container_width=True):
        style = {}
        columns = st.columns(2)
        for column, kind, label in zip(columns, ("source", "translation"), (t("Source subtitles"), t("Translated subtitles"))):
            current = get_subtitle_style(kind)
            with column:
                st.markdown(f"**{label}**")
                font_name = st.text_input(
                    t("Font"), value=current["font_name"], placeholder=default_font(), key=f"subtitle_font_{kind}",
                    help=t("Name of a font installed on this computer, empty for the default font"))
                font_size = st.number_input(
                    t("Font size"), min_value=1, max_value=100, value=int(current["font_size"]), key=f"subtitle_size_{kind}")
                color = st.color_picker(t("Font color"), value=to_hex_color(current["font_color"]), key=f"subtitle_color_{kind}")
            # The transparency of the configured color is kept as long as the color is
            font_color = current["font_color"] if color == to_hex_color(current["font_color"]) else to_ass_color(color)
            style[kind] = {**current, "font_name": font_name.strip(), "font_size": font_size, "font_color": font_color}
        if style != {kind: get_subtitle_style(kind) for kind in style}:
            save_subtitle_style(style)
            st.rerun()


# The languages shown first keep the order of the old dropdown
LANGUAGE_LABELS = {
    "en": "🇺🇸 English", "zh": "🇨🇳 简体中文", "es": "🇪🇸 Español", "ru": "🇷🇺 Русский",
    "fr": "🇫🇷 Français", "de": "🇩🇪 Deutsch", "it": "🇮🇹 Italiano", "ja": "🇯🇵 日本語",
    "yue": "🇭🇰 粵語", "ko": "🇰🇷 한국어", "pt": "🇵🇹 Português", "ar": "🇸🇦 العربية",
    "id": "🇮🇩 Bahasa Indonesia", "th": "🇹🇭 ไทย", "vi": "🇻🇳 Tiếng Việt", "tr": "🇹🇷 Türkçe",
    "hi": "🇮🇳 हिन्दी", "ms": "🇲🇾 Bahasa Melayu", "nl": "🇳🇱 Nederlands", "sv": "🇸🇪 Svenska",
    "da": "🇩🇰 Dansk", "fi": "🇫🇮 Suomi", "pl": "🇵🇱 Polski", "cs": "🇨🇿 Čeština",
    "fil": "🇵🇭 Filipino", "fa": "🇮🇷 فارسی", "el": "🇬🇷 Ελληνικά", "ro": "🇷🇴 Română",
    "hu": "🇭🇺 Magyar", "mk": "🇲🇰 Македонски",
}


def recognition_languages(configured=None):
    """Label -> code for the recognition language dropdown: every Qwen3-ASR language."""
    from core.asr_backend.qwen_asr_local import ISO_TO_QWEN

    langs = {"Auto": "auto"}
    for code in [*LANGUAGE_LABELS, *ISO_TO_QWEN]:
        if code in ISO_TO_QWEN and code not in langs.values():
            langs[LANGUAGE_LABELS.get(code, ISO_TO_QWEN[code])] = code
    # A code set by hand in config.yaml (WhisperX and ElevenLabs know more languages) stays selectable
    if configured not in langs.values():
        langs[str(configured)] = configured
    return langs


def _fetch_model_list(base_url, api_key):
    """Fetch available models from OpenAI-compatible /v1/models endpoint."""
    if not base_url or (not api_key and not is_local_endpoint(base_url)):
        return []
    url = normalize_base_url(base_url) + "/models"
    try:
        resp = requests.get(
            url, headers={"Authorization": f"Bearer {api_key}"} if api_key else {}, timeout=10
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        return sorted([m["id"] for m in data if "id" in m])
    except Exception:
        return []


def _search_models(search_term, **kwargs):
    """Search function for st_searchbox — returns models matching the search term."""
    models = st.session_state.get("_model_list", [])
    if not search_term:
        return models if models else []
    term = search_term.lower()
    matched = [m for m in models if term in m.lower()]
    # Always include the raw input as an option so users can type custom model names
    if search_term not in matched:
        matched.insert(0, search_term)
    return matched


def page_setting():
    # Widen the sidebar slightly to accommodate the model searchbox
    st.markdown(
        """<style>[data-testid="stSidebar"] {min-width: 420px; max-width: 420px;}</style>""",
        unsafe_allow_html=True,
    )

    with st.expander(t("LLM Configuration"), expanded=True):
        config_input(t("API_KEY"), "api.key", placeholder=t("Enter your API key"))
        config_input(
            t("BASE_URL"),
            "api.base_url",
            help=t("Openai format, will add /v1/chat/completions automatically"),
        )

        # Try to use searchbox for model selection, fall back to text_input
        try:
            from streamlit_searchbox import st_searchbox
            from streamlit_searchbox import _list_to_options_js, _list_to_options_py

            if st.button(
                t("Fetch Model List"), key="fetch_models", use_container_width=True
            ):
                with st.spinner(t("Fetching models...")):
                    models = _fetch_model_list(
                        load_key("api.base_url"), load_key("api.key")
                    )
                    st.session_state["_model_list"] = models
                    if models:
                        # Update searchbox internal state directly so dropdown shows options
                        sb_key = "model_searchbox"
                        if sb_key in st.session_state:
                            st.session_state[sb_key]["options_js"] = (
                                _list_to_options_js(models)
                            )
                            st.session_state[sb_key]["options_py"] = (
                                _list_to_options_py(models)
                            )
                        st.toast(
                            t("Fetched {n} models").replace("{n}", str(len(models))),
                            icon="✅",
                        )
                    else:
                        st.toast(
                            t(
                                "Failed to fetch models, please check API Key and Base URL"
                            ),
                            icon="❌",
                        )

            current_model = load_key("api.model")
            model_list = st.session_state.get("_model_list", None)

            sb_key = "model_searchbox"
            selected = st_searchbox(
                _search_models,
                placeholder=t("Search or enter model name"),
                default=current_model if current_model else None,
                default_searchterm=current_model if current_model else "",
                default_use_searchterm=True,
                default_options=model_list if model_list else None,
                key=sb_key,
                clear_on_submit=False,
            )
            if selected and selected != load_key("api.model"):
                update_key("api.model", selected)

            if st.button("📡 " + t("Check API"), key="api", use_container_width=True):
                with st.spinner(t("Check API") + "..."):
                    is_valid, error = check_api()
                show_api_check(is_valid, error)
        except ImportError:
            c1, c2 = st.columns([4, 1])
            with c1:
                config_input(
                    t("MODEL"),
                    "api.model",
                    help=t("click to check API validity") + " 👉",
                    placeholder=t("Search or enter model name"),
                )
            with c2:
                if st.button("📡", key="api"):
                    is_valid, error = check_api()
                    show_api_check(is_valid, error)
        llm_support_json = st.toggle(
            t("LLM JSON Format Support"),
            value=load_key("api.llm_support_json"),
            help=t("Enable if your LLM supports JSON mode output"),
        )
        if llm_support_json != load_key("api.llm_support_json"):
            update_key("api.llm_support_json", llm_support_json)
            st.rerun()
    with st.expander(t("Subtitles Settings"), expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            langs = recognition_languages(load_key("whisper.language"))
            lang = st.selectbox(
                t("Recog Lang"),
                options=list(langs.keys()),
                index=list(langs.values()).index(load_key("whisper.language")),
            )
            if langs[lang] != load_key("whisper.language"):
                update_key("whisper.language", langs[lang])
                st.rerun()

        runtimes = ["local", "elevenlabs", "mai"]
        configured_runtime = load_key("whisper.runtime")
        if configured_runtime not in runtimes:
            st.warning(t("The 302.ai WhisperX cloud service has been retired. Select Local or ElevenLabs to continue."))
        runtime = st.selectbox(
            t("ASR Runtime"),
            options=runtimes,
            index=runtimes.index(configured_runtime) if configured_runtime in runtimes else None,
            format_func=lambda x: {
                "local": t("Local"),
                "elevenlabs": t("ElevenLabs"),
                "mai": t("MAI-Transcribe-2"),
            }[x],
            help=t(
                "Local Qwen3-ASR runs best on an NVIDIA GPU or Apple Silicon (CPU works but is slow); cloud recognition requires a provider API key."
            ),
        )
        if runtime is not None and runtime != configured_runtime:
            update_key("whisper.runtime", runtime)
            st.rerun()
        if runtime == "local":
            backends = ["qwen", "whisperx"]
            configured_backend = load_key_or("whisper.backend", "qwen")
            backend = st.selectbox(
                t("Local ASR Backend"),
                options=backends,
                index=backends.index(configured_backend) if configured_backend in backends else 0,
                format_func=lambda x: {
                    "qwen": t("Qwen3-ASR + ForcedAligner (default)"),
                    "whisperx": t("WhisperX (manual install)"),
                }[x],
            )
            if backend != configured_backend:
                update_key("whisper.backend", backend, add_missing=True)
                st.rerun()
            if backend == "whisperx" and importlib.util.find_spec("whisperx") is None:
                st.warning(t("WhisperX is not installed. Run `python installer.py --with-whisperx` (Windows/Linux), or set whisper.backend to qwen. See docs/pages/docs/whisperx-manual.en-US.md."))
        if runtime == "elevenlabs":
            config_input(t("ElevenLabs API"), "whisper.elevenlabs_api_key")
        elif runtime == "mai":
            from core.asr_backend.mai_asr import detect_region

            configured_provider = str(load_key_or("whisper.mai_provider", "azure") or "azure").lower()
            providers = ["azure", "openrouter"]
            provider = st.selectbox(
                t("MAI provider"), options=providers,
                index=providers.index(configured_provider) if configured_provider in providers else 0,
                format_func=lambda value: "Azure Speech" if value == "azure" else "OpenRouter",
            )
            if provider != configured_provider:
                update_key("whisper.mai_provider", provider, add_missing=True)
                st.rerun()
            if provider == "openrouter":
                stored_key = str(load_key_or("whisper.mai_openrouter_api_key", "") or "")
                router_key = st.text_input(
                    t("OpenRouter API key"), value=stored_key, type="password",
                )
                if router_key != stored_key:
                    update_key("whisper.mai_openrouter_api_key", router_key, add_missing=True)
                st.caption(t("Audio is sent to OpenRouter for MAI-Transcribe-2 and may incur charges."))
            else:
                stored_key = str(load_key_or("whisper.mai_api_key", "") or "")
                speech_key = st.text_input(
                    t("Azure Speech key"), value=stored_key, type="password",
                    help=t("Use a key from an Azure Speech resource in a region where MAI-Transcribe-2 is available."),
                )
                if speech_key != stored_key:
                    update_key("whisper.mai_api_key", speech_key, add_missing=True)
                stored_region = str(load_key_or("whisper.mai_region", "") or "")
                region = st.text_input(
                    t("Azure Speech region or resource endpoint"), value=stored_region,
                    help=t("Enter a region such as eastus, or the HTTPS endpoint of your Azure Speech resource. Leave blank to detect the region from the key."),
                )
                if region != stored_region:
                    update_key("whisper.mai_region", region.strip(), add_missing=True)
                if st.button(t("Detect Azure Speech region"), disabled=not speech_key.strip()):
                    with st.spinner(t("Detecting Azure Speech region...")):
                        detected = detect_region(speech_key.strip())
                    if detected:
                        update_key("whisper.mai_region", detected, add_missing=True)
                        st.rerun()
                    else:
                        st.warning(t("Region detection failed. Check the key or enter the resource region manually."))
                st.caption(t("MAI-Transcribe-2 is in public preview. Audio is sent to Azure Speech."))

        with c2:
            target_language = st.text_input(
                t("Target Lang"),
                value=load_key("target_language"),
                help=t(
                    "Input any language in natural language, as long as llm can understand"
                ),
            )
            if target_language != load_key("target_language"):
                update_key("target_language", target_language)
                st.rerun()

        configured_style = str(load_key_or("translation_style", "") or "")
        translation_style = st.text_input(
            t("Translation style"),
            value=configured_style,
            placeholder=t("e.g. colloquial, short sentences"),
            help=t("Optional. Describe the style of the translation in your own words, it is added to the translation prompt"),
        )
        if translation_style != configured_style:
            update_key("translation_style", translation_style, add_missing=True)
            st.rerun()

        demucs_available = importlib.util.find_spec("demucs") is not None
        if not demucs_available and load_key("demucs"):
            update_key("demucs", False)
        demucs = st.toggle(
            t("Vocal separation enhance"),
            value=load_key("demucs"),
            disabled=not demucs_available,
            help=t("Vocal separation is unavailable in this installation") if not demucs_available else t(
                "Recommended for videos with loud background noise, but will increase processing time"
            ),
        )
        if demucs != load_key("demucs"):
            update_key("demucs", demucs)
            st.rerun()

        from core._1_ytdlp import is_audio_only_input
        audio_only = is_audio_only_input()
        if audio_only:
            st.toggle(
                t("Burn-in Subtitles"),
                value=False,
                disabled=True,
                help=t("Audio-only input produces subtitle files only; no video is generated."),
            )
        else:
            burn_subtitles = st.toggle(
                t("Burn-in Subtitles"),
                value=load_key("burn_subtitles"),
                help=t(
                    "Whether to burn subtitles into the video, will increase processing time"
                ),
            )
            if burn_subtitles != load_key("burn_subtitles"):
                update_key("burn_subtitles", burn_subtitles)
                st.rerun()
            if burn_subtitles:
                burned_subtitles_settings()
                subtitle_style_settings()

        pause_before_translate = st.toggle(
            t("Pause before translation"),
            value=load_key("pause_before_translate"),
            help=t("Pause after the terminology is extracted, so that you can edit `output/log/terminology.json` before the translation starts"),
        )
        if pause_before_translate != load_key("pause_before_translate"):
            update_key("pause_before_translate", pause_before_translate)
            st.rerun()

        pause_after_translate = st.toggle(
            t("Pause after translation"),
            value=bool(load_key_or("pause_after_translate", False)),
            help=t("Pause after the translation, so that you can edit the `Translation` column of `output/log/translation_results.xlsx` before the subtitles are generated"),
        )
        if pause_after_translate != bool(load_key_or("pause_after_translate", False)):
            update_key("pause_after_translate", pause_after_translate, add_missing=True)
            st.rerun()
    with st.expander(t("Dubbing Settings"), expanded=True):
        tts_method_labels = {
            "openai_tts": t("OpenAI TTS"),
            "fish_audio": t("Fish Audio"),
            "sf_fish_tts": t("SiliconFlow Fish TTS"),
            "edge_tts": t("Edge TTS"),
            "gpt_sovits": t("GPT-SoVITS"),
            "custom_tts": t("Custom TTS"),
            "sf_cosyvoice2": t("SiliconFlow CosyVoice2"),
            "f5tts": t("F5-TTS"),
            "omnivoice": t("OmniVoice"),
            "vieneu": t("VieNeu-TTS"),
        }
        from core.st_utils.tts_settings import select_tts_method
        select_tts = select_tts_method(tts_method_labels)

        # sub settings for each tts method
        if select_tts == "sf_fish_tts":

            # Add mode selection dropdown
            mode_options = {
                "preset": t("Preset"),
                "custom": t("Refer_stable"),
                "dynamic": t("Refer_dynamic"),
            }
            selected_mode = st.selectbox(
                t("Mode Selection"),
                options=list(mode_options.keys()),
                format_func=lambda x: mode_options[x],
                index=list(mode_options.keys()).index(load_key("sf_fish_tts.mode"))
                if load_key("sf_fish_tts.mode") in mode_options.keys()
                else 0,
            )
            if selected_mode != load_key("sf_fish_tts.mode"):
                update_key("sf_fish_tts.mode", selected_mode)
                st.rerun()
            if selected_mode == "preset":
                config_input(t("Voice"), "sf_fish_tts.voice")

        elif select_tts == "openai_tts":
            config_input(t("OpenAI Voice"), "openai_tts.voice")

        elif select_tts == "fish_audio":
            from core.st_utils.tts_settings import save_settings
            from core.tts_backend.fish_audio_tts import voice_id_of
            mode_options = {
                "clone": t("Voice of the video"),
                "preset": t("Fixed voice"),
            }
            current_mode = load_key_or("fish_audio.mode", "clone")
            selected_mode = st.selectbox(
                t("Mode Selection"),
                options=list(mode_options.keys()),
                format_func=lambda x: mode_options[x],
                index=list(mode_options.keys()).index(current_mode) if current_mode in mode_options else 0,
            )
            if selected_mode != current_mode:
                save_settings({"fish_audio.mode": selected_mode})
                st.rerun()
            if selected_mode == "preset":
                voices = dict(load_key_or("fish_audio.voices", None) or {})
                voice_id = str(load_key_or("fish_audio.voice_id", "") or "")
                other = t("Another voice of fish.audio")
                names = list(voices) + [other]
                listed = next((name for name, listed_id in voices.items() if listed_id == voice_id), other)
                # The key keeps "Another voice" selected while its field is still empty
                selected_voice = st.selectbox(t("Voice"), names, index=names.index(listed), key="fish_audio_voice")
                if selected_voice == other:
                    entered = st.text_input(t("Voice ID or the address of its page"),
                                            value="" if voice_id in voices.values() else voice_id,
                                            help=t("Open a voice on fish.audio and copy the address of its page"))
                    selected_id = voice_id_of(entered)
                else:
                    selected_id = voices[selected_voice]
                if selected_id and selected_id != voice_id:
                    save_settings({"fish_audio.voice_id": selected_id})

        elif select_tts == "gpt_sovits":
            st.info(t("Please refer to Github homepage for GPT_SoVITS configuration"))
            config_input(t("SoVITS Character"), "gpt_sovits.character")

            refer_mode_options = {
                1: t("Mode 1: Use provided reference audio only"),
                2: t("Mode 2: Use first audio from video as reference"),
                3: t("Mode 3: Use each audio from video as reference"),
            }
            selected_refer_mode = st.selectbox(
                t("Refer Mode"),
                options=list(refer_mode_options.keys()),
                format_func=lambda x: refer_mode_options[x],
                index=list(refer_mode_options.keys()).index(
                    load_key("gpt_sovits.refer_mode")
                ),
                help=t("Configure reference audio mode for GPT-SoVITS"),
            )
            if selected_refer_mode != load_key("gpt_sovits.refer_mode"):
                update_key("gpt_sovits.refer_mode", selected_refer_mode)
                st.rerun()

        elif select_tts == "edge_tts":
            config_input(t("Edge TTS Voice"), "edge_tts.voice")

        elif select_tts == "omnivoice":
            try:
                import omnivoice  # noqa: F401
            except ImportError:
                st.error(t("OmniVoice runs locally and requires the `omnivoice` package (github.com/k2-fsa/OmniVoice)."))
                st.caption("Run: uv run start.py (auto-installs omnivoice on Windows/Linux)")
            from core.st_utils.tts_settings import save_settings
            refer_mode_options = {
                "stable": t("Voice of the video"),
                "dynamic": t("Refer_dynamic"),
                "custom": t("Custom voice"),
            }
            current_mode = load_key_or("omnivoice.refer_mode", "stable")
            if current_mode not in refer_mode_options:
                current_mode = "stable"
            selected_mode = st.selectbox(
                t("Mode Selection"),
                options=list(refer_mode_options.keys()),
                format_func=lambda x: refer_mode_options[x],
                index=list(refer_mode_options.keys()).index(current_mode),
            )
            if selected_mode != current_mode:
                save_settings({"omnivoice.refer_mode": selected_mode})
                st.rerun()
            if selected_mode == "custom":
                import os
                from core.tts_backend.omnivoice_tts import save_custom_reference
                if st.session_state.pop("_omnivoice_custom_clear", False):
                    # Fresh widgets, so the cleared reference is not saved again from stale state
                    st.session_state["_omnivoice_custom_epoch"] = st.session_state.get("_omnivoice_custom_epoch", 0) + 1
                    st.session_state.pop("_omnivoice_custom_upload_id", None)
                    st.session_state.pop("omnivoice_custom_ref_text", None)
                epoch = st.session_state.get("_omnivoice_custom_epoch", 0)
                saved_audio = str(load_key_or("omnivoice.custom_ref_audio", "") or "")
                saved_text = str(load_key_or("omnivoice.custom_ref_text", "") or "")
                if saved_audio and os.path.exists(saved_audio):
                    st.caption(t("Custom reference in use: {file}").replace("{file}", saved_audio))
                    try:
                        st.audio(saved_audio)
                    except Exception:
                        pass
                else:
                    st.warning(t("Upload a 3-10 s clear clip of one speaker and enter its exact transcript."))
                note = st.session_state.pop("_omnivoice_custom_note", None)
                if note:
                    (st.warning if note[0] == "warning" else st.success)(note[1])
                uploaded = st.file_uploader(
                    t("Upload reference audio"),
                    type=["wav", "mp3", "m4a", "flac", "ogg"],
                    key=f"omnivoice_custom_ref_{epoch}",
                    help=t("3-10 s of clear speech of one speaker. Used for the whole video."),
                )
                if uploaded is not None:
                    upload_id = f"{uploaded.name}:{uploaded.size}"
                    if st.session_state.get("_omnivoice_custom_upload_id") != upload_id:
                        try:
                            out_path, kept, original = save_custom_reference(uploaded.getvalue(), uploaded.name)
                            save_settings({"omnivoice.custom_ref_audio": out_path})
                            st.session_state["_omnivoice_custom_upload_id"] = upload_id
                            if original > kept + 0.05:
                                st.session_state["_omnivoice_custom_note"] = ("warning", t("Audio was {orig}s; only the first {kept}s are kept. The transcript must match exactly that kept part — listen to the preview to check.").replace("{orig}", f"{original:.1f}").replace("{kept}", f"{kept:.1f}"))
                            else:
                                st.session_state["_omnivoice_custom_note"] = ("success", t("Reference audio saved ({dur}s): {file}").replace("{dur}", f"{kept:.1f}").replace("{file}", out_path))
                            st.rerun()
                        except Exception as e:
                            st.error(f"{t('Could not use this audio')}: {e}")
                ref_text = st.text_area(
                    t("Reference text (exact transcript)"),
                    value=saved_text,
                    placeholder=t("Type exactly what is said in the reference audio"),
                    key="omnivoice_custom_ref_text",
                )
                if ref_text != saved_text:
                    save_settings({"omnivoice.custom_ref_text": ref_text.strip()})
                if saved_audio and os.path.exists(saved_audio):
                    if st.button(t("Clear custom reference"), key="omnivoice_custom_clear"):
                        save_settings({"omnivoice.custom_ref_audio": "", "omnivoice.custom_ref_text": ""})
                        st.session_state["_omnivoice_custom_clear"] = True
                        st.rerun()
            config_input(t("Device"), "omnivoice.device", help=t("auto (cuda:0/mps/cpu), cuda:0, mps, xpu, or cpu"))
            config_input(t("Diffusion steps"), "omnivoice.num_step", help=t("Diffusion steps: 32 for quality, 16 for speed"))

        elif select_tts == "vieneu":
            try:
                import vieneu  # noqa: F401
            except ImportError:
                st.error(t("VieNeu-TTS runs locally and requires the `vieneu` package (docs.vieneu.io)."))
                st.caption("Run: pip install vieneu")
                st.caption(t("VieNeu-TTS also needs the eSpeak NG system package (Ubuntu/Debian: sudo apt install espeak-ng, macOS: brew install espeak)."))
            from core.st_utils.tts_settings import save_settings
            refer_mode_options = {
                "preset": t("Preset voice"),
                "stable": t("Voice of the video"),
                "dynamic": t("Refer_dynamic"),
                "custom": t("Custom voice"),
            }
            current_mode = str(load_key_or("vieneu.refer_mode", "stable") or "stable")
            if current_mode not in refer_mode_options:
                current_mode = "stable"
            selected_mode = st.selectbox(
                t("Mode Selection"),
                options=list(refer_mode_options.keys()),
                format_func=lambda x: refer_mode_options[x],
                index=list(refer_mode_options.keys()).index(current_mode),
                key="vieneu_refer_mode",
            )
            if selected_mode != current_mode:
                save_settings({"vieneu.refer_mode": selected_mode})
                st.rerun()
            if selected_mode == "preset":
                config_input(
                    t("VieNeu preset voice"),
                    "vieneu.voice",
                    help=t("Built-in voice name, e.g. Minh Quân Pro, Adam, Trúc Ly. VieNeu-TTS has 25 preset voices."),
                )
            if selected_mode == "custom":
                import os
                from core.tts_backend.vieneu_tts import save_custom_reference
                if st.session_state.pop("_vieneu_custom_clear", False):
                    # Fresh widgets, so the cleared reference is not saved again from stale state
                    st.session_state["_vieneu_custom_epoch"] = st.session_state.get("_vieneu_custom_epoch", 0) + 1
                    st.session_state.pop("_vieneu_custom_upload_id", None)
                epoch = st.session_state.get("_vieneu_custom_epoch", 0)
                saved_audio = str(load_key_or("vieneu.custom_ref_audio", "") or "")
                if saved_audio and os.path.exists(saved_audio):
                    st.caption(t("Custom reference in use: {file}").replace("{file}", saved_audio))
                    try:
                        st.audio(saved_audio)
                    except Exception:
                        pass
                else:
                    st.warning(t("Upload a 3-8 s clear clip of one speaker. VieNeu-TTS v3 needs no transcript."))
                note = st.session_state.pop("_vieneu_custom_note", None)
                if note:
                    (st.warning if note[0] == "warning" else st.success)(note[1])
                uploaded = st.file_uploader(
                    t("Upload reference audio"),
                    type=["wav", "mp3", "m4a", "flac", "ogg"],
                    key=f"vieneu_custom_ref_{epoch}",
                    help=t("3-8 s of clear speech of one speaker. Used for the whole video."),
                )
                if uploaded is not None:
                    upload_id = f"{uploaded.name}:{uploaded.size}"
                    if st.session_state.get("_vieneu_custom_upload_id") != upload_id:
                        try:
                            out_path, kept, original = save_custom_reference(uploaded.getvalue(), uploaded.name)
                            save_settings({"vieneu.custom_ref_audio": out_path})
                            st.session_state["_vieneu_custom_upload_id"] = upload_id
                            if original > kept + 0.05:
                                st.session_state["_vieneu_custom_note"] = ("warning", t("Audio was {orig}s; only the first {kept}s are kept.").replace("{orig}", f"{original:.1f}").replace("{kept}", f"{kept:.1f}"))
                            else:
                                st.session_state["_vieneu_custom_note"] = ("success", t("Reference audio saved ({dur}s): {file}").replace("{dur}", f"{kept:.1f}").replace("{file}", out_path))
                            st.rerun()
                        except Exception as e:
                            st.error(f"{t('Could not use this audio')}: {e}")
                if saved_audio and os.path.exists(saved_audio):
                    if st.button(t("Clear custom reference"), key="vieneu_custom_clear"):
                        save_settings({"vieneu.custom_ref_audio": ""})
                        st.session_state["_vieneu_custom_clear"] = True
                        st.rerun()
            if selected_mode in ("stable", "dynamic", "custom"):
                denoise = st.toggle(
                    t("Denoise reference audio"),
                    value=bool(load_key_or("vieneu.denoise", True)),
                    help=t("Clean the reference clip before cloning. Keep on unless the clip is already clean."),
                )
                if denoise != bool(load_key_or("vieneu.denoise", True)):
                    save_settings({"vieneu.denoise": denoise})
                    st.rerun()
            engine_options = {
                "standard": t("Standard (v3 Turbo, best quality)"),
                "v3nano": t("Nano (faster, lower quality)"),
            }
            current_engine = str(load_key_or("vieneu.mode", "standard") or "standard")
            if current_engine not in engine_options:
                current_engine = "standard"
            selected_engine = st.selectbox(
                t("VieNeu engine"),
                options=list(engine_options.keys()),
                format_func=lambda x: engine_options[x],
                index=list(engine_options.keys()).index(current_engine),
                key="vieneu_engine",
            )
            if selected_engine != current_engine:
                save_settings({"vieneu.mode": selected_engine})
                st.rerun()
            precision_options = {"fp32": "fp32", "int8": "int8"}
            current_precision = str(load_key_or("vieneu.precision", "fp32") or "fp32")
            if current_precision not in precision_options:
                current_precision = "fp32"
            selected_precision = st.selectbox(
                t("VieNeu precision (CPU only)"),
                options=list(precision_options.keys()),
                index=list(precision_options.keys()).index(current_precision),
                help=t("fp32 is the best quality. int8 is faster but needs a VNNI-capable CPU."),
                key="vieneu_precision",
            )
            if selected_precision != current_precision:
                save_settings({"vieneu.precision": selected_precision})
                st.rerun()
            config_input(t("VieNeu backend"), "vieneu.backend", help=t("auto (CUDA if available, else CPU) or onnx (force CPU)"))



def check_api():
    """Returns (is_valid, error): one request without retries, never answered from the cache."""
    try:
        resp = ask_gpt.__wrapped__(
            f"This is a test ({time.time():.0f}), response 'message':'success' in json format.",
            resp_type="json",
            log_title="None",
        )
        if resp.get("message") == "success":
            return True, ""
        return False, f"Unexpected response: {str(resp)[:300]}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:500]}"


def show_api_check(is_valid, error):
    if is_valid:
        st.toast(t("API Key is valid"), icon="✅")
    else:
        st.toast(t("API check failed"), icon="❌")
        st.error(f"{t('API check failed')}: {error}")


if __name__ == "__main__":
    check_api()
