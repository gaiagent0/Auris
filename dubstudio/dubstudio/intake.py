"""Nyers epizód intake: magyar felirat + leválasztott hang és M&E.

A szinkronlánc eddig kész feliratot, ``vocals.wav``-t és ``me.wav``-t várt.
Ez a modul ezeket állítja elő egy nyers MKV/MP4-ből:

1. **felirat** — a projekt nyelvű külső vagy beágyazott szöveges feliratát
   kimenti, majd OpenAI-kompatibilis helyi LLM-mel magyarra fordítja
   (időzítés változatlan; a kimenet ``subs/<EP>_hun.srt``);
2. **hangsáv** — a forrás eredeti hangsávját kimenti, normalizálja, majd az
   ``audio-separator`` UVR/MDX modellel leválasztja a vokált. A zene+effekt
   (M&E) sáv a maradék beszéd eltávolításával készül;
3. a kész sávokat a projekt által várt helyre teszi
   (``work/<EP>/vocals.wav``, ``work/<EP>/me.wav``), így a pipeline és a
   sorozatfigyelő változtatás nélkül tovább tud lépni.

A modul nem tölt TTS- vagy ASR-modellt. A fordítás csak akkor indul, ha a
magyar felirat még nincs meg; a szétválasztás csak a hiányzó vokál/M&E
esetén. A folyamat állapotát ``work/<EP>/intake.json`` rögzíti, hogy egy
hiba után a figyelő ne próbálkozzon másodpercenként újra.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import media as media_mod
from . import subs
from .config import APP_ROOT, Project

_TEXT_SUB_CODECS = {"subrip", "srt", "ass", "ssa", "mov_text", "webvtt", "text"}
_LANG_ALIASES = {
    "en": {"eng", "en", "english"},
    "eng": {"eng", "en", "english"},
    "he": {"heb", "he", "iw", "hebrew"},
    "heb": {"heb", "he", "iw", "hebrew"},
}

_DEFAULTS = {
    "translate": {
        "enabled": True,
        # Ollama és LM Studio is ezt az OpenAI-kompatibilis végpontot adja
        "base": "http://127.0.0.1:11434/v1",
        # a gondolkodó modellek (qwen3.5) az OpenAI-végponton a tokenkeret
        # nagy részét reasoning-re költik; a ministral gyorsan ad tiszta JSON-t
        "model": "ministral-3:8b",
        "api_key": "",
        "api_key_env": "",
        "batch": 20,
        "timeout": 300,
        "max_tokens": 4096,
        "source_lang": "",
    },
    "separate": {
        "enabled": True,
        "command": "",
        "model": "UVR_MDXNET_Main.onnx",
        "model_dir": "",
        "audio_stream": None,
        "second_pass": True,
        "sample_rate": 44100,
        "timeout": 3600,
        "retry_s": 900,
    },
}


class IntakeError(RuntimeError):
    """Az intake egyik lépése nem tudott lefutni."""


def _deep_merge(base: dict, extra: dict) -> dict:
    out = dict(base)
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def settings(project: Project) -> dict:
    """A projekt ``intake`` szakasza az alapértékekkel kiegészítve."""
    return _deep_merge(_DEFAULTS, dict(project.intake or {}))


def _marker(project: Project, tag: str) -> Path:
    return project.work / tag.upper() / "intake.json"


def _write_marker(project: Project, tag: str, **values) -> dict:
    path = _marker(project, tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    old = {}
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    data = {**old, **values, "tag": tag.upper(),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8")
    tmp.replace(path)
    return data


def _read_marker(project: Project, tag: str) -> dict:
    try:
        return json.loads(_marker(project, tag).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _external_subtitle(project: Project, tag: str, cfg: dict) -> tuple[Path, str] | None:
    preferred = str(cfg.get("source_lang") or "").strip().lower()
    order = [preferred] if preferred else []
    order += ["eng", "heb", project.source_lang]
    seen = set()
    for lang in order:
        lang = str(lang or "").strip().lower()
        if not lang or lang in seen:
            continue
        seen.add(lang)
        path = project.subtitle(tag, lang)
        if path.is_file():
            return path, lang
    return None


def _language_matches(stream_lang: str, wanted: str) -> bool:
    got = str(stream_lang or "").strip().lower()
    if not wanted:
        return True
    return got in _LANG_ALIASES.get(wanted, {wanted})


def _embedded_subtitle(project: Project, tag: str, cfg: dict) -> tuple[Path, str] | None:
    """A forrás első szöveges feliratsávja (a kért nyelv előnyben)."""
    src = project.episode_media(tag)
    streams = media_mod.subtitle_streams(src)
    if not streams:
        return None
    wanted = [str(cfg.get("source_lang") or "").strip().lower(),
              "eng", "heb", project.source_lang]
    wanted = [w for w in dict.fromkeys(wanted) if w]
    chosen = None
    for lang in wanted:
        chosen = next(
            (s for s in streams
             if (s.get("codec_name") or "").lower() in _TEXT_SUB_CODECS
             and _language_matches((s.get("tags") or {}).get("language"), lang)),
            None)
        if chosen is not None:
            break
    if chosen is None:
        # nyelvjelölés nélküli, de szöveges sáv is jó
        chosen = next((s for s in streams
                       if (s.get("codec_name") or "").lower() in _TEXT_SUB_CODECS), None)
    if chosen is None:
        return None
    stream_lang = str((chosen.get("tags") or {}).get("language") or "").lower()
    lang = "eng" if stream_lang in _LANG_ALIASES["eng"] else \
        "heb" if stream_lang in _LANG_ALIASES["heb"] else \
        (wanted[0] if wanted else "und")
    project.subs_dir.mkdir(parents=True, exist_ok=True)
    out = project.subtitle(tag, lang)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-i", str(src), "-map", f"0:{chosen['index']}",
           "-c:s", "srt", str(out)]
    _run(cmd, timeout=120)
    if not out.is_file() or out.stat().st_size == 0:
        raise IntakeError(f"Nem sikerült feliratot kinyerni: {src}")
    return out, lang


def find_source_subtitle(project: Project, tag: str, cfg: dict | None = None,
                         extract: bool = True) -> tuple[Path, str] | None:
    """A fordítás forrása: külső SRT, majd beágyazott szöveges sáv."""
    cfg = cfg or settings(project)["translate"]
    external = _external_subtitle(project, tag, cfg)
    if external is not None:
        return external
    if not extract:
        return None
    try:
        return _embedded_subtitle(project, tag, cfg)
    except FileNotFoundError:
        # nincs még forrásmédia: ez nem hiba, csak hiányzó felirat
        return None


def _run(cmd: list[str], timeout: float | None = 300,
         env: dict | None = None) -> None:
    try:
        proc = subprocess.run([str(c) for c in cmd], capture_output=True,
                              text=True, encoding="utf-8", errors="replace",
                              timeout=timeout, env=env)
    except FileNotFoundError as exc:
        raise IntakeError(f"Hiányzó eszköz: {cmd[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise IntakeError(f"Az eszköz túllépte az időkorlátot: {cmd[0]}") from exc
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-6:]
        raise IntakeError(f"{Path(str(cmd[0])).name} hiba: " + " | ".join(tail))


def _separator_env() -> dict | None:
    """Windows ARM64-en nincs torchvision kerék; az onnx2torch importjához
    adunk egy minimális shim-et, de csak akkor, ha valódi torchvision nincs."""
    import importlib.util

    if importlib.util.find_spec("torchvision") is not None:
        return None
    env = os.environ.copy()
    shim = Path(__file__).with_name("_shims")
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(shim) + (os.pathsep + current if current else "")
    return env


def _endpoint(base: str) -> str:
    value = str(base or "").strip().rstrip("/")
    if not value:
        raise IntakeError("A fordító LLM alap-URL-je üres.")
    if value.endswith("/chat/completions"):
        return value
    if value.endswith("/v1"):
        return value + "/chat/completions"
    return value + "/v1/chat/completions"


_LINE_PAIR = re.compile(
    r'\{\s*"id"\s*:\s*(\d+)\s*,\s*"text"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}')


def _salvage_lines(content: str) -> list[dict]:
    """Hibás JSON-ból a kész ``{"id","text"}`` párok kimentése."""
    out: list[dict] = []
    for raw_id, raw_text in _LINE_PAIR.findall(content):
        try:
            text = json.loads(f'"{raw_text}"')
        except ValueError:
            continue
        out.append({"id": int(raw_id), "text": text})
    return out


def _chat_json(cfg: dict, system: str, prompt: str) -> dict:
    base = str(cfg.get("base") or "").strip()
    model = str(cfg.get("model") or "").strip()
    if not model:
        raise IntakeError("A fordító LLM modellje nincs beállítva.")
    payload = {
        "model": model,
        "temperature": 0,
        "stream": False,
        "max_tokens": int(cfg.get("max_tokens") or 4096),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }
    headers = {"Content-Type": "application/json"}
    key = str(cfg.get("api_key") or "").strip()
    if not key:
        # A kulcs a környezetből is jöhet (``api_key_env``) — így nem kerül
        # titok a projektfájlba, és a projekt megosztható marad.
        env_name = str(cfg.get("api_key_env") or "").strip()
        if env_name:
            key = os.environ.get(env_name, "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(
        _endpoint(base), data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=float(cfg.get("timeout") or 300)) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        raise IntakeError(f"Fordító HTTP {exc.code}: {detail[-500:]}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise IntakeError(
            f"A fordító LLM nem érhető el ({base}): {exc}. "
            "Indítsd el az Ollamát/LM Studiót, vagy állítsd be az intake.translate.base-et."
        ) from exc
    try:
        data = json.loads(raw)
        content = data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise IntakeError("A fordító LLM válasza nem OpenAI-kompatibilis.") from exc
    if isinstance(content, list):
        content = "".join(str(part.get("text") or "") for part in content
                           if isinstance(part, dict))
    content = str(content or "").strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
        content = re.sub(r"\s*```$", "", content)
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end <= start:
        raise IntakeError("A fordító LLM nem JSON-t adott vissza.")
    try:
        parsed = json.loads(content[start:end + 1])
    except ValueError as exc:
        # A nagyobb modellek (pl. Gemini) néha csonkán/markdownnal adják a
        # JSON-t. A kész sorpárokat kimentjük — a hiányzókat a hívó újra kéri.
        salvaged = _salvage_lines(content)
        if salvaged:
            return {"lines": salvaged}
        raise IntakeError(f"A fordító LLM JSON-ja hibás: {exc}") from exc
    if not isinstance(parsed, dict):
        raise IntakeError("A fordító LLM válasza nem JSON-objektum.")
    return parsed


_SYSTEM = """You are a professional audiovisual subtitle translator.
Translate the supplied dialogue into natural, spoken Hungarian for dubbing.
Keep the meaning, names, punctuation and speaker dashes (leading \"- \" ).
Some lines start with a [Xs] target slot: that is how long the original cue
lasts. Keep the Hungarian line speakable in about that time (roughly 14
characters per second): prefer shorter synonyms and drop filler words rather
than the meaning. Never merge, split or reorder lines.
Do not add explanations, notes, timestamps or line breaks inside a line.
Return ONLY JSON in this shape:
{\"lines\":[{\"id\":1,\"text\":\"magyar szöveg\"}]}
Every input id must appear exactly once, in the same order."""

#: Hungarian TTS speaks roughly this many characters per second.  Used only as
#: guidance for the translator and as a soft over-length report, never as a
#: hard cut: the plan/synthesis step still measures the real duration.
HU_CHARS_PER_SECOND = 14.0
HU_TIGHT_SLACK = 6


def _slot_hint(seconds: float | None) -> str:
    if seconds is None or seconds <= 0:
        return ""
    return f"[{seconds:.1f}s] "


_SLOT_ECHO = re.compile(r"^\s*\[\d+(?:[.,]\d+)?\s*s\]\s*", re.IGNORECASE)


def _clean_translation(text: object) -> str:
    """Normalise a returned line and drop an echoed slot hint.

    The model occasionally copies the prompt's ``[2.5s]`` marker into the
    translation; it is guidance, never dialogue, so it must not reach the SRT
    or the TTS.
    """
    line = " ".join(str(text or "").split())
    return _SLOT_ECHO.sub("", line).strip()


def _fit_warning(lines: dict[int, str], limits: dict[int, float]) -> list[int]:
    """Cue ids whose translation is likely too long for the original slot."""
    tight: list[int] = []
    for idx, text in lines.items():
        seconds = limits.get(idx)
        if not seconds or seconds <= 0:
            continue
        budget = seconds * HU_CHARS_PER_SECOND + HU_TIGHT_SLACK
        if len(text) > budget:
            tight.append(idx)
    return sorted(tight)


def _translate_batch(cfg: dict, rows: list[tuple[int, str]], source_lang: str,
                     limits: dict[int, float] | None = None) -> dict[int, str]:
    limits = limits or {}
    numbered = "\n".join(f"{i}. {_slot_hint(limits.get(i))}{text}" for i, text in rows)
    prompt = (f"Source language: {source_lang or 'unknown'}\n"
              "Translate every numbered subtitle line to Hungarian.\n\n" + numbered)
    parsed = _chat_json(cfg, _SYSTEM, prompt)
    result: dict[int, str] = {}
    for item in parsed.get("lines") or []:
        try:
            idx = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        text = _clean_translation(item.get("text"))
        if idx in {i for i, _ in rows} and text:
            result[idx] = text
    missing = [(i, text) for i, text in rows if i not in result]
    if missing:
        # kisebb, csak a hiányzó sorokat tartalmazó második kérés
        prompt = ("These subtitle lines are still missing from the translation. "
                  "Translate only these and return the same JSON shape.\n\n"
                  + "\n".join(f"{i}. {_slot_hint(limits.get(i))}{text}"
                              for i, text in missing))
        retry = _chat_json(cfg, _SYSTEM, prompt)
        for item in retry.get("lines") or []:
            try:
                idx = int(item.get("id"))
            except (TypeError, ValueError):
                continue
            text = _clean_translation(item.get("text"))
            if idx in {i for i, _ in missing} and text:
                result[idx] = text
    # utolsó próba soronként: egyetlen makacs sor (többnyire egy hosszú,
    # szűk időrésbe szorított cue) miatt ne álljon meg az egész film
    for idx, text in [(i, t) for i, t in rows if i not in result]:
        try:
            single = _chat_json(
                cfg, _SYSTEM,
                f"Source language: {source_lang}\n"
                "Translate this single subtitle line to Hungarian.\n\n"
                f"{idx}. {_slot_hint(limits.get(idx))}{text}")
        except IntakeError:
            continue
        for item in single.get("lines") or []:
            try:
                got = int(item.get("id"))
            except (TypeError, ValueError):
                continue
            value = _clean_translation(item.get("text"))
            if got == idx and value:
                result[idx] = value
    still = [i for i, _ in rows if i not in result]
    if still:
        raise IntakeError("A fordító kihagyta ezeket a feliratsorokat: "
                          + ", ".join(map(str, still[:12])))
    return result


def translate_subtitles(project: Project, tag: str, *, force: bool = False,
                        cfg: dict | None = None,
                        progress=None) -> dict:
    """Magyar SRT készítése a forrásfeliratból (időzítés megtartásával)."""
    cfg = cfg or settings(project)["translate"]
    if not cfg.get("enabled", True):
        raise IntakeError("A feliratfordítás ki van kapcsolva (intake.translate.enabled).")
    out = project.subtitle(tag, "hun")
    if out.is_file() and not force and out.stat().st_size > 0:
        return {"subtitle": str(out), "translated": 0, "skipped": True,
                "source": "", "source_lang": ""}
    found = find_source_subtitle(project, tag, cfg)
    if found is None:
        raise IntakeError(
            f"Nincs fordítható felirat ehhez: {tag}. Tegyél egy "
            f"subs/{tag.upper()}_eng.srt vagy _heb.srt fájlt a projektbe, "
            "vagy legyen a mkv-ban szöveges (nem kép) feliratsáv.")
    source, source_lang = found
    cues = subs.parse_srt(source)
    if not cues:
        raise IntakeError(f"A felirat üres vagy nem SRT: {source}")
    batch = max(1, int(cfg.get("batch") or 20))
    limits = {c.index: max(0.0, float(c.end) - float(c.start)) for c in cues}
    translated: dict[int, str] = {}
    total = len(cues)
    for start in range(0, total, batch):
        part = cues[start:start + batch]
        rows = [(c.index, subs.one_line(c.text)) for c in part]
        translated.update(_translate_batch(cfg, rows, source_lang, limits))
        done = min(total, start + len(part))
        print(f"[translate] {tag} {done}/{total} sor "
              f"({100 * done / max(1, total):.0f}%)", flush=True)
        if progress:
            progress(done, total)
    project.subs_dir.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".srt.tmp")
    _write_srt(tmp, [(c, translated[c.index]) for c in cues])
    tmp.replace(out)
    tight = _fit_warning(translated, limits)
    _write_marker(project, tag, translated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                  subtitle_source=str(source), subtitle_lang=source_lang,
                  translated_lines=total, tight_lines=len(tight))
    return {"subtitle": str(out), "translated": total, "skipped": False,
            "source": str(source), "source_lang": source_lang,
            "tight_lines": len(tight), "tight_ids": tight[:20]}


def _ts(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _wrap(text: str, width: int = 42, max_lines: int = 2) -> list[str]:
    words = str(text or "").split()
    if not words:
        return [""]
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
            if len(lines) == max_lines - 1:
                break
    if current and len(lines) < max_lines:
        lines.append(current)
    consumed = sum(len(line.split()) for line in lines)
    if consumed < len(words) and lines:
        lines[-1] = " ".join(lines[-1].split() + words[consumed:])
    return lines


def _write_srt(path: Path, rows: list[tuple[subs.Cue, str]]) -> None:
    blocks = []
    for number, (cue, text) in enumerate(rows, start=1):
        body = "\n".join(_wrap(text))
        blocks.append(f"{number}\n{_ts(cue.start)} --> {_ts(cue.end)}\n{body}")
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def _separator_command(project: Project, cfg: dict) -> Path:
    raw = str(cfg.get("command") or "").strip()
    candidates = []
    if raw:
        p = Path(raw).expanduser()
        candidates += [p] if p.is_absolute() else [project.root / p, APP_ROOT / p]
    candidates += [
        APP_ROOT.parent / "reader" / ".venv" / "Scripts" / "audio-separator.exe",
        APP_ROOT.parent / "reader" / ".venv" / "bin" / "audio-separator",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    found = shutil.which("audio-separator")
    if found:
        return Path(found)
    raise IntakeError(
        "Nincs audio-separator. Telepítsd az Auris venv-jébe, vagy állítsd be "
        "az intake.separate.command értékét.")


def _separator_model_dir(project: Project, cfg: dict) -> Path:
    raw = str(cfg.get("model_dir") or project.models.get("separator_model_dir") or "").strip()
    if raw:
        p = Path(raw).expanduser()
        return p if p.is_absolute() else (project.root / p).resolve()
    return (project.root / "models").resolve()


def _extract_audio(project: Project, tag: str, cfg: dict) -> Path:
    src = project.episode_media(tag)
    ordinal = cfg.get("audio_stream")
    if ordinal in (None, ""):
        ordinal = project.original_audio
    try:
        ordinal = int(ordinal)
    except (TypeError, ValueError):
        ordinal = 0
    out = project.work / tag.upper() / "intake" / "source.wav"
    if out.is_file() and out.stat().st_size > 1000:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    media_mod.extract_audio(src, ordinal, out,
                           sr=int(cfg.get("sample_rate") or 44100))
    return out


def _normalize(src: Path, target: Path) -> Path:
    """Csúcsnormalizálás ffmpeg volume szűrővel (a teljes sáv memóriába
    emelése nélkül). Hiba esetén a forrást adja vissza."""
    if target.is_file() and target.stat().st_size > 1000:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    probe = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(src),
         "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    match = re.search(r"max_volume:\s*(-?\d+(?:\.\d+)?)\s*dB", probe.stderr or "")
    if not match:
        shutil.copy2(src, target)
        return target
    peak_db = float(match.group(1))
    # 0,9 lineáris csúcs = -0,915 dBFS
    gain_db = -0.915 - peak_db
    if abs(gain_db) < 0.05:
        shutil.copy2(src, target)
    else:
        _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
              "-i", str(src), "-af", f"volume={gain_db:.3f}dB",
              "-c:a", "pcm_s16le", str(target)])
    return target


def _copy_or_link(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file():
        return dst
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return dst


def _subtract_beds(inst1: Path, voc2: Path, out: Path) -> Path:
    """M&E = instrumental1 - vocals2, blokkonként (nagy fájlhoz is)."""
    import soundfile as sf

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part.wav")
    peak = 0.0
    with sf.SoundFile(str(inst1)) as a, sf.SoundFile(str(voc2)) as b:
        sr, channels = a.samplerate, a.channels
        with sf.SoundFile(str(tmp), "w", samplerate=sr, channels=channels,
                          subtype="FLOAT") as dst:
            while True:
                x = a.read(262144, dtype="float32", always_2d=True)
                if len(x) == 0:
                    break
                y = b.read(len(x), dtype="float32", always_2d=True)
                n = min(len(x), len(y))
                if n == 0:
                    break
                z = x[:n] - y[:n]
                peak = max(peak, float(abs(z).max(initial=0.0)))
                dst.write(z)
    if peak > 0.99:
        scaled = out.with_suffix(".scale.wav")
        with sf.SoundFile(str(tmp)) as src, sf.SoundFile(
                str(scaled), "w", samplerate=src.samplerate, channels=src.channels,
                subtype="FLOAT") as dst:
            gain = 0.99 / peak
            while True:
                chunk = src.read(262144, dtype="float32", always_2d=True)
                if len(chunk) == 0:
                    break
                dst.write(chunk * gain)
        tmp.unlink(missing_ok=True)
        scaled.replace(out)
    else:
        tmp.replace(out)
    return out


def _run_separator(project: Project, tag: str, source: Path, outdir: Path,
                   cfg: dict) -> tuple[Path, Path]:
    vocals = sorted(outdir.glob("*Vocals*.wav"))
    inst = sorted(outdir.glob("*Instrumental*.wav"))
    if vocals and inst:
        return vocals[0], inst[0]
    outdir.mkdir(parents=True, exist_ok=True)
    command = _separator_command(project, cfg)
    model = str(cfg.get("model") or "UVR_MDXNET_Main.onnx")
    model_dir = _separator_model_dir(project, cfg)
    if not (model_dir / model).is_file():
        # az audio-separator néha a modellnevet .onnx nélkül várja
        if not (model_dir / f"{model}.onnx").is_file():
            raise IntakeError(f"Nincs szeparációs modell: {model_dir / model}")
    cmd = [str(command), "-m", model, "--model_file_dir", str(model_dir),
           "--output_dir", str(outdir), "--output_format", "wav",
           "--log_level", "warning", "--sample_rate",
           str(int(cfg.get("sample_rate") or 44100)), str(source)]
    print(f"[intake] szétválasztás: {' '.join(cmd)}", flush=True)
    _run(cmd, timeout=float(cfg.get("timeout") or 3600), env=_separator_env())
    vocals = sorted(outdir.glob("*Vocals*.wav"))
    inst = sorted(outdir.glob("*Instrumental*.wav"))
    if not (vocals and inst):
        raise IntakeError(f"A szétválasztás nem adott Vocals/Instrumental sávot: {outdir}")
    return vocals[0], inst[0]


def separate_audio(project: Project, tag: str, *, force: bool = False,
                   cfg: dict | None = None, progress=None) -> dict:
    """Forrás hang → vocals.wav + me.wav (opcionális második körrel)."""
    cfg = cfg or settings(project)["separate"]
    if not cfg.get("enabled", True):
        raise IntakeError("A hangsáv-szétválasztás ki van kapcsolva.")
    tag = tag.upper()
    work = project.work / tag
    vocals_out = work / "vocals.wav"
    me_out = work / "me.wav"
    if not force:
        # a régi projekteknél a projekt glob-mintái (pl. work/E01_me1/*Vocals*)
        # is érvényesek; ne válasszuk szét újra a már kész sávokat
        try:
            old_vocals = project.vocals(tag)
            old_me = project.me_bed(tag)
        except (FileNotFoundError, OSError):
            old_vocals = old_me = None
        if old_vocals is not None and old_me is not None \
                and old_vocals.is_file() and old_me.is_file():
            return {"vocals": str(old_vocals), "me": str(old_me),
                    "skipped": True, "second_pass": False}

    source = _extract_audio(project, tag, cfg)
    if progress:
        progress("normalizálás", 0, 1)
    normalized = _normalize(source, source.with_name("source_norm.wav"))
    sep1 = work / "intake" / "sep1"
    if progress:
        progress("vokál szétválasztása", 0, 2)
    voc1, inst1 = _run_separator(project, tag, normalized, sep1, cfg)
    _copy_or_link(voc1, vocals_out)

    second = bool(cfg.get("second_pass", True))
    if second:
        if progress:
            progress("maradék beszéd eltávolítása", 1, 2)
        sep2 = work / "intake" / "sep2"
        _voc2, _inst2 = _run_separator(project, tag, inst1, sep2, cfg)
        voc2 = sorted(sep2.glob("*Vocals*.wav"))[0]
        _subtract_beds(inst1, voc2, me_out)
    else:
        _copy_or_link(inst1, me_out)
    if progress:
        progress("kész", 2, 2)
    _write_marker(project, tag, separated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                  separator_model=str(cfg.get("model") or ""),
                  second_pass=second)
    return {"vocals": str(vocals_out), "me": str(me_out), "skipped": False,
            "second_pass": second}


def status(project: Project, tag: str) -> dict:
    """Az intake állapota a könyvtárhoz (nem futtat ffprobe-ot)."""
    tag = tag.upper()
    cfg_t = settings(project)["translate"]
    source = _external_subtitle(project, tag, cfg_t)
    hun = project.subtitle(tag, "hun")
    try:
        vocals = project.vocals(tag)
        vocals_ok = vocals.is_file()
    except (FileNotFoundError, OSError):
        vocals = None
        vocals_ok = False
    try:
        me = project.me_bed(tag)
        me_ok = me.is_file()
    except (FileNotFoundError, OSError):
        me = None
        me_ok = False
    marker = _read_marker(project, tag)
    error = str(marker.get("last_error") or "")
    updated = str(marker.get("updated_at") or "")
    retry_s = int(settings(project)["separate"].get("retry_s") or 900)
    retry_at = 0.0
    if error and updated:
        try:
            retry_at = time.mktime(time.strptime(updated, "%Y-%m-%d %H:%M:%S")) + retry_s
        except ValueError:
            retry_at = 0.0
    state = "ready" if (hun.is_file() and vocals_ok and me_ok) else (
        "error" if error else ("partial" if (hun.is_file() or vocals_ok or me_ok)
                               else "missing"))
    return {
        "tag": tag,
        "source": str(source[0]) if source else None,
        "source_lang": source[1] if source else None,
        "subtitle": str(hun) if hun.is_file() else None,
        "subtitle_ready": hun.is_file(),
        "vocals": str(vocals) if vocals_ok else None,
        "me": str(me) if me_ok else None,
        "separated": bool(vocals_ok and me_ok),
        "state": state,
        "error": error,
        "updated_at": updated,
        "retry_allowed": (not error) or time.time() >= retry_at,
    }


def ensure_ready(project: Project, tag: str, *, translate: bool = True,
                 separate: bool = True, force: bool = False,
                 progress=None) -> dict:
    """A hiányzó feliratot és hangsávokat pótolja; hibát összegyűjt."""
    errors: list[str] = []
    result = {"tag": tag.upper(), "subtitle": None, "audio": None, "errors": errors}
    cfg = settings(project)
    if translate:
        try:
            result["subtitle"] = translate_subtitles(
                project, tag, force=force, cfg=cfg["translate"], progress=progress)
        except Exception as exc:             # noqa: BLE001 — a másik lépés még futhat
            errors.append(f"felirat: {exc}")
    if separate:
        try:
            result["audio"] = separate_audio(
                project, tag, force=force, cfg=cfg["separate"], progress=progress)
        except Exception as exc:             # noqa: BLE001
            errors.append(f"hang: {exc}")
    if errors:
        _write_marker(project, tag, last_error=" | ".join(errors))
        raise IntakeError(" | ".join(errors))
    _write_marker(project, tag, last_error="", completed_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    return result


def should_retry(project: Project, tag: str) -> bool:
    """A figyelő csak a retry_s letelte után próbálja újra a hibás intake-et."""
    return bool(status(project, tag).get("retry_allowed", True))


if __name__ == "__main__":                     # pragma: no cover
    print(__doc__)
    raise SystemExit(0)
