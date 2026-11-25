"""Experimental, read-only source-speaker audit for one short episode excerpt.

This is deliberately independent of ``prepare/cluster/dub``: it writes only
under ``work/<EP>/source_audit/`` and never changes production cast or outputs.
It separates (a) Hebrew subtitle alignment, (b) diarization turns, and (c) CAM++
voice similarity against current E06 utterance clusters. A cluster similarity
is evidence for human review, not an identity decision.

Example (Windows ARM64 laptop, local inference):
  python -m dubstudio speaker-audit E06 --start 30 --duration 45

Model download is an explicit separate action; the Hebrew audio stays local.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import audio, subs
from .config import Project

#: Hugging Face repo of the local Hebrew char-CTC aligner (Apache-2.0).
ALIGNER_REPO = "XDcobra/wav2vec2-large-xlsr-53-hebrew-ONNX"
ALIGN_MODEL_FILES = ("onnx/model_int8.onnx", "onnx/model_fp16.onnx", "onnx/model.onnx")
ALIGN_SR = 16000

#: Angol szóidőhöz használt ASR (CC-BY-4.0, ONNX, CPU): szavankénti idővel jön.
ASR_EN_MODEL = "nemo-parakeet-tdt-0.6b-v3"


LANG_CODES = {"he": {"heb", "he"}, "en": {"eng", "en"}}


def subtitle_stream_index(project: Project, tag: str, lang: str = "he",
                          prefer_title: str | None = None) -> tuple[Path, int]:
    """A kiválasztott nyelvű feliratsáv (opcionálisan cím szerint előnyben)."""
    src = project.episode_media(tag)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "s",
         "-show_entries", "stream=index:stream_tags=language,title", "-of", "json", str(src)],
        capture_output=True, text=True, check=True)
    streams = json.loads(probe.stdout).get("streams", [])
    codes = LANG_CODES.get(lang, {lang.lower()})
    matches = [s for s in streams
               if (s.get("tags") or {}).get("language", "").lower() in codes]
    if not matches:
        raise FileNotFoundError(f"Nincs {lang} feliratsáv a forrásban: {src}")
    if prefer_title:
        want = prefer_title.lower()
        preferred = [s for s in matches
                     if want in ((s.get("tags") or {}).get("title") or "").lower()]
        if preferred:
            matches = preferred
    return src, int(matches[0]["index"])


def extract_subtitles(project: Project, tag: str, out: Path, lang: str = "he",
                      prefer_title: str | None = None) -> Path:
    """Egy feliratsáv kiemelése SRT-be (a forrás nem módosul)."""
    src, index = subtitle_stream_index(project, tag, lang, prefer_title)
    key = hashlib.sha256(f"{src.resolve()}|{src.stat().st_size}|{index}|srt".encode()).hexdigest()
    key_path = out.with_suffix(".key")
    if out.exists() and key_path.exists() and key_path.read_text(encoding="ascii") == key:
        return out
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(src), "-map", f"0:{index}",
                    "-c:s", "srt", "-y", str(out)], check=True)
    key_path.write_text(key, encoding="ascii")
    return out


def extract_hebrew_subtitles(project: Project, tag: str, out: Path) -> Path:
    """Extract the first Hebrew subtitle stream to SRT (no source modifications)."""
    return extract_subtitles(project, tag, out, lang="he")


def resolve_source_audio_stream(project: Project, tag: str,
                                stream: int | None = None,
                                lang: str | None = None) -> tuple[Path, int]:
    """Return the source MKV and selected global audio stream index.

    ``lang`` nyelv szerint választ (pl. ``"en"`` a fő hanghoz), ha nincs
    kifejezett sorszám; a kiválasztás mindig kiíródik a riportba.
    """
    src = project.episode_media(tag)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index:stream_tags=language,title", "-of", "json", str(src)],
        capture_output=True, text=True, check=True)
    tracks = json.loads(probe.stdout).get("streams", [])
    if not tracks:
        raise FileNotFoundError(f"Nincs hangsáv a forrásban: {src}")
    if stream is not None:
        selected = next((s for s in tracks if int(s["index"]) == stream), None)
        if selected is None:
            raise ValueError(f"Nincs {stream}. hangsáv: {src}")
        return src, int(selected["index"])
    if lang:
        codes = LANG_CODES.get(lang, {lang.lower()})
        wanted = [s for s in tracks
                  if (s.get("tags") or {}).get("language", "").lower() in codes]
        if not wanted:
            raise FileNotFoundError(f"Nincs {lang} hangsáv a forrásban: {src}")
        return src, int(wanted[0]["index"])
    return src, int(tracks[0]["index"])


def extract_source_audio(project: Project, tag: str, out: Path,
                         start: float, duration: float, stream: int | None = None,
                         lang: str | None = None) -> Path:
    """Extract a mono 16 kHz excerpt from a selected original audio stream."""
    src, selected_index = resolve_source_audio_stream(project, tag, stream, lang)
    key = hashlib.sha256(
        f"{src.resolve()}|{src.stat().st_size}|{start:.3f}|{duration:.3f}|{selected_index}|mono16k".encode()
    ).hexdigest()
    key_path = out.with_suffix(".key")
    if out.exists() and key_path.exists() and key_path.read_text(encoding="ascii") == key:
        return out
    subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-i", str(src),
                    "-t", f"{duration:.3f}", "-map", f"0:{selected_index}",
                    "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-y", str(out)],
                   check=True)
    key_path.write_text(key, encoding="ascii")
    return out


def normalize_hebrew(text: str) -> str:
    """Drop subtitle markup/SDH labels while retaining Hebrew words and punctuation."""
    t = re.sub(r"\{[^}]*\}", " ", text or "")
    t = re.sub(r"<[^>]*>", " ", t)
    t = re.sub(r"\[[^]]*\]", " ", t)
    t = re.sub(r"\([^)]*\)", " ", t)
    t = t.replace("\u200f", " ").replace("\u200e", " ").replace("\u202b", " ")
    t = t.replace("\u202c", " ").replace("\u2066", " ").replace("\u2069", " ")
    t = re.sub(r"\s+", " ", t).strip()
    # Exclude non-dialogue captions and cues with no Hebrew-script content.
    return t if re.search(r"[\u0590-\u05ff]", t) else ""


def normalize_english(text: str) -> str:
    """Angol felirat: markup és SDH-jelölés kivétele, betűs tartalom megőrzése."""
    t = re.sub(r"\{[^}]*\}", " ", text or "")
    t = re.sub(r"<[^>]*>", " ", t)
    t = re.sub(r"\[[^]]*\]", " ", t)
    t = re.sub(r"\([^)]*\)", " ", t)
    t = t.replace("\u200e", " ").replace("\u200f", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t if re.search(r"[A-Za-z]", t) else ""


def subtitle_segments(path: Path, start: float, end: float,
                      lang: str = "he") -> list[dict]:
    """Convert subtitle cues to WhisperX-like segments in excerpt time.

    An SDH speaker label such as ``[מיראן]`` is kept as ``sdh_label`` for
    human review; the label itself never enters the aligned text.
    """
    clean = normalize_english if lang == "en" else normalize_hebrew
    out = []
    for cue in subs.parse_srt(path):
        if cue.end <= start or cue.start >= end:
            continue
        text = clean(cue.text)
        if text:
            m = re.search(r"\[([^\]]+)\]", cue.text or "")
            out.append({"id": len(out), "source_cue": cue.index,
                        "start": max(0.0, cue.start - start),
                        "end": min(end - start, cue.end - start), "text": text,
                        "sdh_label": m.group(1).strip() if m else None})
    return out


def tokens_to_words(tokens: list[str], timestamps: list[float]) -> list[dict]:
    """Parakeet token-ek (szókezdő szóközzel) -> szavak kezdő/vég idővel.

    A tokenek szubszavak: ``" qu" + "iet"`` egy szó, a szókezdő szóköz jelzi
    az új szót. A szó vége a következő szó kezdete (a TDT időbélyegei
    szókezdetek), az utolsó szóé az utolsó token + 0.12 s.
    """
    words: list[dict] = []
    cur: dict | None = None
    for tok, ts in zip(tokens or [], timestamps or []):
        t = str(tok)
        if t.startswith(" ") and cur is not None:
            words.append(cur)
            cur = None
        if cur is None:
            cur = {"word": t.strip(), "start": float(ts), "end": float(ts)}
        else:
            cur["word"] += t
        cur["end"] = float(ts)
    if cur is not None:
        words.append(cur)
    # A szó vége az utolsó tokenje + rövid tartalék — **nem** a következő szó
    # kezdete: egy hosszú csend/zene különben 30+ másodperces "szót" csinálna.
    for i, w in enumerate(words):
        w["start"] = round(w["start"], 3)
        end = min(w["end"] + 0.12, w["start"] + 3.0)
        nxt = words[i + 1]["start"] if i + 1 < len(words) else None
        if nxt is not None and nxt < end:
            end = nxt - 0.01
            if end <= w["start"]:
                end = w["start"] + 0.01
        w["end"] = round(end, 3)
        if w["end"] <= w["start"]:
            w["end"] = round(w["start"] + 0.06, 3)
    return [w for w in words if w["word"]]


def _asr_words_english(wav: Path, model_name: str = ASR_EN_MODEL) -> list[dict]:
    """Angol szavak idővel a helyi ONNX ASR-ból (nincs felirat-illesztés)."""
    import onnx_asr

    model = onnx_asr.load_model(model_name, quantization="int8").with_timestamps()
    res = model.recognize(str(wav), language="en")
    return tokens_to_words(list(getattr(res, "tokens", []) or []),
                           list(getattr(res, "timestamps", []) or []))


def turn_segment_ids(turns: list[dict], words: list[dict],
                     margin_s: float = 0.08) -> list[dict]:
    """Assign per-word speaker only on a unique temporal overlap; retain ambiguity."""
    output = []
    for w in words:
        a, b = w.get("start"), w.get("end")
        if a is None or b is None:
            output.append({**w, "speaker": None, "speaker_confidence": "no_word_timing"})
            continue
        scores = defaultdict(float)
        for t in turns:
            lo = max(float(a), float(t["start"]) - margin_s)
            hi = min(float(b), float(t["end"]) + margin_s)
            if hi > lo:
                scores[str(t["speaker"])]+=hi-lo
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if not ranked:
            sid, confidence = None, "no_turn_overlap"
        elif len(ranked) > 1 and ranked[1][1] >= 0.2 * max(ranked[0][1], 1e-9):
            # A meaningful second speaker overlap is uncertainty even if one
            # dominates the word duration; never silently collapse crosstalk.
            sid, confidence = None, "ambiguous_overlap"
        else:
            sid = ranked[0][0]
            confidence = "turn_overlap_only"
        output.append({**w, "speaker": sid, "speaker_confidence": confidence})
    return output


def compare_cam_clusters(turns: list[dict], words: list[dict], wav_path: Path,
                         utterances_path: Path, embedding_model: Path,
                         min_turn_s: float = 0.7,
                         prototypes: dict | None = None) -> list[dict]:
    """Compare clean-ish diarized turn CAM++ embeddings with source clusters.

    This remains a candidate list, never an automatic cast remap. ``prototypes``
    may be supplied by the caller so the turns are embedded only once.
    """
    import numpy as np
    from .embeddings import cosine
    from . import identity

    cluster_vecs: dict[str, list] = defaultdict(list)
    for utt in json.loads(utterances_path.read_text(encoding="utf-8")):
        if utt.get("cluster") is None or not utt.get("emb"):
            continue
        cluster_vecs[str(utt["cluster"])].append(np.asarray(utt["emb"], dtype=np.float32))
    cluster_protos = {}
    for cid, vectors in cluster_vecs.items():
        v = np.mean(np.stack(vectors), axis=0)
        cluster_protos[cid] = v / max(float(np.linalg.norm(v)), 1e-12)

    protos = prototypes if prototypes is not None else identity.prototypes_from_turns(
        turns, wav_path, embedding_model, min_turn_s=min_turn_s)

    candidates = []
    for sid, proto in sorted(protos.items()):
        vec = proto["vector"]
        ranked = sorted(((float(cosine(vec, cproto)), cid)
                         for cid, cproto in cluster_protos.items()), reverse=True)[:5]
        candidates.append({"speaker": sid, "evidence_turns": proto["turns"],
                           "evidence_seconds": proto["seconds"],
                           "candidate_clusters": [{"cluster": cid, "cosine": round(score, 4)}
                                                  for score, cid in ranked],
                           "status": "review_required"})
    return sorted(candidates, key=lambda c: c["speaker"])


def _find_align_model(project: Project, tag: str, explicit: Path | None) -> Path:
    """Locate the downloaded Hebrew char-CTC ONNX pack (shared across episodes)."""
    # Csak **kifejezetten kért** modell: nincs automatikus héber útvonal, és
    # a projekt nem használ héber illesztőt (a héber kísérlet kivéve).
    candidates = [Path(explicit)] if explicit is not None else []
    for c in candidates:
        if (c / "vocab.json").is_file() and _align_model_file(c, strict=False) is not None:
            return c
    raise FileNotFoundError(
        "Nincs héber forced-alignment ONNX modell. Töltsd le egyszer: "
        f"huggingface_hub.hf_hub_download('{ALIGNER_REPO}', 'onnx/model_int8.onnx', "
        "local_dir=<mappa>) a vocab.json / config.json fájlokkal együtt; "
        f"keresett helyek: {', '.join(str(c) for c in candidates)}")


def _align_model_file(model_dir: Path, strict: bool = True) -> Path | None:
    for name in ALIGN_MODEL_FILES:
        p = model_dir / name
        if p.is_file():
            return p
    if strict:
        raise FileNotFoundError(f"Nincs ONNX modellfájl ebben a csomagban: {model_dir}")
    return None


def tokenize_hebrew(text: str, vocab: dict) -> tuple[list[tuple[int, int]], list[str]]:
    """Map subtitle text to CTC ids + word groups.

    Returns ``(items, words)`` where each item is ``(ctc_id, word_index)`` and
    ``word_index`` is ``-1`` for a word separator. Characters outside the model
    vocabulary (punctuation, niqqud, marks) are dropped, never invented.
    """
    words = (text or "").split()
    kept_per_word = []
    for part in words:
        n = 0
        for ch in part:
            if vocab.get(ch) is not None:
                n += 1
        kept_per_word.append(n)
    # Keep only characters in the model vocabulary; insert a word separator
    # between consecutive words that both produced at least one CTC target.
    items: list[tuple[int, int]] = []
    sep = vocab.get("|")
    for wi, part in enumerate(words):
        if not kept_per_word[wi]:
            continue
        for ch in part:
            idx = vocab.get(ch)
            if idx is not None:
                items.append((int(idx), wi))
        if wi + 1 < len(words) and kept_per_word[wi + 1] and sep is not None:
            items.append((int(sep), -1))
    return items, words


def ctc_viterbi(log_probs: np.ndarray, tokens: list[int], blank: int) -> np.ndarray:
    """Forced CTC alignment: best state path over blank-interleaved target.

    Transitions: stay, advance one target index, or skip one index only when
    that index is a blank and the surrounding tokens differ (never merge
    repeated letters). Returns the target index per frame.
    """
    target: list[int] = []
    for t in tokens:
        target.extend((blank, t))
    target.append(blank)
    tgt = np.asarray(target, dtype=np.int64)
    L = tgt.size
    T = log_probs.shape[0]
    if T < len(tokens):
        raise ValueError(f"Az ablak túl rövid a szöveghez: {T} keret < {len(tokens)} token")
    NEG = -1e30
    dp = np.full((T, L), NEG, dtype=np.float64)
    bp = np.zeros((T, L), dtype=np.int8)
    dp[0, 0] = log_probs[0, blank]
    if L > 1:
        dp[0, 1] = log_probs[0, tgt[1]]
    allow_skip = np.zeros(L, dtype=bool)
    allow_skip[2:] = (tgt[1:-1] == blank) & (tgt[2:] != tgt[:-2])
    for t in range(1, T):
        emit = log_probs[t, tgt]
        stay = dp[t - 1]
        adv = np.full(L, NEG)
        adv[1:] = dp[t - 1, :-1]
        skp = np.full(L, NEG)
        skp[2:] = np.where(allow_skip[2:], dp[t - 1, :-2], NEG)
        choice = np.where((adv >= stay) & (adv >= skp), 1,
                          np.where(skp >= stay, 2, 0)).astype(np.int8)
        dp[t] = emit + np.maximum(np.maximum(stay, adv), skp)
        bp[t] = choice
    end_states = (L - 1,) if L < 2 else (L - 1, L - 2)
    best_j = int(end_states[int(np.argmax(dp[T - 1, list(end_states)]))])
    if dp[T - 1, best_j] <= NEG / 2:
        raise ValueError("A CTC kényszerített illesztés nem talált érvényes utat")
    path = np.empty(T, dtype=np.int64)
    path[T - 1] = best_j
    j = best_j
    for t in range(T - 1, 0, -1):
        j = j - int(bp[t, j])
        path[t - 1] = j
    return path


def words_from_window(log_probs: np.ndarray, text: str, vocab: dict, blank: int,
                      offset_s: float = 0.0, source_cue: int | None = None,
                      frame_step_s: float = 0.02) -> list[dict]:
    """One alignment window -> word spans in absolute excerpt time."""
    items, parts = tokenize_hebrew(text, vocab)
    if not items:
        return []
    tokens = [i for i, _ in items]
    path = ctc_viterbi(log_probs, tokens, blank)
    spans: dict[int, tuple[int, int]] = {}
    scores: dict[int, float] = {}
    for i, (cid, _) in enumerate(items):
        pos = 2 * i + 1
        ts = np.flatnonzero(path == pos)
        if not ts.size:
            continue
        a, b = int(ts[0]), int(ts[-1]) + 1
        spans[i] = (a, b)
        scores[i] = float(log_probs[a:b, cid].mean())
    grouped: dict[int, list[int]] = defaultdict(list)
    for i, (_, wi) in enumerate(items):
        if wi >= 0 and i in spans:
            grouped[wi].append(i)
    out = []
    for wi in sorted(grouped):
        idxs = grouped[wi]
        a = min(spans[i][0] for i in idxs)
        b = max(spans[i][1] for i in idxs)
        conf = float(np.exp(np.mean([scores[i] for i in idxs])))
        out.append({"word": parts[wi],
                    "start": round(offset_s + a * frame_step_s, 3),
                    "end": round(offset_s + b * frame_step_s, 3),
                    "score": round(conf, 4), "source_cue": source_cue})
    return out


def _align_hebrew(wav: Path, segments: list[dict], model_dir: Path,
                  margin_s: float = 0.35) -> list[dict]:
    """Local Hebrew forced alignment: ONNX wav2vec2 char-CTC per cue window.

    Each cue gets its own audio window (cue time +/- margin, bounded by the
    midpoints toward neighbouring cues) so speech between cues never absorbs
    another cue's text. No ASR and no translation: the supplied Hebrew subtitle
    text is what gets timed.
    """
    import soundfile as sf
    import onnxruntime as ort

    model_file = _align_model_file(model_dir)
    vocab = json.loads((model_dir / "vocab.json").read_text(encoding="utf-8"))
    blank = int(vocab.get("[PAD]", vocab.get("<pad>")))
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(model_file), so, providers=["CPUExecutionProvider"])
    signal, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    signal = signal.mean(axis=1)
    if sr != ALIGN_SR:
        signal = audio.to_16k(signal, int(sr))
    total = float(signal.shape[0]) / ALIGN_SR
    words: list[dict] = []
    for i, seg in enumerate(segments):
        if not tokenize_hebrew(seg["text"], vocab)[0]:
            continue
        lo, hi = float(seg["start"]), float(seg["end"])
        prev_end = float(segments[i - 1]["end"]) if i else None
        if prev_end is not None and lo >= prev_end:
            lo = max(lo - margin_s, 0.5 * (prev_end + lo))
        else:
            lo = max(0.0, lo - margin_s)
        if i + 1 < len(segments) and hi <= float(segments[i + 1]["start"]):
            hi = min(hi + margin_s, 0.5 * (hi + float(segments[i + 1]["start"])))
        else:
            hi = min(total, hi + margin_s)
        a, b = int(lo * ALIGN_SR), int(hi * ALIGN_SR)
        if b - a < 64:
            continue
        x = signal[a:b]
        x = (x - x.mean()) / (x.std() + 1e-7)
        logits = sess.run(None, {"input_values": x[None, :]})[0][0]
        m = logits.max(axis=-1, keepdims=True)
        lp = logits - (m + np.log(np.exp(logits - m).sum(axis=-1, keepdims=True)))
        try:
            got = words_from_window(lp, seg["text"], vocab, blank,
                                    offset_s=float(a) / ALIGN_SR,
                                    source_cue=seg.get("id"))
        except ValueError:
            # The cue is clipped by the excerpt boundary (or otherwise too
            # short for its text): its words cannot be timed here.
            continue
        words.extend(got)
    return words


def _diarize(wav: Path, segmentation_model: Path, embedding_model: Path,
             threads: int = 4) -> list[dict]:
    """Run the documented sherpa-onnx Pyannote segmentation + speaker embedding path."""
    import soundfile as sf
    import sherpa_onnx

    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(segmentation_model), window_shift_ratio=0.1),
            num_threads=threads, provider="cpu"),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(embedding_model), num_threads=threads, provider="cpu"),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=-1, threshold=0.5, compute_confidence=True),
        min_duration_on=0.25, min_duration_off=0.3)
    if not cfg.validate():
        raise RuntimeError("A sherpa-onnx diarizálási modell/config nem érvényes")
    diarizer = sherpa_onnx.OfflineSpeakerDiarization(cfg)
    signal, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    if sr != diarizer.sample_rate:
        raise ValueError(f"Diarizálási mintavétel: {sr}, elvárt: {diarizer.sample_rate} Hz")
    # preserve both channels as two independent analysis passes if caller extracts
    # mono; original source remains untouched and the selected channel is explicit.
    result = diarizer.process(signal[:, 0]).sort_by_start_time()
    return [{"start": round(float(s.start), 3), "end": round(float(s.end), 3),
             "speaker": f"SPEAKER_{int(s.speaker):02d}"} for s in result]


def run(project: Project, tag: str, start: float, duration: float,
        audio_stream: int | None = None, segmentation_model: Path | None = None,
        embedding_model: Path | None = None, align_model: Path | None = None,
        align: bool = True,
        diarize: bool = True, campp: bool = True,
        identity_link: bool = True, language: str = "he") -> Path:
    """Run short, read-only audit; returns the JSON result path.

    ``language`` a **fő hang** nyelve: ``"en"`` esetén az angol hangsávot és az
    angol (dubtitle) feliratot használjuk, a szóidőket pedig helyi ASR adja
    (Parakeet), nem héber CTC-illesztés. A héber ilyenkor csak felirat.
    """
    if start < 0 or duration <= 0 or duration > 300:
        raise ValueError("A próba 0–300 másodperc közti, pozitív hosszú legyen")
    if language not in ("he", "en"):
        raise ValueError(f"Ismeretlen nyelv: {language!r} (he vagy en)")
    tag = tag.upper()
    if campp and not (project.work / tag / "utterances.json").is_file():
        raise ValueError("Ehhez a próbához előbb futtasd az Előkészítést "
                         "(még nincs utterances.json).")
    out_dir = project.ep_dir(tag) / "source_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    if language == "en":
        srt = extract_subtitles(project, tag, out_dir / "source_english.srt",
                                lang="en", prefer_title="Dubtitle")
    else:
        srt = extract_hebrew_subtitles(project, tag, out_dir / "source_hebrew.srt")
    source_media, selected_stream = resolve_source_audio_stream(
        project, tag, audio_stream, None if audio_stream is not None else language)
    wav = extract_source_audio(project, tag,
                               out_dir / f"source_{language}_{start:.0f}_{duration:.0f}.wav",
                               start, duration, stream=selected_stream)
    segments = subtitle_segments(srt, start, start + duration, lang=language)
    if not segments:
        raise ValueError(f"A kiválasztott {start:g}–{start+duration:g}s tartományban "
                         f"nincs {language} feliratszöveg")
    align_path = out_dir / f"{language}_words_{start:.0f}_{duration:.0f}.json"
    aligner = None
    if align and language == "en":
        key = hashlib.sha256(json.dumps(
            {"audio_stream": selected_stream, "start": start, "duration": duration,
             "asr": ASR_EN_MODEL,
             "audio": [str(wav.resolve()), wav.stat().st_size, wav.stat().st_mtime_ns]},
            ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        key_path = align_path.with_suffix(".key")
        if align_path.exists() and key_path.exists() \
                and key_path.read_text(encoding="ascii") == key:
            words = json.loads(align_path.read_text(encoding="utf-8"))
        else:
            words = _asr_words_english(wav)
            align_path.write_text(json.dumps(words, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
            key_path.write_text(key, encoding="ascii")
    elif align:
        aligner = _find_align_model(project, tag, align_model)
        model_file = _align_model_file(aligner)
        # Cached timings are valid only for the same transcript, audio excerpt,
        # aligner model and actual stream. Include those inputs so reruns cannot
        # reuse a prior different-track / different-text timing file by accident.
        align_key = hashlib.sha256(json.dumps(
            {"segments": segments, "audio_stream": selected_stream, "start": start,
             "duration": duration, "aligner": "onnx-wav2vec2-he-ctc",
             "model": [str(model_file.resolve()), model_file.stat().st_size,
                       (aligner / "vocab.json").stat().st_size],
             "audio": [str(wav.resolve()), wav.stat().st_size, wav.stat().st_mtime_ns]},
            ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        if not align_path.exists() or not (align_path.with_suffix(".key").exists()
                and align_path.with_suffix(".key").read_text(encoding="ascii") == align_key):
            words = _align_hebrew(wav, segments, aligner)
            align_path.write_text(json.dumps(words, ensure_ascii=False, indent=2), encoding="utf-8")
            align_path.with_suffix(".key").write_text(align_key, encoding="ascii")
        else:
            words = json.loads(align_path.read_text(encoding="utf-8"))
    else:
        words = []
    turns_path = out_dir / f"speaker_turns_{start:.0f}_{duration:.0f}.json"
    if diarize:
        seg_model = segmentation_model or project.model("diar_segmentation")
        emb_model = embedding_model or (project.model("embedding"))
        if not seg_model.is_file():
            bundled = project.root / "models" / "diar" / "sherpa-onnx-pyannote-segmentation-3-0" / "model.int8.onnx"
            if bundled.is_file():
                seg_model = bundled
        if not seg_model.is_file():
            legacy = project.work / "models" / "pyannote-segmentation-3-0.onnx"
            if legacy.is_file():
                seg_model = legacy
        if not seg_model.is_file():
            raise FileNotFoundError(f"Nincs sherpa diarizáló szegmentálómodell: {seg_model}")
        if not emb_model.is_file():
            raise FileNotFoundError(f"Nincs CAM++ embeddingmodell: {emb_model}")
        diarize_key = hashlib.sha256(
            f"{wav.resolve()}|{wav.stat().st_size}|{seg_model.resolve()}|{seg_model.stat().st_size}|"
            f"{emb_model.resolve()}|{emb_model.stat().st_size}|sherpa-onnx|4".encode("utf-8")
        ).hexdigest()
        key_path = turns_path.with_suffix(".key")
        if not turns_path.exists() or not (key_path.exists()
                and key_path.read_text(encoding="ascii") == diarize_key):
            turns = _diarize(wav, seg_model, emb_model)
            turns_path.write_text(json.dumps(turns, ensure_ascii=False, indent=2), encoding="utf-8")
            key_path.write_text(diarize_key, encoding="ascii")
        else:
            turns = json.loads(turns_path.read_text(encoding="utf-8"))
    else:
        turns = []
    word_speakers = turn_segment_ids(turns, words) if turns and words else []
    emb_for_id = emb_model if diarize else (embedding_model or project.model("embedding"))
    # A fordulók beágyazását egyszer végezzük el, és mind a klaszter-összevetés,
    # mind a stabil C### azonosság ugyanazt a lenyomatot használja.
    protos: dict = {}
    if turns and (campp or identity_link):
        from . import identity as identity_mod
        protos = identity_mod.prototypes_from_turns(turns, wav, emb_for_id)
    clusters = []
    if campp and turns:
        clusters = compare_cam_clusters(turns, words, wav,
            project.ep_dir(tag) / "utterances.json", emb_for_id,
            prototypes=protos)
    identity_result: dict = {}
    if identity_link and turns:
        from . import identity as identity_mod
        from . import review as review_mod
        identity_result = identity_mod.link_speakers(
            project, tag, turns, wav, emb_for_id, start, duration,
            prototypes=protos)
        identity_result = review_mod.apply(
            identity_result, review_mod.load(project, tag))

    if language == "en":
        transcript = "container English dubtitle subtitle (Hebrew stays subtitle-only)"
        alignment = (f"local ONNX ASR word timestamps ({ASR_EN_MODEL}, int8)"
                     if align else "disabled")
        limits = ["No automatic cast/voice assignment is made.",
                  "Word timings come from English ASR, not from forced alignment; "
                  "ASR never hears the subtitle text, so text and timing are independent.",
                  "Hebrew dialogue is subtitle-only; the main dub follows the English track.",
                  "Film music, overlap, short turns and off-screen speech need human review.",
                  "Identity links are CAM++ cosine evidence; a human confirmation "
                  "is what makes a C### record stable across episodes."]
    else:
        transcript = "container Hebrew subtitle (not translated Hungarian)"
        alignment = (f"local ONNX wav2vec2 Hebrew char-CTC ({ALIGNER_REPO})"
                     if align else "disabled")
        limits = ["No automatic cast/voice assignment is made.",
                  "Subtitle alignment does not validate the transcript.",
                  "Word timings are character-CTC estimates; spot-check them against playback.",
                  "Film music, overlap, short turns and off-screen speech need human review.",
                  "Identity links are CAM++ cosine evidence; a human confirmation "
                  "is what makes a C### record stable across episodes."]
    result = {"episode": tag, "audio_source": str(source_media),
              "audio_stream": selected_stream, "language": language, "start": start,
              "duration": duration,
              "method": {"transcript": transcript, "alignment": alignment,
                         "diarization": "sherpa-onnx Pyannote segmentation + CAM++" if diarize else "disabled",
                         "identity_link": "existing CAM++ prototype cosine; review required" if campp else "disabled"},
              "subtitle_srt": str(srt),
              "subtitle_segments": segments, "words": word_speakers,
              "speaker_turns": turns, "candidate_cast_links": clusters,
              "speaker_identity": identity_result,
              "limitations": limits}
    # A nyelv a fájlnévben: a héber és az angol próba ne írja felül egymást.
    stem = (f"audit_{start:.0f}_{duration:.0f}" if language == "he"
            else f"audit_{language}_{start:.0f}_{duration:.0f}")
    result_path = out_dir / f"{stem}.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    ap.add_argument("episode")
    ap.add_argument("--start", type=float, default=30.0)
    ap.add_argument("--duration", type=float, default=45.0)
    ap.add_argument("--audio-stream", type=int, default=None,
                    help="ffprobe globális stream-index; alapértelmezésben az első hangsáv")
    ap.add_argument("--segmentation-model", type=Path)
    ap.add_argument("--embedding-model", type=Path)
    ap.add_argument("--align-model", type=Path, default=None,
                    help="héber ONNX illesztőmodell csomag mappája (alap: work/models/…)")
    ap.add_argument("--no-align", action="store_true")
    ap.add_argument("--no-diarize", action="store_true")
    ap.add_argument("--no-campp", action="store_true")
    ap.add_argument("--no-identity", action="store_true")
    ap.add_argument("--language", default="en", choices=("en", "he"),
                    help="a fő hang nyelve (en: angol hangsáv + ASR szóidő, héber csak felirat)")
    args = ap.parse_args(argv)
    from .config import load_project
    try:
        result = run(load_project(args.project), args.episode, args.start, args.duration,
                     audio_stream=args.audio_stream,
                     segmentation_model=args.segmentation_model,
                     embedding_model=args.embedding_model, align_model=args.align_model,
                     align=not args.no_align,
                     diarize=not args.no_diarize, campp=not args.no_campp,
                     identity_link=not args.no_identity, language=args.language)
    except Exception as exc:
        print(f"[source-audit] HIBA: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"[source-audit] riport: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())