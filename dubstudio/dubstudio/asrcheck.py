"""A kész MKV vissza-felismerése: a magyar szinkronsáv ASR-feliratozása és
összevetése a szándékolt szöveggel.

Ez a **hanggenerálás UTÁNI** ellenőrzés (a ``textcheck`` a szöveget ellenőrzi a
TTS előtt; ez a kész hangot) — megmutatja, amit a néző ténylegesen hall:

  * epizód-szintű CER/WER a teljes magyar szinkronsávon,
  * soronkénti CER az időbélyeg-alapú illesztéssel,
  * gyanús sorok listája (időpont + szándékolt szöveg + hallott szöveg +
    hiányzó szavak) — a kihagyott/sikertelen szintézis-sorok itt „nincs
    hallott szó”ként jelennek meg.

Három üzemmód (a projekt ``asr.mode`` beállítása):

  * ``whisper`` — a korábbi út: a magyar Whisper-finetune a teljes sávot
    ablakosan dolgozza fel (~1,5× valós idő);
  * ``parakeet`` — csak a gyors Parakeet (ONNX, CPU; ~0,1× valós idő);
  * ``hybrid`` — **Parakeet az elsődleges** (gyors teljes átfésülés), majd a
    gyanús sorokat a Whisper-HU hallgatja újra, és a **jobb olvasat** számít
    (mint az Auris ``qa_api._evaluate`` hibrid útján). Így a drága modell
    csak a vitás helyekre fut, a mérés pedig megtartja a Whisper-HU
    pontosságát.

A küszöb (``BAD_CER``) alapján kiválasztott sorok újrahallgatása a
szakirodalom szerinti „rizikó alapú újradekódolás”: nincs univerzális
küszöb, a saját anyagon kell kalibrálni — ezért a kapu külön, modellenként
állítható (``gate.asr_cer_max`` szótárként: ``{"parakeet": …, "whisper": …}``),
mert a két modell CER-skálája nem összemérhető.

Az Auris ``core.qa`` eszközeit használja (magyar Whisper-finetune +
Parakeet + ``score_transcript``), hogy a számok egyezzenek az auris-oldali
mérésekkel.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import sys
import time
from pathlib import Path

from . import media, subs
from .config import Project
from .plan import Entry

#: ennél magasabb sor-szintű CER gyanús (zenei aláfestés mellett is tartható)
BAD_CER = 0.30
#: második, „figyelendő” sáv: még nem hiba, de a riport láttatja
WATCH_CER = 0.20

#: A Whisper-pipeline CPU-memóriája a köteg méretével **lineárisan** nő: 8
#: ablaknál mért 13–20 GB munkakészlet és lapozás (31,6 GB-os gép, 48 szál).
#: Ezért a köteget felülről korlátoljuk, a szálakat megkötjük, és minden
#: köteg után felszabadítjuk a szeleteket.
MAX_BATCH = 2
#: e fölött a futás **egyszerre egy ablakra** vált vissza (a kötegelt
#: előny kisebb, mint a lapozás kára)
RSS_SOFT_MB = 11264.0
#: e fölött figyelmeztetünk a naplóban (nem állítjuk le a futást). A modell
#: betöltve önmagában ~9 GB, ezért a 6 GB-os küszöb folyamatos zajt adott.
RSS_WARN_MB = 11264.0


def _rss_mb() -> float | None:
    """A folyamat munkakészlete MB-ban (Windows; máshol None)."""
    try:
        import ctypes
        import ctypes.wintypes as wt

        class _PMC(ctypes.Structure):
            _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        pmc = _PMC()
        pmc.cb = ctypes.sizeof(_PMC)
        h = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc),
                                                   pmc.cb):
            return pmc.WorkingSetSize / (1024 * 1024)
    except Exception:
        pass
    return None


def _trim_working_set() -> None:
    """A munkakészlet visszaadása a rendszernek minden köteg után.

    A torch/OpenMP arena a Windows-on nem adja vissza a szabad oldalakat,
    ezért a munkakészlet monoton nő (mért: 8→20 GB egy kör alatt), és a gép
    lapozni kezd. Az ``EmptyWorkingSet`` + ``gc.collect`` visszaveszi a
    már nem használt lapokat (a szemétgyűjtés után).
    """
    try:
        import gc

        gc.collect()
    except Exception:
        pass
    try:
        import ctypes

        h = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.psapi.EmptyWorkingSet(h)
    except Exception:
        pass


def _limit_torch_threads() -> None:
    """A CPU-szálak megkötése (alapértelmezés 4).

    A 48 szálas alap beállítás nem gyorsabb (a Whisper-dekódolás soros),
    viszont szálanként külön memória-arena nő — ez volt a 63 GB-os commit.
    """
    try:
        import torch

        n = int(os.environ.get("DUBSTUDIO_TORCH_THREADS", "4"))
        torch.set_num_threads(max(1, n))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass          # már inicializált — nem hiba
    except Exception:
        pass
MIN_CHARS = 6         # ennél rövidebb szövegnél nem értékelünk

#: a gyanús sor három fajtája — a valódi audibilitási találatokat választjuk el
#: attól, amit a felismerés rontott el (üres ablak / ismétlés-hurok)
KIND_LABEL = {"real": "valódi", "loop": "hurok", "empty": "üres"}
_KIND_ORDER = {"real": 0, "loop": 1, "empty": 2}

#: hibrid megerősítés: a gyanús sorok körüli ablak (környezet a Whispernek),
#: és az összevonás: ennél közelebbi ablakok egy hívásba kerülnek. A hézag
#: szándékosan kicsi: mérés szerint a hosszabb szelet (4 s-os összevonás)
#: kevesebb hamis találatot old fel, mert a szó-időbélyegek a szelet szélén
#: elcsúsznak, és a szomszédos beszéd bekerül a sor ablakába. Inkább több,
#: rövid hívás — a találat-értékelés pontossága fontosabb a pár percnél.
CONFIRM_PAD = 0.30
MERGE_GAP_S = 0.75

#: a becsomagolt szeletek közti csend: a Whisper ne olvassza össze a két
#: távoli szeletet (a szó-időbélyegek így se csússzanak át a határon).
#: Mérés (E05): a 0,25 s kevés — a 3 hívásos csomag 5-ről 1-re rontotta a
#: feloldott sorokat; 0,4 s mellett az arbitráció megegyezik az alappal.
PACK_SILENCE_S = 0.40

#: egy csomagolt hívás maximális hossza — mérés (E05, 27 szelet):
#:   alap (szeletenként, 27 hívás)  938,2 s · 5 feloldott · 170/43 valódi/üres
#:   29 s / 0,25 s (3 hívás)        156,2 s · 1 feloldott · 164/53
#:   14 s / 0,4 s  (5 hívás)        202,8 s · 5 feloldott · 168/45  ← ez a jó
#: A rövid keret kevesebb szó-időbélyeg-driftet ad (a hosszú, összefüggő
#: hívásban a modell a szeletek között is „beszél”), a több hívás így is
#: 4,6×-el gyorsabb, mint a szeletenkénti út.
PACK_LIMIT_S = 14.0

#: a megerősítő hívások becsomagolása ki-be (``DUBSTUDIO_ASR_CONFIRM_PACK=0``
#: kikapcsolja — a régi, szeletenkénti út A/B tesztjéhez)
DEFAULT_CONFIRM_PACK = 1


def confirm_pack() -> bool:
    """A megerősítő szeletek becsomagolása (kevesebb Whisper-hívás)."""
    try:
        v = os.environ.get("DUBSTUDIO_ASR_CONFIRM_PACK")
        if v is None:
            return bool(DEFAULT_CONFIRM_PACK)
        return v.strip().lower() not in ("0", "false", "no", "off")
    except Exception:
        return bool(DEFAULT_CONFIRM_PACK)


def _env_float(name: str, default: float) -> float:
    try:
        v = float(os.environ.get(name) or 0)
    except Exception:
        return default
    return v if v > 0 else default


def pack_silence() -> float:
    """A becsomagolt szeletek közi csend (``DUBSTUDIO_ASR_PACK_SILENCE``)."""
    return _env_float("DUBSTUDIO_ASR_PACK_SILENCE", PACK_SILENCE_S)


def pack_limit() -> float:
    """Egy csomagolt hívás maximális hossza (``DUBSTUDIO_ASR_PACK_LIMIT``).

    Rövidebb keret = kevesebb szó-időbélyeg-drift a hosszú, összefüggő
    hívásban (a megerősítés soronként illeszt), de több hívás.
    """
    return _env_float("DUBSTUDIO_ASR_PACK_LIMIT", PACK_LIMIT_S)

_tok_re = re.compile(r"[\w']+")


def _is_loop(text: str) -> bool:
    """Whisper-ismétlés-hurok felismerése a hallott szövegben.

    Zene alatt a modell ugyanazt a szót/mondatot kezdi ismételni
    („mindenki mindenki mindenki…”, „… mindenki különbözött mindenki
    különbözött”). Ez felismerési hiba, nem audibilitási találat — a 0,30-as
    küszöb önmagában nem különíti el, ezért külön kategória.
    """
    w = [t.lower() for t in _tok_re.findall(text or "")
         if any(c.isalnum() for c in t)]
    if len(w) < 3:
        return False
    for s in range(len(w) - 2):
        for n in range(1, (len(w) - s) // 2 + 1):
            pat = w[s:s + n]
            i, covered = s + n, 0
            while w[i:i + n] == pat:
                covered += n
                i += n
            if covered >= n and (n + covered >= 4 or covered >= 2 * n):
                return True
    return False


def _kind(heard: str) -> str:
    """A sor fajtája a hallott szöveg alapján: üres / hurok / valódi."""
    if not heard.strip():
        return "empty"
    return "loop" if _is_loop(heard) else "real"


def _sort_rows(rows: list[dict]) -> list[dict]:
    """Rendezés: először a valódi találatok (CER szerint csökkenő), utána a
    hurkos, végül az üres sorok — a riportban így a lényeg áll elöl."""
    rows.sort(key=lambda r: (_KIND_ORDER.get(r.get("kind", "real"), 0),
                             -float(r.get("cer", 0.0))))
    return rows


def _auris():
    """Auris core.qa betöltése (ugyanaz az útkeresés, mint a subs.prepare_hu)."""
    reader = os.environ.get("DUBSTUDIO_AURIS_READER", "")
    cands = [Path(reader)] if reader else []
    cands += [Path(__file__).resolve().parents[2] / "reader"]
    for c in cands:
        if (c / "core" / "qa.py").is_file():
            if str(c) not in sys.path:
                sys.path.insert(0, str(c))
            from core.qa import Transcriber, score_transcript  # type: ignore
            return Transcriber, score_transcript
    raise RuntimeError("az Auris core.qa nem elérhető "
                       "(DUBSTUDIO_AURIS_READER környezeti változó?)")


def _auris_classes():
    """A két ASR-osztály (a ``_auris`` már betette az Auris-t a sys.path-ra)."""
    _auris()
    from core.qa import ParakeetTranscriber, Transcriber  # type: ignore

    return ParakeetTranscriber, Transcriber


def words_in_window(words: list[dict], a: float, b: float,
                    pad: float = 0.15) -> list[str]:
    """Az ``a``–``b`` ablakba eső szavak (szó-időbélyeg szerint)."""
    out = []
    for w in words:
        s = w.get("start")
        if s is None:
            continue
        if a - pad <= float(s) <= b + pad:
            t = str(w.get("word") or "").strip()
            if t:
                out.append(t)
    return out


#: az ablak mérete ≤ 30 s, hogy a transformers long-form ága sose induljon el
WIN_S = 29.0
#: lépésköz: az (WIN_S - STEP_S) / 2 = 2,0 s átfedés mindkét oldalon
#: biztosítja, hogy a határon átívelő szó (≤4 s) teljesen benne legyen
#: abban az ablakban, amelyik megtartja
STEP_S = 25.0


#: a kötegelt Whisper-feldolgozás alapértelmezése — 1 = a régi, soronkénti
#: út. A projekt ``asr.batch`` (vagy a ``DUBSTUDIO_ASR_BATCH`` környezeti
#: változó) több ablakot küld egyetlen pipeline-hívásba: azonos modell,
#: azonos ablakok, kevesebb hívás-ráfordítás (mérés: work/bench_batch.py).
DEFAULT_BATCH = 1


def batch_size(project=None) -> int:
    """Az ablakok kötegének mérete: ``asr.batch`` a projektből, különben
    ``DUBSTUDIO_ASR_BATCH``, végül a biztonságos alapértelmezés (1 = soronként)."""
    try:
        v = int((((getattr(project, "asr", None) or {}).get("batch")) or 0))
    except Exception:
        v = 0
    if not v:
        try:
            v = int(os.environ.get("DUBSTUDIO_ASR_BATCH") or 0)
        except Exception:
            v = 0
    v = v if v > 0 else DEFAULT_BATCH
    # A köteg felső korlátja nem kényelmi kérdés: 8 ablaknál mértünk 13–20 GB
    # munkakészletet és lapozást; 2 ablaknál a csúcs töredéke ennek.
    return max(1, min(int(v), MAX_BATCH))


def _whisper_local(res: dict) -> list[dict]:
    """A pipeline egy hívásának szavai — az ablak saját koordinátájában."""
    out: list[dict] = []
    for ch in res.get("chunks") or []:
        st_en = ch.get("timestamp") or (None, None)
        st = st_en[0]
        if st is None:
            continue
        out.append({"word": str(ch.get("text") or "").strip(),
                    "start": float(st),
                    "end": float(st_en[1]) if st_en[1] is not None else None})
    return out


def _pipe_windows(tr, segs: list, batch: int, verbose: bool = True,
                  label: str = "ASR") -> list[list[dict]]:
    """A Whisper-pipeline hívása az ablakokon — kötegelve, visszaeséssel.

    Egy hívásba ``batch`` ablak kerül (a pipeline listás bemeneten tölti fel
    a mintákat), így a motor kevesebb hívásköltséggel dolgozik — a mérés a
    lényeg, nem a sebesség feltételezése (``work/bench_batch.py``). Ha a
    kötegelt hívás nem működik (API-elütés, memória), ugyanazok az ablakok
    soronként futnak tovább: a kimenet ilyenkor is a régi úté.
    """
    gk = {"task": "transcribe", "language": "hu"}
    step = max(1, int(batch or 1))
    outs: list[list[dict]] = []
    total = len(segs)
    try:
        import torch

        guard = torch.inference_mode()
    except Exception:                      # torch nélkül (ONNX-út) nem kell
        guard = contextlib.nullcontext()
    for i in range(0, total, step):
        chunk = segs[i:i + step]
        try:
            with guard:                    # ne épüljön autograd-gráf
                res = tr._pipe([{"raw": s, "sampling_rate": 16000}
                                for s in chunk],
                               batch_size=len(chunk), generate_kwargs=gk,
                               return_timestamps="word")
            if isinstance(res, dict):
                res = [res]
        except Exception as exc:            # a köteg nem ment → soronként
            if step > 1 and verbose:
                print(f"  [{label}] köteg ({len(chunk)} ablak) nem ment: "
                      f"{type(exc).__name__} — soronként folytatom",
                      flush=True)
            with guard:
                res = [tr._pipe({"raw": s, "sampling_rate": 16000},
                                generate_kwargs=gk,
                                return_timestamps="word")
                       for s in chunk]
        outs.extend(_whisper_local(r) for r in res)
        k = i + len(chunk)
        if verbose and (k >= total or k // 5 > i // 5):
            print(f"  [{label}] {min(k, total)}/{total} ablak felismerve",
                  flush=True)
    return outs


def _load_checkpoint(path: Path | None) -> dict[int, list[dict]]:
    """A korábbi, megszakított futás ablakonkénti eredménye.

    A Whisper-futás CPU-n ~90 perc; ha a folyamat (vagy a gép) megszakad,
    a kész ablakok újrahallgatása tiszta veszteség. A checkpoint soronként
    egy ablak szavait tartja (``{"i": 12, "words": [...]}``).
    """
    done: dict[int, list[dict]] = {}
    if path is None or not path.exists():
        return done
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                done[int(rec["i"])] = list(rec.get("words") or [])
            except (ValueError, KeyError, TypeError):
                continue
    except OSError:
        return {}
    return done


def _transcribe_windowed(tr, wav_path: Path, verbose: bool = True,
                          batch: int = 0,
                          checkpoint: Path | None = None
                          ) -> tuple[list[dict], str]:
    """A teljes sáv felismerése ≤ WIN s-os ablakokban, globális időbélyegekkel.

    Miért nem a ``Transcriber.transcribe()``: az a teljes sávot egyben adja a
    pipeline-nak; a transformers 5.x long-form generálása ilyenkor egyben
    küldené a ~35 perces sáv log-meljét az encodernek (~100 000 token
    figyelem-mátrix) → natív memóriahiba, a folyamat segfaulttal pusztul el
    (ez ölte meg a korábbi teljes sávos futást). A ≤ 29 s-os hívás a
    whiskertől megszokott 30 s-os úton megy, a pipeline pedig a rövid
    fájloknál bevált szó-időbélyeg-kezelését használja; a globális időt
    mi toljuk rá az ablak kezdetével. Ugyanez az út viszi a Parakeet-et
    (ONNX) is: a teljes sávra az a ~23 GB-ot próbált allokálni (OOM),
    ablakonként azonban kényelmesen elfér.
    """
    import soundfile as sf

    x, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    dur = len(x) / sr
    ov = (WIN_S - STEP_S) / 2.0
    onnx_path = hasattr(tr, "_model") and not hasattr(tr, "_pipe")
    if onnx_path:                    # Parakeet (ONNX) — lusta betöltés itt
        if tr._model is None:
            import onnx_asr

            tr._model = onnx_asr.load_model(tr.model_id).with_timestamps()
    else:
        tr._load()

    # az ablakok előre, hogy a kötegelt út is ugyanezt a rácsot használja
    wins: list[tuple[float, float, float]] = []
    k = 0
    pos = 0.0
    while pos < dur - 1e-6:
        wins.append((pos, max(0.0, pos - ov), min(dur, pos + STEP_S + ov)))
        k += 1
        pos = k * STEP_S

    locals_: list[list[dict]] = []
    if onnx_path:                    # Parakeet: soronként (a maga ONNX-hívása)
        from core.qa import _parakeet_words            # type: ignore

        for k, (pos, a, b) in enumerate(wins, 1):
            seg = x[int(a * sr):int(b * sr)]
            res = tr._model.recognize(seg)
            locals_.append([{"word": w["word"], "start": w["start"],
                             "end": w["end"]}
                            for w in _parakeet_words(res.tokens, res.timestamps,
                                                     len(seg) / sr)])
            if verbose and k % 5 == 0:
                print(f"  [ASR] {min(pos, dur):.0f}/{dur:.0f} s felismerve",
                      flush=True)
    else:                            # Whisper: kötegelt pipeline-hívások
        _limit_torch_threads()
        step = max(1, min(int(batch or batch_size()), MAX_BATCH))
        total = len(wins)
        done = _load_checkpoint(checkpoint)
        locals_ = [done.get(i) for i in range(total)]
        if verbose and done:
            print(f"  [ASR] ellenőrzőpont: {len(done)}/{total} ablak már kész",
                  flush=True)
        for i in range(0, total, step):
            if all(locals_[k] is not None for k in range(i, min(i + step, total))):
                continue
            part = wins[i:i + step]
            # Csak az aktuális köteg hangja van a memóriában (a korábbi út
            # mind a 86 ablakot előre kimásolta), és a köteg után elengedjük.
            segs = [x[int(a * sr):int(b * sr)] for (_, a, b) in part]
            outs = _pipe_windows(tr, segs, len(segs), verbose=False)
            for off, words_i in enumerate(outs):
                locals_[i + off] = words_i
            del segs, outs
            if checkpoint is not None:
                try:
                    with checkpoint.open("a", encoding="utf-8") as fh:
                        for off in range(len(part)):
                            fh.write(json.dumps({"i": i + off,
                                                 "words": locals_[i + off]},
                                                ensure_ascii=False) + "\n")
                except OSError:
                    pass
            # a csúcsot a trim **előtt** mérjük (utána a rendszer visszaveszi
            # a lapokat, és 0-hoz közeli értéket mutatna)
            rss = _rss_mb()
            _trim_working_set()
            if rss and step > 1 and rss > RSS_SOFT_MB:
                # Dinamikus visszalépés: a mért csúcs alapján, nem tippből.
                step = 1
                if verbose:
                    print(f"  [ASR] memóriacsúcs {rss / 1024:.1f} GB → "
                          f"egyszerre egy ablak", flush=True)
            if verbose:
                k = min(i + step, total)
                if k >= total or k // 8 > i // 8:
                    extra = (f" · memória {rss / 1024:.1f} GB"
                             if rss else "")
                    warn = "  ⚠ memóriacsúcs" if (rss or 0) > RSS_WARN_MB else ""
                    print(f"  [ASR] {k}/{total} ablak felismerve{extra}{warn}",
                          flush=True)

    words: list[dict] = []
    for (pos, a, _b), local in zip(wins, locals_):
        if not local:
            continue
        core_b = min(pos + STEP_S, dur)
        for w in local:
            if w.get("start") is None:
                continue
            g = float(w["start"]) + a                  # globális kezdés
            if pos - 1e-6 <= g < core_b:               # a magjába eső szót tartjuk
                words.append({"word": str(w.get("word") or "").strip(),
                              "start": g,
                              "end": (float(w["end"]) + a)
                              if w.get("end") is not None else None})
    words.sort(key=lambda w: float(w["start"]))
    text = " ".join(w["word"] for w in words if w["word"])
    return words, text


def transcribe_words(wav_path: str | Path, language: str = "hu",
                     verbose: bool = True) -> tuple[list[dict], str]:
    """Egy hangsáv szavai időbélyeggel — projekt nélkül (a narrációs modulnak).

    Ugyanaz az ablakos út, mint az ASR-visszaellenőrzésnél: a hosszú sáv nem
    megy egyben a transformer long-form ágára (az natív memóriahibát okoz).
    """
    _auris()
    from core.qa import (ParakeetTranscriber, Transcriber,  # type: ignore
                         asr_backend_for)

    # A ``Transcriber.for_language()`` hibrid módban egy összevont
    # ``model_id``-t ad ("parakeet + whisper"), amit az ONNX-betöltő nem tud
    # értelmezni. A hibrid elsődleges motorja a Parakeet, ezért azt használjuk
    # önállóan; ha nem elérhető, a Whisper megy.
    backend = asr_backend_for(language)
    tr = (Transcriber.whisper_for(language) if backend == "whisper"
          else ParakeetTranscriber())
    return _transcribe_windowed(tr, Path(wav_path), verbose=verbose)


def heard_for(e: Entry, words: list[dict], pad: float = 0.30,
              wide: float = 1.5) -> str:
    """A sor körüli hallott szöveg; üres ablaknál tágabb rákeresés.

    A Whisper szó-időbélyege néhány másodpercet csúszhat (különösen hosszú
    csend után), ezért előbb a szűk, majd a tágabb ablakot nézzük — csak
    akkor mondjuk „nincs hallott szó”-nak, ha egyik sem ad szót.
    """
    heard = " ".join(words_in_window(words, e.cue_start, e.cue_end, pad))
    if not heard.strip() and wide > pad:
        heard = " ".join(words_in_window(words, e.cue_start, e.cue_end, wide))
    return heard


def _score_entry(e: Entry, words: list[dict], score_transcript,
                 heard: str | None = None
                 ) -> tuple[bool, dict | None]:
    """Egy sor kiértékelése felismert szavakból: (ellenőrzött?, gyanús sor)."""
    expected = subs.one_line(e.hu or "")
    if not expected.strip() or len(expected) < MIN_CHARS:
        return False, None
    if heard is None:
        heard = heard_for(e, words)
    if not heard.strip():
        return True, {"i": e.i, "start": e.cue_start, "cer": 1.0, "kind": "empty",
                      "expected": expected[:100], "heard": "",
                      "missing": ["<nincs hallott szó>"]}
    s = score_transcript(expected, heard, "hu")
    if s["cer"] < WATCH_CER:
        return True, None
    return True, {"i": e.i, "start": e.cue_start, "cer": s["cer"],
                  "kind": _kind(heard), "expected": expected[:100],
                  "heard": heard[:100], "missing": s["missing_words"][:8],
                  "level": "fail" if s["cer"] >= BAD_CER else "watch"}


def score_lines(entries: list[Entry], words: list[dict],
                 *, model_id: str, mkv, seconds: float = 0.0,
                 row_cap: int = 500) -> dict:
    """Az ablakos felismerés soronkénti kiértékelése (már felismert szavakból).

    A CER a **felismert** sorokból számol (a teljes sáv szövegét nem használjuk:
    a kieső szeletek és a szó-időbélyeg-csúszás így nem terhelik), a
    lefedettséget viszont külön visszaadja (``coverage``), hogy a kapu lássa.
    """
    _, score_transcript = _auris()
    todo = [e for e in entries if not e.skip]
    rows: list[dict] = []
    expected_all: list[str] = []
    heard_all: list[str] = []
    checked = bad = 0
    lines_total = 0
    counts = {"real": 0, "loop": 0, "empty": 0}
    watch = 0
    watch_rows: list[dict] = []
    for e in todo:
        expected = subs.one_line(e.hu or "")
        if not expected.strip():
            continue
        lines_total += 1
        heard_line = heard_for(e, words)
        # Csak a tényleg felismert sorokat hasonlítjuk össze: az „üres” sor
        # (a modell nem adott szót az ablakra) mérési hiány, nem a szinkron
        # hibája — a teljes sáv CER-jét nem terhelheti, külön számoljuk.
        if heard_line.strip():
            expected_all.append(expected)
            heard_all.append(heard_line)
        ok, row = _score_entry(e, words, score_transcript, heard=heard_line)
        if not ok:
            continue
        checked += 1
        if row is None:
            continue
        # Kétfokozatú gyanú: a `watch` sáv még nem hiba (a kaput nem billenti),
        # de látszik a riportban — így a határeset-sorok nem tűnnek el, és
        # nem is hígítják a valódi találatokat.
        if row.get("level") == "watch":
            watch += 1
            if len(watch_rows) < row_cap:
                watch_rows.append(row)
            continue
        bad += 1
        counts[row["kind"]] += 1
        rows.append(row)

    _sort_rows(rows)
    full = score_transcript(" ".join(expected_all), " ".join(heard_all), "hu")
    return {
        "model": str(model_id),
        "mkv": str(mkv),
        "lines": len(todo),
        "checked": checked,
        "bad": bad,
        "watch": watch,
        "watch_cer": WATCH_CER,
        "watch_rows": watch_rows[:row_cap],
        # lefedettség: a felismert (nem üres) sorok aránya. Az „üres” sor
        # mérési hiány, ezért a CER-t nem terheli — de a kapunak látnia kell,
        # különben a mérés elhallgatásával lehetne zöld.
        "lines_total": lines_total,
        "recognized": len(expected_all),
        "coverage": round(len(expected_all) / max(lines_total, 1), 4),
        "real": counts["real"],
        "loops": counts["loop"],
        "empties": counts["empty"],
        "bad_cer": BAD_CER,
        "cer": full["cer"],
        "wer": full["wer"],
        "missing_words": full["missing_words"][:20],
        "rows": rows[:row_cap],
        "seconds": round(seconds, 1),
    }


def _merge_spans(spans: list[tuple[float, float]], gap: float = MERGE_GAP_S,
                 limit: float = WIN_S) -> list[tuple[float, float]]:
    """A gyanús sorok ablakainak összevonása ≤ ``limit`` hosszú szeletekké.

    Egy Whisper-hívás drága: a szomszédos gyanús sorokat egy szeletben
    hallgatjuk meg (a rövid, kontextus nélküli kivágások amúgy is rosszabb
    olvasatot adnak). A szelet soha nem hosszabb ``limit``-nél, mert a
    transformers long-form ága 30 s felett memóriahibába fut.
    """
    out: list[list[float]] = []
    for a, b in sorted(spans):
        if b - a > limit:
            b = a + limit
        if out and a - out[-1][1] <= gap and b - out[-1][0] <= limit:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _pack_spans(spans: list[tuple[float, float]], limit: float = WIN_S,
                silence: float = PACK_SILENCE_S
                ) -> list[list[tuple[float, float, float]]]:
    """A megerősítő szeletek becsomagolása ≤ ``limit`` hosszú hívásokba.

    A Whisper-hívás drága, mert minden hívás feltölti a 30 s-os
    encoderkeretet — a költséget a **hívások száma** határozza meg, nem a
    tényleges szelet-hossz. A gyanús sorok viszont szétszórtak az epizódban:
    30 sor külön hívása 938 s CPU (mérés), miközben a bennük lévő beszéd
    összesen ~2 perc. A független szeleteket egymás mellé rakjuk (közéjük
    rövid csendet téve), így ugyanaz az anyag kevesebb hívásra fér.

    Csoportonként: ``[(globális_kezdés, globális_zárás, kezdés_a_hívásban),
    …]`` — a visszaillesztés az eltolással történik (``_map_words``).
    """
    groups: list[list[tuple[float, float, float]]] = []
    cur: list[tuple[float, float, float]] = []
    cur_len = 0.0
    for a, b in sorted(spans):
        limit = pack_limit()
        silence = pack_silence()
        while b - a > limit:            # hosszú szelet felvágása a keretre
            if cur:
                groups.append(cur)
                cur, cur_len = [], 0.0
            groups.append([(a, a + limit, 0.0)])
            a += limit
        n = len(cur)
        add = silence if n else 0.0
        if n and cur_len + add + (b - a) > limit:
            groups.append(cur)
            cur, cur_len, add = [], 0.0, 0.0
        cur.append((a, b, cur_len + add))
        cur_len += add + (b - a)
    if cur:
        groups.append(cur)
    return groups


def _pack_audio(group: list[tuple[float, float, float]], x, sr: int):
    """Egy csoport hangja a híváshoz: szeletek + közéjük rövid csend."""
    import numpy as np

    parts = []
    sil = np.zeros(int(pack_silence() * sr), dtype="float32")
    for k, (a, b, _off) in enumerate(group):
        if k:
            parts.append(sil)
        parts.append(x[int(a * sr):int(b * sr)])
    return np.concatenate(parts) if parts else np.zeros(0, dtype="float32")


def _map_words(group: list[tuple[float, float, float]],
               local: list[dict]) -> list[dict]:
    """A hívás helyi szavainak visszaillesztése a globális időtengelyre.

    Minden szó a ``group`` azon tagjához kerül, amelyiknek a (híváson belüli)
    ablakja a legközelebb esik — a szeletek közi csendbe eső szót a
    szomszédhoz rendeljük (a maximális eltérés fél csend-hossz, jóval a
    ``words_in_window`` 0,15 s-os padja alatt).
    """
    out: list[dict] = []
    for w in local:
        s = w.get("start")
        if s is None:
            continue
        s = float(s)
        best, best_d = None, None
        for (a, b, off) in group:
            lo, hi = off, off + (b - a)
            d = 0.0 if lo <= s <= hi else (lo - s if s < lo else s - hi)
            if best_d is None or d < best_d:
                best, best_d = (a, off), d
        a, off = best
        out.append({"word": str(w.get("word") or "").strip(),
                    "start": a + (s - off), "end": None})
    return out


def _confirm_rows(rows: list[dict], entries: list[Entry], wav_path: Path,
                   whisper, max_rows: int = 0, verbose: bool = True,
                   batch: int = 0
                   ) -> tuple[list[dict], int, list[tuple[float, float]], int]:
    """A gyanús sorok újrahallgatása a Whisper-HU-val (hibrid megerősítés).

    Csak a gyanús sorok környékét ismerjük fel újra — ez a hibrid lényege: a
    Parakeet gyors teljes átfésülése után a lassú, pontosabb modell csak a
    vitás helyeket hallgatja meg. A sor **jobb olvasata** számít: ha a Whisper
    szerint a sor rendben van, a találat feloldódik (a Parakeet tévedett); ha
    továbbra is gyanús, a kisebb CER-ű olvasat kerül a riportba.

    ``max_rows`` > 0 esetén csak a **legrosszabb** ennyi sor kerül
    megerősítésre: a Whisper minden hívást 30 s-ra tölt fel, így a szétszórt
    találatok egyenként ~45 s CPU-t kérnek — a korlát nélkül a hibrid lassabb
    lenne a teljes Whisper-futásnál (mérés: 60 s-os szeleten 7 találat =
    6 hívás, 188 s). A kapu a Parakeet teljes epizód-CER-jére vonatkozik, a
    megerősítés a legfontosabb találatok auditzsűrűje.
    """
    import soundfile as sf

    by_i = {e.i: e for e in entries}
    pick: set[int] = set(range(len(rows)))
    if max_rows > 0 and len(rows) > max_rows:
        order = sorted(range(len(rows)),
                       key=lambda i: -float(rows[i].get("cer") or 0.0))
        pick = set(order[:max_rows])
    spans: list[tuple[float, float]] = []
    for i, r in enumerate(rows):
        if i not in pick:
            continue
        e = by_i.get(r.get("i"))
        if e is None or e.skip:
            continue
        spans.append((max(0.0, e.cue_start - CONFIRM_PAD),
                      e.cue_end + CONFIRM_PAD))
    merged = _merge_spans(spans)
    if not merged:
        return rows, 0, [], 0

    x, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    dur = len(x) / sr
    whisper._load()
    # a szeletek egyszerre a pipeline elé — a megerősítés így kevesebb
    # hívással dolgozik (ugyanazok a szeletek, ugyanaz a kiértékelt szöveg)
    spans = [(a, min(b, dur)) for a, b in merged if min(b, dur) - a >= 0.2]
    if confirm_pack():
        groups = _pack_spans(spans)
        segs = [_pack_audio(g, x, sr) for g in groups]
        if verbose and len(groups) < len(spans):
            print(f"  [megerősítés] {len(spans)} szelet → {len(groups)} "
                  f"csomagolt hívás (becsomagolás)", flush=True)
    else:                            # a régi út: szeletenként egy hívás
        groups = [[(a, b, 0.0)] for a, b in spans]
        segs = [x[int(a * sr):int(b * sr)] for a, b in spans]
    parts = _pipe_windows(whisper, segs, batch or batch_size(),
                          verbose=verbose, label="megerősítés")
    words: list[dict] = []
    for group, local in zip(groups, parts):
        words.extend(_map_words(group, local))
    words.sort(key=lambda w: float(w["start"]))

    _, score_transcript = _auris()
    out: list[dict] = []
    resolved = confirmed = 0
    for i, r in enumerate(rows):
        if i not in pick:                 # a korlát miatt nem hallgattuk meg
            out.append(r)
            continue
        e = by_i.get(r.get("i"))
        ok, alt = (False, None) if e is None else _score_entry(
            e, words, score_transcript)
        if not ok:                        # nem értékelhető: marad, ahogy volt
            out.append(r)
            continue
        confirmed += 1
        if alt is None:                   # a Whisper szerint rendben van
            resolved += 1
        elif float(alt["cer"]) < float(r.get("cer", 1.0)):
            out.append({**alt, "confirmed": True})
        else:
            out.append({**r, "confirmed": True})
    return _sort_rows(out), resolved, merged, confirmed


#: az „üres” sorok második felismerésének kivágás-margója (a szó-időbélyeg
#: csúszása miatt kell mindkét oldalon)
RECOVER_PAD = 0.35


def _write_words_cache(cache: Path, mkv: Path, model_id: str,
                       words: list[dict]) -> None:
    """A szó-gyorsítótár írása (egy helyen, hogy ne legyen két példány)."""
    try:
        cache.write_text(json.dumps(
            {"mkv": str(mkv), "mtime": round(mkv.stat().st_mtime, 3),
             "model": model_id,
             "text": " ".join(w["word"] for w in words),
             "words": words}, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def _recover_enabled(project: Project) -> bool:
    """Kapcsoló: ``asr.recover_empty`` (alapból be)."""
    v = (project.asr or {}).get("recover_empty", True)
    return bool(v)


#: e hossztól a sor saját kivágásán is meghallgatjuk (mért: 2,5 s felett a
#: soronkénti olvasat CER-je 0,166 → 0,118 a 29 s-os ablakossal szemben)
LONG_LINE_S = 2.5


def _recover_empties(entries: list[Entry], rows: list[dict],
                     words: list[dict], wav_path: Path, tr,
                     verbose: bool = True,
                     replace_inside: bool = False) -> tuple[list[dict], int]:
    """Az „üres” sorok második, soronkénti felismerése.

    Mért ok (E04, Parakeet): a 29 üres sorból **mindegyikben** van beszéd
    (RMS −23…−32 dB), és a saját 1–3 s-os kivágásukra a modell mindegyiknél
    ad szöveget („Üdvözöllek!”, „Atrophia.”, „Probléma van.”), a 29 s-os
    ablakban viszont kihagyja ezeket a rövid, halk fordulókat. Nem eltűnt
    hangról van szó, hanem a hosszú kontextusú hívás rövid fordulókat
    elnyelő viselkedéséről — ezért a gyógyszer nem a pad növelése, hanem a
    sor külön meghallgatása.

    Csak a tényleg üres sorokat hallgatjuk meg (más szó nem duplikálódik),
    a szavakat a globális időtengelyre illesztjük, és a meglévő szólistához
    fűzzük. Visszaadja az új szólistát és a helyreállított sorok számát.

    ``replace_inside`` = a hosszú sorok útja: az ablakos olvasat szavai a sor
    **saját** időtartamában cserélődnek a soronkénti olvasatra (a szomszéd
    sorok szavaihoz nem nyúlunk). A hosszú sorokon az ablakos olvasat
    összemossa a szomszéd fordulókat, ezért a saját szelet pontosabb.
    """
    import soundfile as sf

    by_i = {e.i: e for e in entries}
    picks: list[Entry] = []
    for r in rows:
        e = by_i.get(r.get("i"))
        if e is None or e.skip:
            continue
        if e.cue_end - e.cue_start <= 0.0:
            continue
        picks.append(e)
    if not picks:
        return words, 0

    if tr._model is None:                 # Parakeet lusta betöltés
        import onnx_asr

        tr._model = onnx_asr.load_model(tr.model_id).with_timestamps()
    _auris()
    from core.qa import _parakeet_words             # type: ignore

    x, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    dur = len(x) / sr
    added: list[dict] = []
    got = 0
    for e in picks:
        a = max(0.0, e.cue_start - RECOVER_PAD)
        b = min(dur, e.cue_end + RECOVER_PAD)
        if b - a < 0.2:
            continue
        seg = x[int(a * sr):int(b * sr)]
        try:
            res = tr._model.recognize(seg)
        except Exception:                 # egy sor ne vigye el a futást
            continue
        local = _parakeet_words(res.tokens, res.timestamps, len(seg) / sr)
        if not local:
            continue
        got += 1
        if replace_inside:                # csak a sor saját idejébe eső szavak
            words = [w for w in words
                     if not (e.cue_start <= float(w["start"]) < e.cue_end)]
            local = [w for w in local
                     if e.cue_start <= float(w["start"]) + a < e.cue_end]
        for w in local:
            text = str(w.get("word") or "").strip()
            if not text or w.get("start") is None:
                continue
            added.append({"word": text,
                          "start": float(w["start"]) + a,
                          "end": (float(w["end"]) + a)
                          if w.get("end") is not None else None})
    if verbose:
        print(f"  [ASR] üres sorok második felismerése: {got}/{len(picks)} "
              f"sor kapott szöveget, {len(added)} szó", flush=True)
    if not added:
        return words, 0
    merged = sorted(list(words) + added, key=lambda w: float(w["start"]))
    return merged, got


def _threshold(project: Project, model_id: str) -> float | None:
    """A ``gate.asr_cer_max`` a modellhez (szótár esetén család szerint).

    A Parakeet és a Whisper CER-skálája nem összemérhető (a Parakeet
    szótagonként tokenizál, a Whisper szóközösen), ezért a küszöb
    modellenként állítható; a régi, sima szám továbbra is működik.
    """
    thr = (project.gate or {}).get("asr_cer_max")
    if isinstance(thr, dict):
        fam = "parakeet" if "parakeet" in str(model_id).lower() else "whisper"
        thr = thr.get(fam, thr.get("default"))
    return None if thr is None else float(thr)


def check(project: Project, tag: str, entries: list[Entry], mkv: Path,
          verbose: bool = True) -> dict:
    """A kész MKV magyar sávjának felismerése és soronkénti összevetése."""
    t0 = time.time()
    mode = str((project.asr or {}).get("mode") or "whisper").lower()
    if mode not in ("whisper", "parakeet", "hybrid"):
        mode = "whisper"

    wav = project.ep_dir(tag) / "asr_input.wav"
    media.extract_audio(mkv, 0, wav, sr=16000)      # 0 = a mi magyar szinkronunk

    ParakeetTranscriber, Transcriber = _auris_classes()
    if mode in ("parakeet", "hybrid"):
        tr = ParakeetTranscriber()
        label = "Parakeet (gyors előfésülés)"
    else:
        tr = Transcriber.whisper_for("hu")
        label = "Whisper-HU (teljes sáv)"
    if verbose:
        print(f"[ASR] {label}: {mkv.name} (magyar sáv) — mód: {mode}",
              flush=True)
    bsz = batch_size(project)
    model_id = str(getattr(tr, "model_id", "?"))
    # A felismerés ~90 perc a CPU-n: a szavakat elmentjük, hogy egy küszöb-
    # vagy pontozásváltozás után újra lehessen pontozni újrahallgatás nélkül.
    cache = project.ep_dir(tag) / "asr_words.json"
    words: list[dict] | None = None
    heard_text = ""
    try:
        data = json.loads(cache.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    if (data.get("words") and data.get("mkv") == str(mkv)
            and data.get("model") == model_id
            and abs(float(data.get("mtime") or 0) - mkv.stat().st_mtime) < 1.0):
        words = list(data["words"])
        heard_text = str(data.get("text") or "")
        if verbose:
            print(f"[ASR] gyorsítótár: {len(words)} szó újrahasználva "
                  f"({cache.name})", flush=True)
    if words is None:
        # ablakonkénti ellenőrzőpont: megszakítás után nem indul újra a semmiből
        checkpoint = project.ep_dir(tag) / f"asr_windows_{int(mkv.stat().st_mtime)}.jsonl"
        words, heard_text = _transcribe_windowed(tr, wav, verbose=verbose,
                                                 batch=bsz,
                                                 checkpoint=checkpoint)
        try:
            cache.write_text(json.dumps(
                {"mkv": str(mkv), "mtime": round(mkv.stat().st_mtime, 3),
                 "model": model_id, "text": heard_text, "words": words},
                ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
    stats = score_lines(entries, words,
                        model_id=model_id, mkv=mkv,
                        seconds=time.time() - t0)

    # „Üres” sorok második, soronkénti felismerése: a hosszú kontextusú
    # hívás a rövid, halk fordulókat kihagyhatja (mért: E04 Parakeet, 29 sor,
    # mindegyikben van beszéd). A CER-t ez nem lazítja — a szavak valódiak,
    # és a lefedettségi kapu látja a javulást.
    empties = [r for r in (stats.get("rows") or [])
               if r.get("kind") == "empty"]
    recovered = healed = 0
    if mode in ("parakeet", "hybrid") and _recover_enabled(project):
        if empties:
            if verbose:
                print(f"[ASR] {len(empties)} üres sor újrahallgatása "
                      f"soronként…", flush=True)
            words, recovered = _recover_empties(entries, empties, words, wav,
                                                tr, verbose=verbose)
            if recovered:
                stats = score_lines(entries, words, model_id=model_id, mkv=mkv,
                                    seconds=time.time() - t0)

        # Hosszú sorok: az ablakos olvasat összemossa a szomszéd fordulókat.
        # **Szűk** menet: csak a már gyanús (hiba-sávos) hosszú sorokat
        # hallgatjuk újra — mérve ugyanis a teljes hosszú-sávon a haszon
        # kicsi volt a kapu CER-ében (0,248 → 0,245), viszont 9 határeset-sor
        # átbillent. Ahol viszont a kapu pirosat ad, ott a szomszéd-olvasat a
        # valószínű ok, és a saját szelet jobb (89 soron 0,166 → 0,118).
        long_s = float((project.asr or {}).get("long_line_s", LONG_LINE_S) or 0.0)
        if long_s > 0:
            bad_i = {r["i"] for r in (stats.get("rows") or [])}
            longs = [e for e in entries if not e.skip and e.i in bad_i
                     and (e.cue_end - e.cue_start) >= long_s]
            if longs:
                if verbose:
                    print(f"[ASR] {len(longs)} gyanús hosszú sor "
                          f"(≥ {long_s:.1f} s) újrahallgatása soronként…",
                          flush=True)
                words, healed = _recover_empties(
                    entries, [{"i": e.i, "kind": "long"} for e in longs],
                    words, wav, tr, verbose=verbose, replace_inside=True)
                if healed:
                    stats = score_lines(entries, words, model_id=model_id,
                                        mkv=mkv, seconds=time.time() - t0)

    if recovered or healed:               # a gyorsítótár az új szavakat kapja
        _write_words_cache(cache, mkv, model_id, words)
    stats["recovered"] = recovered
    stats["long_reheard"] = healed
    t_primary = time.time() - t0
    stats["mode"] = mode
    stats["primary_seconds"] = round(t_primary, 1)

    if mode == "hybrid" and stats["rows"]:
        if verbose:
            print(f"[ASR] megerősítés: {stats['bad']} gyanús sor a "
                  f"Whisper-HU-val (csak a vitás szakaszok)…", flush=True)
        try:
            tr.unload()
        except Exception:
            pass
        import gc

        gc.collect()                 # a Parakeet ~2,4 GB-ját elengedjük
        whisper = Transcriber.whisper_for("hu")
        max_rows = int((project.asr or {}).get("confirm_max", 0) or 0)
        t1 = time.time()
        rows, resolved, spans, confirmed = _confirm_rows(
            stats["rows"], entries, wav, whisper, max_rows=max_rows,
            verbose=verbose, batch=bsz)
        stats["rows"] = rows
        stats["resolved"] = resolved
        stats["confirmed_rows"] = confirmed
        stats["confirm_max"] = max_rows
        stats["confirm_spans"] = len(spans)
        stats["confirm_seconds"] = round(time.time() - t1, 1)
        stats["primary_model"] = stats["model"]
        stats["confirmed_by"] = str(getattr(whisper, "model_id", "whisper"))
        stats["model"] = (f"{stats['primary_model']} + "
                          f"{stats['confirmed_by'].split('/')[-1]} (megerősítés)")
        counts = {"real": 0, "loop": 0, "empty": 0}
        for r in rows:
            counts[r.get("kind", "real")] = counts.get(r.get("kind", "real"), 0) + 1
        stats["real"], stats["loops"], stats["empties"] = (
            counts.get("real", 0), counts.get("loop", 0), counts.get("empty", 0))
        stats["bad"] = len(rows)
        stats["seconds"] = round(time.time() - t0, 1)
        if verbose:
            print(f"[ASR] megerősítés kész: {resolved} sor feloldva, "
                  f"{stats['bad']} maradt gyanús · "
                  f"{stats['confirm_seconds']}s ({len(spans)} szelet)",
                  flush=True)

    if verbose:
        print(f"[ASR] össz-CER {stats['cer']:.1%} · WER {stats['wer']:.1%} · "
              f"{stats['bad']}/{stats['checked']} sor gyanús "
              f"(CER ≥ {BAD_CER:.0%}) · {stats.get('watch', 0)} figyelendő "
              f"(≥ {WATCH_CER:.0%}) · {stats['real']} valódi · "
              f"{stats['loops']} hurok · {stats['empties']} üres · "
              f"{stats['seconds']}s", flush=True)
    return stats


def apply_gate(metrics: dict, stats: dict, project: Project) -> dict:
    """Opcionális ``asr_cer_max`` kapu a projekt ``gate`` blokkjából.

    Alapból nincs ilyen küszöb (a zenei aláfetchés torzít): a riportban a CER
    mindig látszik, a kapu csak akkor van, ha a projekt kéri. Szótár alakban
    modellenként külön küszöb adható (a Parakeet és a Whisper CER-skálája
    eltér).
    """
    thr = _threshold(project, str(stats.get("model") or ""))
    if thr is None:
        return metrics
    ok = float(stats["cer"]) <= thr
    model = str(stats.get("model") or "")
    tag = "Parakeet-első" if "parakeet" in model.lower() else "Whisper"
    if stats.get("resolved"):
        tag += f" · {stats['resolved']} feloldva"
    gates = [g for g in (metrics.get("gates") or [])
             if g.get("name") not in ("asr", "asr_coverage")]
    gates.append({"name": "asr", "ok": bool(ok),
                  "detail": f"CER {stats['cer']:.3f} (max {thr:.3f}) · "
                            f"{stats['bad']} gyanús sor · "
                            f"{stats.get('watch', 0)} figyelendő · {tag}"})
    # Lefedettségi kapu: a CER csak a felismert sorokból számol, ezért külön
    # kell látni, mennyi maradt ellenőrizetlen — enélkül a mérés lazítása
    # zöldre vihetné a kaput.
    cov = stats.get("coverage")
    cov_min = float((project.gate or {}).get("asr_coverage_min", 0.85))
    if cov is not None:
        total = int(stats.get("lines_total") or 0)
        done = int(stats.get("recognized") or 0)
        gates.append({"name": "asr_coverage",
                      "ok": float(cov) >= cov_min,
                      "detail": f"{done}/{total} sor ellenőrizve "
                                f"({float(cov):.3f}, min {cov_min:.2f})"})
    metrics["gates"] = gates
    metrics["passed"] = all(bool(g.get("ok")) for g in gates)
    return metrics
