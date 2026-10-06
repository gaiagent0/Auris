# Auris — Teljes használati útmutató (Web UI + CLI)

> Snapdragon X Elite / Windows 11 ARM64 natív build. Frissítve: 2026-10-06.
> Ez az útmutató a `reader/app.py` útjaielk és az `auris_cli.py` `--help`-ekből
> származik — minden parancs és URL valósan futtatható ezen a gépen.

---

## 0. Első induló

```bat
cd /d "C:\Users\istva\Dev\portfolio\Projects\audiobook narrator\auris"
reader\setup.bat       :: csak egyszer — venv + függőségek
reader\run.bat         :: induló — a TTS modell hátérterülben betölt, az app rögtön nyit
```

- Web UI: **http://127.0.0.1:7860**
- CLI: `reader\.venv\Scripts\python.exe reader\auris_cli.py ...`
- A modell a `reader\models\supertonic-3\`-ba van (fp32) — induláskor hátérterülben betölt, a lap ablaktív, amíg fut.

**Kulcs-beállítások (Settings → Beszédmotorok):** `tts_engine=auto` (→ Supertonic 3),
`supertonic_batch=4`, `supertonic_steps=10`, `supertonic_variant=fp32`.
> A `tts_engine`-t **ne írd át tartósan** `supertonic`-ra — az auto a hardver-javaslatot
> adja, és a végpont-végpont futtatás ebből a beállításból szól.

---

## 1. Web UI — a teljes folyamat

Mind a képek a **http://127.0.0.1:7860**-on futnak.

| Kép | Útvjal | Mi van itt |
|---|---|---|
| Könyvtár | `/` | imported könyvek listája, import-form, export-jobs |
| Olvasó | `/reader/<id>` | a könyv felolvasása, hang lejátszás, felirat színkron |
| Hangstúdió | `/voice-studio/<id>` | szereplő hangjai, referenciahang-kiválasztás, klón |
| Beállítások | `/settings` | motor, batch, lépésszám, hangprofilok, LLM-kapcsolat |
| Dokumentáció | `/docs` | a beépített útmutató a felületen |
| Állapotnapló | `/api/jobs` + `/jobs` | export- és generálási feladatok |

### 1.1 Munkafolyamat (első könyv)

1. **Könyvtár (`/`)** → „Import book" → EPUB/PDF/DOCX/TXT fájl.
   - Import parancssorból: `python auris_cli.py import konyv.epub --wait` (a `--wait` a
     szereplőelemzésre várja; nálad lezárdalom meg a felületen).
2. **Hangstúdió (`/voice-studio/<id>`)** → karakterenként hang kiválasztás.
   - Beépített: F1–F5 (női), M1–M5 (férfi).
   - Klón: „Upload WAV reference" → az XTTSv2-ONNX úton a hang klónozódik (lásd §3).
3. **Olvasó (`/reader/<id>`)** → a fejezet felolvasása, hang lejátszás.
4. **Export** → WAV/MP3/Opus/FLAC/M4B, opcionális SRT/ASS felirat, fejezetkiválasztás.

### 1.2 A beépített API (a frontend használja ezt)

| Móds | Útvjal | Szolgál |
|---|---|---|
| POST | `/api/books/import` | import |
| GET | `/api/books` | könyvlista |
| PUT/DELETE | `/api/books/<id>` | könyv javítás/törlés |
| GET | `/api/books/<id>/chapters` | fejezetek |
| POST | `/api/tts/generate` | hanggenerálás |
| GET | `/api/tts/status` | TTS-modell állapota |
| POST | `/api/tts/load` | modell betöltése |
| GET | `/api/audio/<cache_key>` | a gyorsítótárazott WAV |
| GET | `/api/jobs` / POST `/api/jobs/<id>/cancel` | export-feladatok |
| GET | `/docs` | dokumentáció |

---

## 2. CLI — pontosan

Az `auris` (`reader\auris_cli.py`) parancssor, `--server`-rel a futó szerverhozzósz. Ha
nem fut szerver, `--start` indíta be belsőleg (a szövegfik virágon felhasználja).

```bat
set AURIS=reader\.venv\Scripts\python.exe reader\auris_cli.py
"%AURIS%" --help
```

### 2.1 `books`
```bat
"%AURIS%" books
```
A könyvek listája (id, teitel, állapota).

### 2.2 `import`
```
auris import [--title T] [--author A] [--language L] [--mode single|characters] [--wait] FILE
```
- `file` — EPUB/PDF/DOCX/TXT/MOBI/PRC.
- `--wait` — a szereplőelemzés **befejezéséhez** (nálad lezárdalom meg).
- Kimenet: könyv-ID, amit a többi parancsból használhatod.

### 2.3 `generate`
```
auris generate [--chapters 1,3,5-8 | all] BOOK_ID
```
A megadott fejezetek hangját elkészíti. A gyorsítótárazott mondatokot nem generálja újra.

### 2.4 `qa` — minőségellenőrzés
```
auris qa [--chapter N] [--no-asr] [--no-regenerate] BOOK_ID
```
Visszaírja ASR-rel (magyar), számolja WER/CER, jelzi a vágás/hosszú szünet/túl gyors vagy
lassú mondatokot. A `--no-asr` a visszaírást kihagyja, a `--no-regenerate` a hibás mondatokot
nem generálja újra.

### 2.5 `export`
```
auris export [--format wav|mp3|m4b|opus|flac] [--subtitles none|srt|ass]
             [--package none|epub3|audiobookshelf|acx|daw] [--chapters 1,3,5-8]
             [--intro] [--outro] [--sample] [--upload] BOOK_ID
```
- `--package audiobookshelf` — Audiobookshelf-compatibilis mappastruktúra; `--upload` feltölti.
- `--sample` — egy rövid bemutató minta (az export beállítások vétele).

### 2.6 `speak` — gyors teszt
```
auris speak [-o OUT] [--voice F1] [--format mp3|wav|opus|flac|aac] [--speed 1.0] [--language hu] "Szöveg"
```
Egy szöveget szintetizálja fájlba — a leggyorsabb mód a motor ellenőrzésére:
```bat
"%AURIS%" speak --format wav -o teszt.wav "Szia, ez egy teszt a magyar motoron."
```

---

## 3. Hangok és klónozás

| Hangforrás | Mi ez | Kimenet |
|---|---|---|
| **Beépített Supertonic** | F1–F5, M1–M5 | `tts_engine=auto`, beállítási mezőben |
| **Magyar Piper-hang** (anna, berta, imre) | külső ONNX, `D:\VoiceAI\apps\hu-voice-ai\models\piper\` | ONNX Runtime-tal fut, de a phonemizálás ARM64-en nyitott kérdés |
| **XTTSv2-ONNX klón** | a **legjobb klónozó**: magyar zero-shot, tiszta ONNX, WER 0,18 %, 45,4 perc/óra | referencia WAV a Hangstúdióban; a modell `D:\XTTSv2-Streaming-ONNX` (CC BY-NC 4.0) |
| **Sherpa-onnx Supertonic int8** | a **leggyorsabb motor** (RTF 0,14), de **beépített hangokon fut, nem klónoz** | `D:\VoiceAI\envs\sherpa-arm64` + `hub\scripts\sherpa_supertonic_full.py` |

**Klón-korlát:** a sherpa-onnx Supertonic a referenciahang-klónozást **nem támogatja**
(fixed-voice, voice.bin). A custom hangklónhoz az **XTTSv2-ONNX** a követendő út.

---

## 4. Mérés és ellenőrzés

- **WER/CER a magyar korpuszon** (24 mondat): `reader\.venv\Scripts\python.exe -m core.hu_wer`
- **ASR-ellenőrzés tetszőleges WAV-készletre**: `reader\scripts\asr_check_wavs.py --json cases.json --wav-dir <dir>`
- **Serial vs batchelt minták** (emberi ellenőrzésre): `reader\scripts\export_test_samples.py`
- **Teljes anyag**: `reader\docs\meresek.md` (M-01…M-45)

A **Parakeet ONNX ASR** (`nemo-parakeet-tdt-0.6b-v3`) a `onnx_asr` resolver-jével a
**`istupakov/parakeet-tdt-0.6b-v3-onnx`** HF-repóról betölt (25 európai nyelv, magyarral,
licenc CC-BY-4.0). Első használtnál automatikusan tölte be.

---

## 5. Hasznos fájlokon

| Mi | Hol |
|---|---|
| App-fejlop | `reader/app.py` |
| CLI | `reader/auris_cli.py` |
| Hangmotor | `reader/core/local_engines.py` (SupertonicEngine, generate_many) |
| ASR/VAD | `reader/core/qa.py` (Transcriber, vad_speech_spans) |
| Gyorsítótár | `reader/audio_cache/` |
| Modellok | `reader/models/supertonic-3/` (fp32), `...-int8/` |
| Doku | `reader/docs/` — **meresek.md** (bizonyítékkatalógus), **handoff-2026-10-03.md** |