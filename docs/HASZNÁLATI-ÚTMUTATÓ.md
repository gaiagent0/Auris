# Auris — Teljes használati útmutató (hangoskönyv és szinkronfilm)

> Snapdragon X Elite / Windows 11 ARM64 natív build. Frissítve: 2026-11-25.
> Az útmutató a `reader/app.py` útvonalaiból, az `auris_cli.py` és a
> `python -m dubstudio` `--help`-eiből származik — minden parancs és URL
> valósan futtatható ezen a gépen.

---

## 0. Első indítás

```bat
cd /d "%USERPROFILE%\Auris"
reader\setup.bat       :: csak egyszer — venv + minden függőség és AI-modell
reader\run.bat         :: indítás — a TTS modell háttérben tölt, az app rögtön nyílik
```

> **Ha a `setup.bat` a „not recognized as an internal or external command”
> hibát ad**, akkor a Windows `NoDefaultCurrentDirectoryInExePath=1`
> környezeti változója miatt a parancssor nem keresi az aktuális könyvtárban
> a batch fájlokat. Ilyenkor a telepítőt teljes úttal futtasd
> (`reader\setup.bat` a repo gyökeréből), vagy a változó értékét állítsd 0-ra.

- Web UI: **http://127.0.0.1:7860** — könyvtár és hangoskönyv-készítés
- Szinkronstúdió: **http://127.0.0.1:7860/szinkron** (film/sorozat) és
  **http://127.0.0.1:7860/felulvizsgalat** (soronkénti emberi ellenőrzés); a
  `dubstudio` munkafájljait használja — a telepítés a 6. szakaszban
- CLI: `reader\.venv\Scripts\python.exe reader\auris_cli.py ...`
- A Supertonic modell a `reader\models\supertonic-3\` alatt van (fp32);
  induláskor háttérben töltődik, a felület addig is használható.

**Kulcs-beállítások (Settings → Beszédmotorok):** `tts_engine=auto`
(→ `supertonic_sherpa`), `supertonic_batch=4`, `supertonic_steps=10`,
`supertonic_variant=fp32`.

> A `tts_engine`-t **ne írd át tartósan** `supertonic`-ra — az `auto` a
> hardver-javaslatot adja, és a végpont-végpont futtatás ebből indul.

---

## 1. Web UI — a teljes folyamat

Minden oldal a **http://127.0.0.1:7860** alatt fut.

| Oldal | Útvonal | Mi van itt |
|---|---|---|
| Könyvtár | `/` | importált könyvek, import-űrlap, export-feladatok |
| Olvasó | `/reader/<id>` | felolvasás, lejátszás, felirat, szókiemelés |
| Hangstúdió | `/voice-studio/<id>` | szereplőhangok, referenciahang, klónozás |
| Produkció | `/production/<id>` | lépéssor, fejezetenkénti állapot, export |
| Ellenőrzés | `/qa/<id>` | minőségellenőrzés, változatok, jóváhagyás |
| Beállítások | `/settings` | motor, batch, lépésszám, hangprofilok, LLM |
| Feladatok | `/jobs` | generálás, export, újraelemzés állapota |
| Dokumentáció | `/docs` | a beépített útmutató |
| Szinkron | `/szinkron` | film/sorozat szinkronlánca (dubstudio) |
| Felülvizsgálat | `/felulvizsgalat` | soronkénti beszélő-ellenőrzés |

### 1.1 Munkafolyamat (első könyv)

1. **Könyvtár (`/`)** → „Könyv hozzáadása” → EPUB/PDF/DOCX/TXT/PRC/MOBI vagy
   nyilvános cikk URL-je.
2. **Hangstúdió (`/voice-studio/<id>`)** → szereplőnként (és a narrátornál)
   hang választása. Beépített: F1–F5 (női), M1–M5 (férfi); klón: feltöltött
   referencia WAV az XTTSv2-ONNX motornál (lásd 3. szakasz).
3. **Produkció (`/production/<id>`)** → a teljes könyv vagy a kijelölt
   fejezetek generálása; a haladás a Feladatok oldalon követhető.
4. **Ellenőrzés (`/qa/<id>`)** → visszahallgatás, hibás mondatok, változatok,
   jóváhagyás.
5. **Export** → WAV/MP3/Opus/FLAC/M4B, opcionális SRT/ASS felirat,
   fejezetkiválasztás, kiadási csomag.

---

## 2. CLI — pontosan

Az `auris` (`reader\auris_cli.py`) parancssor; `--server`-rel egy futó
szerverhez csatlakozik, `--start`-tal maga indítja el a szervert, ha nem fut.

```bat
set AURIS=reader\.venv\Scripts\python.exe reader\auris_cli.py
"%AURIS%" --help
```

### 2.1 `books`

```bat
"%AURIS%" books
```

A könyvek listája (azonosító, cím, állapot).

### 2.2 `import`

```bat
auris import [--title T] [--author A] [--language L] [--mode single|characters] [--wait] FÁJL
```

- `FÁJL` — EPUB/PDF/DOCX/TXT/PRC/MOBI.
- `--wait` — megvárja a szereplőelemzés végét.
- Kimenet: a könyv azonosítója, amit a többi parancs használ.

### 2.3 `generate`

```bat
auris generate [--chapters 1,3,5-8 | all] KÖNYV_ID
```

A megadott fejezetek hangját elkészíti; a már kész (gyorsítótárazott)
mondatokat nem generálja újra.

### 2.4 `qa` — minőségellenőrzés

```bat
auris qa [--chapter N] [--no-asr] [--no-regenerate] KÖNYV_ID
```

Beszédfelismeréssel visszaolvassa a hangot, WER/CER-t számol, és jelzi a
vágást, a hosszú szünetet, a túl gyors vagy túl lassú mondatokat. A `--no-asr`
kihagyja a visszaolvasást, a `--no-regenerate` nem generálja újra a hibás
mondatokat.

### 2.5 `export`

```bat
auris export [--format wav|mp3|m4b|opus|flac] [--subtitles none|srt|ass]
             [--package none|epub3|audiobookshelf|acx|daw] [--chapters 1,3,5-8]
             [--intro] [--outro] [--sample] [--upload] KÖNYV_ID
```

- `--package audiobookshelf` — Audiobookshelf-kompatibilis mappastruktúra;
  `--upload` feltölti a Beállításokban megadott szerverre.
- `--sample` — rövid bemutató minta az exportbeállításokkal.

### 2.6 `speak` — gyors teszt

```bat
auris speak [-o OUT] [--voice F1] [--format mp3|wav|opus|flac|aac] [--speed 1.0] [--language hu] "Szöveg"
```

Egy szöveget szintetizál fájlba — a leggyorsabb mód a motor ellenőrzésére:

```bat
"%AURIS%" speak --format wav -o teszt.wav "Szia, ez egy teszt a magyar motoron."
```

---

## 3. Hangok és klónozás

| Hangforrás | Mi ez | Kimenet |
|---|---|---|
| **Beépített Supertonic** | F1–F5 (női), M1–M5 (férfi) | `tts_engine=auto`, választás a Beállításokban |
| **Sherpa-onnx Supertonic int8** | **a leggyorsabb motor** (mért RTF 0,14), a Snapdragon `auto`-javaslata; `tts_engine=supertonic_sherpa` | modell automatikus HF-letöltés a `reader\models\supertonic-sherpa-int8\` alá |
| **Magyar Piper-hangok** (anna, berta, imre) | külső ONNX, `reader\models\piper\` (vagy a `piper_models_dir` beállítás) | ONNX Runtime; a fonémázás ARM64-en nyitott kérdés |
| **XTTSv2-ONNX klón** | a klónozó motor: magyar zero-shot, tiszta ONNX, WER 0,18 % | referencia WAV a Hangstúdióban; a modell a `vendor\XTTSv2-Streaming-ONNX\` vagy a `VOICEAI_XTTS_DIR` alatt (CC BY-NC 4.0) |

**Klón-korlát:** a sherpa-onnx Supertonic a referenciahang-klónozást **nem
támogatja** (fix hangkészlet). Saját hang klónozásához az **XTTSv2-ONNX** a
követendő út.

A készülék-audit (ASD) és a Parakeet ASR a filmes ellenőrzéshez is használatos;
ezek a `reader\docs\meresek.md` bejegyzéseiben szerepelnek.

---

## 4. Mérés és ellenőrzés

- **WER/CER a magyar korpuszon** (24 mondat):
  `reader\.venv\Scripts\python.exe -m core.hu_wer` (a `reader` mappából)
- **ASR-ellenőrzés tetszőleges WAV-készletre**:
  `reader\scripts\asr_check_wavs.py --json cases.json --wav-dir <mappa>`
- **Serial vs batchelt minták** (emberi ellenőrzésre):
  `reader\scripts\export_test_samples.py`
- **Teljes anyag**: `reader\docs\meresek.md` (M-01…M-45)

A **Parakeet ONNX ASR** (`nemo-parakeet-tdt-0.6b-v3`) az
`istupakov/parakeet-tdt-0.6b-v3-onnx` Hugging Face repóról töltődik be
(25 európai nyelv, köztük magyar; CC-BY-4.0).

---

## 5. Fontos fájlok

| Mi | Hol |
|---|---|
| App belépési pont | `reader/app.py` |
| CLI | `reader/auris_cli.py` |
| Hangmotorok | `reader/core/local_engines.py` (Supertonic, generate_many) |
| ASR/VAD | `reader/core/qa.py` (Transcriber, vad_speech_spans) |
| Gyorsítótár | `reader/audio_cache/` |
| Modellek | `reader/models/supertonic-3/` (fp32), `reader/models/supertonic-sherpa-int8/` |
| Szinkron API | `reader/core/dub_api.py`, `lines_api.py`, `subtitle_upload.py` |
| Közös AI-modellek | `scripts/fetch_models.py` (`--check` = ellenőrzés) |
| Szinkron felület | `reader/templates/szinkron.html`, `felulvizsgalat.html` |
| Doku | `reader/docs/` — **meresek.md** (mérési bizonyítékok), **arm64-install.md** (telepítés) |

---

## 6. Szinkronstúdió — film és sorozat

A szinkronfelület a **dubstudio** láncot használja, amely **a repó része**
(`dubstudio/`). A részletes útmutató:
[`SZINKRON-TELEPITES.md`](SZINKRON-TELEPITES.md); az állapotot a
`scripts\check_dubstudio.py` írja ki.

Rövid lépések:

1. **A dubstudio a repóban van** (`dubstudio\dubstudio\`); a `projects`,
   `work`, `movies`, `series` mappákat a telepítő hozza létre. Más helyen a
   `DUBSTUDIO_ROOT` adja meg a gyökeret. A CLI-t az Auris indítja, külön nem
   kell futtatni.
2. **Csomagok** — a `reader\setup.bat` már feltelepíti a közös venvbe a
   szinkron csomagjait is (`transformers`, `audio-separator`); utólag:
   `... -m pip install -r reader\requirements.txt`.
3. **Programok:** `ffmpeg`, `ffprobe` a PATH-on és `espeak-ng` a Piper-hangokhoz.
4. **Modellek** — a telepítő a `scripts\fetch_models.py`-jal letölti a közös
   Supertonic 3-at és a szinkron modelleket (pyannote szegmentáló, CAM++
   embedding, Piper hu_HU hangok, UVR/MDX) a `dubstudio\work\models\` alá.
   Utólag: `... scripts\fetch_models.py` (`--check` = ellenőrzés).
5. **Projekt:** `/szinkron` → új film varázsló, vagy
   `... scripts\check_dubstudio.py --init "Film címe" --kind movie`.
6. **Fordítás:** csak helyi modell (alapból a helyi Ollama,
   `http://127.0.0.1:11434/v1`, `qwen3.5:9b`). Külső végpontot a fordító
   hibával elutasít — a repó nem küld feliratszöveget külső szolgáltatásnak.

A futás sorrendje: **intake** (felirat + vokál/M&E) → **prepare** (beszélőfordulók)
→ **cluster** (szereplők, hangzár) → **dub** (terv, szintézis, keverés) → **mux**
(`work\out\*.mkv`) → **QA-kapuk**. A kész szinkron a `/felulvizsgalat` oldalon
soronként ellenőrizhető; a döntés a következő futásba íródik.

---

## 7. Docker — a hordozható alternatíva

A fenti lépések a **natív venv**-et használják; ez az ajánlott út a Snapdragondon.
Másik gépen (x64 Linux szerver, más ARM64 gép) a Docker a megfelelő:

```bat
docker compose up -d --build
:: majd: http://127.0.0.1:7860
```

- A kép **CPU-only**: minden motor ONNX Runtime-on fut, nem kell NVIDIA.
- A modellek a `docker-data\models` mappából jönnek bind mounttal, így a
  hoszton letöltött modellek azonnal használhatók, és a konténer
  újraépítésekor nem kell újra letölteni őket.
- **A modell bekötése a konténerben is automatikus** — az első hanggeneráláskor
  a sherpa-Supertonic int8 modell (~139 MB) a
  `docker-data\models\supertonic-sherpa-int8\` mappába töltődik, a Hugging Face
  gyorsítótárat pedig a `huggingface` named volume tartja. A letöltést a
  `docker compose logs -f` mutatja.
- A könyvek, exportok és a hangcache a `docker-data\` alatt marad.
- **Ha a 7860-as port foglalt** (például a natív `run.bat` app fut rajta),
  írj egy `docker-compose.override.yml`-t a repó gyökerébe (gitignore-olt):
  ```yaml
  services:
    auris:
      ports: !override
        - "127.0.0.1:7862:7860"
  ```
  A `!override` lecseréli az alap portlistát (nem fűzi hozzá). Így a két út
  egyszerre futhat: natív a 7860-on, konténer a 7862-n.
- Ha a Docker Engine nem indul el, a javítási lépések:
  [`docker-wsl.md`](docker-wsl.md) — vagy egy parancsban:
  `bash scripts/windows/docker-desktop-repair.sh restart`.

> A Docker-kép az Aurist futtatja; a szinkronstúdióhoz a dubstudio modelljei
> és a külső programok (ffmpeg, espeak-ng) is kellenek (6. szakasz), ezért a
> szinkron út a natív telepítés.

---

## 8. Narráció — egyszereplős hangoskönyv (videó → magyar narráció)

A **Narráció** oldal (`/narracio`) egy egyszereplős, magyar narrációval
felülhangosított MKV-t készít egy videóból + magyar feliratból. Nem
szereplőkezelő (az a 6. szakasz szinkronja), hanem egyetlen, folyamatos hang,
amely a videó hangját (vagy annak feliratát) magyarul szólalja meg — ideális
hangoskönyv-szerű kimenethez vagy egyetlen narrátoros videóhoz.

### 8.1 Mi kell hozzá

- egy videófájl (helyi út vagy YouTube URL),
- hozzá egy magyar felirat (`.srt`/`.vtt`), vagy a beépített ASR-útvonal,
- egy TTS-hang: beépített Supertonic (`SUP_M1`…`SUP_M5`, `SUP_F1`…`SUP_F5`)
  vagy XTTSv2-ONNX klón (`--ref minta.wav`),
- `ffmpeg`/`ffprobe` a PATH-ban (a kimenet muxolásához).

### 8.2 Web UI — `/narracio`

1. **Forrás** — videó kiválasztása (helyi fájlválasztó vagy útvonal),
   felirat (SRT/VTT) feltöltése vagy ASR.
2. **Hang** — legördülőből; klónhoz feltölthető referencia-WAV.
3. **Nyelv** — alap `hu`.
4. **Próba (limit)** — „Csak próba” jelölő: csak az első N blokk készül el;
   a kimenet neve `… - HUN NARRACIO PROBA.mkv`, és a riport `limited: true`
   jelzi, hogy ez nem kész munka. A videó a próba végéig vágódik (nincs csend
   a végén).
5. **Indítás** — a háttérfolyamat a `dubstudio narrate` parancsot futtatja;
   állapot + napló 3 mp-ként frissül (`/api/narration/status`).
6. **Eredmény** — MKV lejátszás/letöltés, hangfájl, riport (blokkok, szint,
   hossz, LUFS/csúcs). A fejlécben a `próba` jelző és a blokkszám `2/62`
   alakban jelzi a korlátozott futást.

### 8.3 CLI

```bat
cd /d "C:\Users\istva\Dev\portfolio\Projects\audiobook narrator\auris\dubstudio"
..\reader\.venv\Scripts\python.exe -m dubstudio narrate ^
  "movies\...\data\source.mp4" ^
  --srt "movies\...\subs\E01_hun.srt" ^
  --voice SUP_M4 ^
  --work "work\narration\20261127-153045-source"
```

Fő kapcsolók: `--out`, `--work`, `--voice`, `--ref` (klón), `--lang`, `--srt`,
`--limit N` (próba), `--block-chars`, `--block-max`, `--gap`, `--models`,
`--replace-audio`, `--no-mux`, `--force`.

### 8.4 Munkamappa újrahasznosítása

A Web UI (és a futás) nem készít mindig új timestampes mappát: ha már van
ugyanahhoz a forráshoz + hanghoz + nyelvhez + klónmintához tartozó munkamappa,
**újrahasznosítja** azt, és a kész blokkokat a gyorsítótárból veszi. A teljes
(nem próba) futást előnyben részesíti a frissebb próbával szemben. A UI
kiírja: *„Korábbi munkamappa újrahasznosítva — a kész blokkok a
gyorsítótárból jönnek.”*

### 8.5 Riportmezők (`report.json`)

`limited`, `limit`, `blocks_total`, `blocks`, `synth`, `mix`, `audio`,
`output`, `seconds`, `trimmed`, `trimmed_to`. A `limited: true` jelzi a próba
futást — egy próba ne tűnjön kész munkának.

### 8.6 API-végpontok

| Végpont | Mit csinál |
|---|---|
| `GET /narracio` | a Narráció oldal |
| `GET /api/narration/voices` | választható hangok + klón lehetőség |
| `POST /api/narration/upload` | videó/hangminta/felirat feltöltése |
| `POST /api/narration/pick` | natív fájlválasztó (tkinter) |
| `POST /api/narration/start` | narráció indítása (`limit` támogatott) |
| `GET /api/narration/status` | állapot + napló-tail + riport |
| `POST /api/narration/stop` | futó feladat leállítása |
| `GET /api/narration/file?path=…` | kimeneti fájl megnyitása/letöltése |
