# Szinkronstúdió — telepítés és beállítás friss klónon

> Az Auris `/szinkron` és `/felulvizsgalat` oldala a **dubstudio** szinkronláncot
> használja. A dubstudio **a repó része** (`dubstudio/`): a futást, a
> klaszterezést, a szintézist és a muxolást a CLI-je végzi, az Auris a felületet,
> a lejátszót és az emberi döntések rögzítését adja hozzá.
>
> Ez az útmutató a friss klón utáni állapotból indul. A `reader\setup.bat`
> **a közös telepítő**: létrehozza a `dubstudio` futásidőbeli mappáit,
> feltelepíti a hangoskönyv és a szinkron közös Python-csomagjait, és letölti a
> közös AI-modelleket is. Ami utána kell, az a két külső program (`ffmpeg`,
> `espeak-ng`) és a projekt létrehozása. Minden parancs a **repo gyökeréből**
> fut, az Auris venv-jével (`reader\.venv\Scripts\python.exe`).

---

## 0. Gyors ellenőrzés — mi hiányzik?

```bat
reader\.venv\Scripts\python.exe scripts\check_dubstudio.py
```

A program soronként kiírja, hogy megvan-e a dubstudio, a Python csomagok, az
`ffmpeg`/`ffprobe`, az `espeak-ng`, az `audio-separator`, a **közös modellek**
(Supertonic, pyannote, CAM++, Piper, UVR), a projektek és a projektfájlokban
hivatkozott modellek. Kilépési kódja `1`, ha bármi hiányzik; `--json`-nal gépi
kimenetet ad. A hiányzó tételeket az alábbi szakaszok pótolják — a csomagokat
és a modelleket a `reader\setup.bat` (illetve a `scripts\fetch_models.py`)
telepíti.

---

## 1. A dubstudio helye

A szinkronlánc **a repó része**, a kód a `dubstudio\dubstudio\` alatt van:

```text
auris\                 ← ez a repó
|-- dubstudio\         ← a szinkronlánc
|   |-- dubstudio\     ← a Python csomag (CLI, klaszterezés, szintézis, mux)
|   |-- tests\
|   |-- docs\
|   |-- projects\      ← futásidőben: projektenként egy JSON
|   |-- work\          ← futásidőben: munkafájlok, modellek, kimenet (out\)
|   `-- movies\        ← futásidőben: filmforrások (a projekt root-ja)
`-- reader\            ← a hangoskönyv és a webes felület
```

A `projects\`, `work\`, `movies\`, `series\` és `out\` mappát nem a repó
tartalmazza: a `reader\setup.bat` (azaz a `reader\setup.py`
`prepare_sync_stack()` lépése) hozza létre, üresen. Kézzel is pótolhatók.

Ha a szinkronláncot máshol akarod tartani, állítsd be a `DUBSTUDIO_ROOT`
környezeti változót a gyökerére (ugyanazt a feloldást használja a felület, az
ellenőrző szkript és a modellletöltő):

```bat
set DUBSTUDIO_ROOT=D:\dubstudio
```

---

## 2. Python csomagok — a telepítő elvégzi

A szinkronlánc **ugyanazt** a virtuális környezetet használja, mint az Auris —
nincs második venv. A `reader\setup.bat` (azaz a `reader\setup.py`) a
hangoskönyv csomagjai mellé ezeket is telepíti:

| Csomag | Mihez kell | Ki használja |
|---|---|---|
| `onnxruntime` | Supertonic, diarizáció, embedding modellek futtatása | hangoskönyv + szinkron |
| `sherpa-onnx` | pyannote szegmentáló + Supertonic TTS | hangoskönyv + szinkron |
| `numpy`, `scipy`, `soundfile` | tömbkezelés, újramintavételezés, WAV I/O | hangoskönyv + szinkron |
| `torch` | a nem-osztályozó modell és a Whisper QA | hangoskönyv + szinkron |
| `transformers` | a beszélő nemét adó modell betöltése | szinkron |
| `audio-separator` | vokál és zene+effekt (M&E) szétválasztás az intakenél | szinkron |

Ha a telepítést később futtatod (vagy csak a csomagokat akarod pótolni):

```bat
reader\.venv\Scripts\python.exe -m pip install -r reader\requirements.txt
```

Az `audio-separator` a venv `Scripts\` mappájába kerül; az intake ott is keresi,
nem kell a PATH-ra tenni. A telepítés után **indítsd újra** az Aurist: a
`/szinkron` oldal a dubstudio moduljait a szerver indításakor importálja.

---

## 3. Programok

| Program | Mihez kell | Telepítés |
|---|---|---|
| `ffmpeg` + `ffprobe` | hangsáv kinyerés, `atempo`, MKV mux | <https://ffmpeg.org/download.html>, majd PATH |
| `espeak-ng` | a Piper-hangok IPA fonémázása | <https://github.com/espeak-ng/espeak-ng/releases> (Windows-telepítő) |

Ellenőrzés:

```bat
ffmpeg -version
ffprobe -version
espeak-ng --version
```

---

## 4. Modellek — a telepítő letölti

A modellek helyét a **projektfájl `models` szakasza** adja meg (abszolút vagy a
projekt `root`-jához viszonyított útvonal). A letöltést a `scripts\fetch_models.py`
végzi, amit a `reader\setup.bat` automatikusan lefuttat; külön:

```bat
:: minden közös + szinkron modell (ami megvan, kimarad)
reader\.venv\Scripts\python.exe scripts\fetch_models.py

:: csak ellenőrzés, hálózat nélkül
reader\.venv\Scripts\python.exe scripts\fetch_models.py --check

:: csak a szinkron modellek, más célmappába
reader\.venv\Scripts\python.exe scripts\fetch_models.py --only sync --models-dir D:\dubstudio\work\models
```

| Kulcs | Modell | Honnan | Méret |
|---|---|---|---|
| `supertonic` | Supertonic 3 sherpa-onnx int8 (10 hang) | Hugging Face; a szkript a `reader\models\supertonic-sherpa-int8\` alá tölti — **ugyanaz a motor**, mint a hangoskönyvben | ~139 MB |
| `diar_segmentation` | pyannote segmentation 3.0 (sherpa-onnx) | `k2-fsa/sherpa-onnx` kiadás: `sherpa-onnx-pyannote-segmentation-3-0.tar.bz2` → `work\models\diar\` | ~6 MB |
| `embedding` / `diar_embedding` | CAM++ / 3D-Speaker ERes2Net beszélő-embedding (`3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx`) | ugyanonnan: `speaker-recongition-models` → `work\models\` | ~28 MB |
| `piper` | Piper hu_HU hangok (`anna`, `berta`, `imre`) | Hugging Face `rhasspy/piper-voices` → `work\models\piper\` | ~63 MB/hang |
| `espeak` | az `espeak-ng` futtatható fájlja | 3. szakasz (külső program) | — |
| `intake.separate.model_dir` | `UVR_MDXNET_Main.onnx` (vokál leválasztás) | `TRvlvr/model_repo` kiadás → `work\models\separate\` | ~50 MB |
| – | nem-osztályozó (`alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech`) | Hugging Face, az első futásnál automatikusan | ~1 GB |

A modelleket nem kell a repóba tenni: a `work\models\` (vagy egy közös
modellmappa) elég, a projektfájlban csak az útvonal szerepel. A szkript a
letöltés után kiírja a használandó útvonalakat; a
`check_dubstudio.py --init` ezekkel tölti ki az új projekt `models` szakaszát.

---

## 5. Projekt létrehozása

Egy film vagy sorozat **egy projekt**. Két út van:

**A felületen:** `/szinkron` → új film varázsló (név, fájl, forrásnyelv) — ez
létrehozza a projektet, elhelyezi a médiát és elindítja az intake-et.

**Parancssorból** (ha kézzel akarod szerkeszteni):

```bat
reader\.venv\Scripts\python.exe scripts\check_dubstudio.py --init "Film címe" --kind movie
reader\.venv\Scripts\python.exe scripts\check_dubstudio.py --init "Sorozat neve" --kind series --season 2 --source-lang heb
```

A szkript a dubstudio `projects\<slug>.json` fájlját írja a gépen talált
modellútvonalakkal, a mért alapbeállításokkal (`tuning`, `gate`) és a helyi LLM
fordítóval. Ugyanezt a projektgyárat használja a felület varázslója és a
dubstudio CLI is; a CLI a projektet `--project` → `DUBSTUDIO_PROJECT` → az
egyetlen `projects\*.json` sorrendben keresi (a repóban nincs beégetett projekt). Utána a fájl szabadon szerkeszthető — a Beállítások fül a felületen
is ugyanezt a JSON-t módosítja (`models`, `tuning`, `gate`, `llm`, `asr`).

A fontosabb mezők:

| Mező | Jelentés |
|---|---|
| `root` | a film/sorozat gyökere (itt van `data\`, `subs\`) |
| `cache_dir` | a munkakönyvtár (`work\<projekt>`): `out\`, `cast-*.json`, epizódmappák |
| `data_dir`, `subs_dir` | a médiafájlok, illetve a `subs\<EP>_<nyelv>.srt` feliratok |
| `vocals_globs`, `me_globs` | hol van a leválasztott énekbeszéd, illetve a zene+effekt sáv |
| `source_lang` | `eng` vagy `heb` — a fordítás forrása |
| `original_audio` | melyik hangsáv a forrás (0 = első) |
| `models` | a modellek útvonalai (4. szakasz) |
| `tuning`, `gate` | a klaszterezés küszöbei és a QA-kapuk (a mért alapértékek) |
| `llm` | szövegellenőrző pont a szintézis előtt (opcionális) |
| `asr` | a kész hangsáv visszaellenőrzése (`parakeet`, `whisper`, `hybrid`) |
| `intake` | feliratfordítás és hangsáv-szétválasztás beállításai |
| `diarization` | pyannote szegmentálás: küszöb, minimális forduló, `num_speakers` (0 = automatikus) |
| `series` | sorozatfigyelő (`auto_process`, `poll_s`) |
| `media_server` | NAS-mappa: innen másol a projekt, ide másolja vissza a kész szinkront |

A projekt `data\` mappájába kerül a média (`S01E05 - név.mkv` sorozatnál, a film
egyetlen fájlja filmnél); a `subs\` mappába az `<EP>_eng.srt` / `<EP>_heb.srt`
forrás és a `<EP>_hun.srt` kész magyar felirat.

---

## 6. Fordítás beállítása

Az intake a feliratot OpenAI-kompatibilis végponttal fordítja. Alapértelmezés a
helyi Ollama:

```json
"intake": { "translate": {
  "enabled": true, "base": "http://127.0.0.1:11434/v1",
  "model": "qwen3.5:9b", "batch": 20, "timeout": 300
}}
```

- **Helyi szerver:** Ollama (`http://127.0.0.1:11434/v1`) vagy LM Studio
  (`http://127.0.0.1:1234/v1`), API-kulcs nélkül. A modell neve a szerveren
  futó modell; a kimenet JSON, ezért érdemes nem „gondolkodó” modellt választani.
- **Külső szolgáltatás nincs:** a fordító csak `127.0.0.1`/`localhost`
  végpontot fogad el, a távoli URL-t hibával elutasítja — a feliratszöveg nem
  hagyja el a gépet, és a projektfájlba nem kerül API-kulcs.

Ha a magyar felirat már kész (`subs\<EP>_hun.srt`), a fordítás kimarad; a
felületre feltöltött magyar SRT-t az intake fordítás nélkül használja.

---

## 7. Első futás — ellenőrzés

```bat
:: 1. a hangok és a modellek elérhetősége (a dubstudio saját ellenőrzése)
cd dubstudio
..\reader\.venv\Scripts\python.exe -m dubstudio voices
cd ..

:: 2. a környezet és a projektek
reader\.venv\Scripts\python.exe scripts\check_dubstudio.py
```

Ezután indítsd az Aurist (`reader\run.bat`), és nyisd meg a
<http://127.0.0.1:7860/szinkron> oldalt. A sorrend:

1. **Új epizód/film** — médiafájl vagy mappaválasztó; a projekt `data\` mappájába
   kerül (hardlink, ha lehet).
2. **Hiányzó anyag** — felirat (fordítás) és vokál/M&E leválasztás (`intake`).
3. **Előkészítés és klaszterezés** — beszélőfordulók, szereplők, hangzár.
4. **Szinkron** — terv → szintézis → keverés → mux (`work\out\*.mkv`), QA-kapukkal.
5. **Ellenőrzés** (`/felulvizsgalat`) — soronkénti emberi döntés; a következő
   futás ezt használja.

A kapuk a projekt `gate` szakaszában állíthatók; a futás a mux után méri őket,
ezért bukó kapu esetén is elkészülnek a fájlok, de a folyamat hibakóddal áll meg.

---

## 8. Hibaelhárítás

| Tünet | Ok | Megoldás |
|---|---|---|
| a `/szinkron` oldal: „Még nincs projekt” | nincs `projects\*.json`, vagy rossz a `DUBSTUDIO_ROOT` | hozz létre egyet az „Új film” űrlappal, vagy `check_dubstudio.py --init "Név"`; rossz `DUBSTUDIO_ROOT` esetén állítsd be, majd indítsd újra az Aurist |
| `ModuleNotFoundError: sherpa_onnx` / `torch` / `audio_separator` | a csomagok nincsenek az Auris venvjében | `reader\.venv\Scripts\python.exe -m pip install -r reader\requirements.txt`, majd újraindítás |
| a futás „hiányzó modell” hibát ad | a modellek nincsenek letöltve | `... scripts\fetch_models.py`, majd `--check`; a projekt `models` útvonalai legyenek a letöltött helyek |
| az intake nem talál feliratot | nincs szöveges felirat a fájlban | tölts fel SRT/VTT-t a felületen (a képfelirat nem fordítható) |
| nincs hang a szinkronban | a `models` útvonalai rosszak | `check_dubstudio.py` megmondja, melyik fájl hiányzik |
| a Piper-hangok nem szólalnak meg | nincs `espeak-ng`, vagy a `models.piper` mappa üres | 3–4. szakasz |
| `ffmpeg` hiba a muxnál | nincs `ffmpeg`/`ffprobe` a PATH-on | 3. szakasz |
| a fordítás üres/hibás JSON | a modell „gondolkodó”, vagy kicsi a `max_tokens` | válassz nem-reasoning modellt, növeld a `max_tokens`-t |
| a QA `one_voice_per_utt` kapuja bukik | egy mondaton belül két hang szól (kézi szóhatár-vágás) | a `/felulvizsgalat` oldalon ellenőrizd a sort, vagy lazítsd a `gate` értékét |

---

## 9. Amit az Auris ad hozzá

- `/szinkron` — projekt- és epizódkönyvtár, intake/futás indítása, beállítások,
  beépített lejátszó (forrás vagy magyar hangsáv, feliratok).
- `/felulvizsgalat` — film és felirat egymás mellett, soronkénti beszélő-döntés
  magyar keresztnevekkel; a döntés a `work\<EP>\review\lines.json`-ba íródik.
- Médiaszerver-visszamásolás — a kész MKV az eredeti fájl helyére (első cseréről
  `.bak` készül).
- Automatikus futás — a `series.auto_process` bekapcsolásával az új epizódok
  maguktól elindulnak, ha minden anyag megvan.

A szinkronlánc részletei (klaszterezés, hangzár, kapuk, riport) a dubstudio
`README.md`-jében vannak.
