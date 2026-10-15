# Teljes könyv-feldolgozás kézileg — generálás → hibrid ellenőrzés → M4B

> Snapdragon X Elite (ARM64), Windows 11. Minden lépés a **Web UI**-on fut, a
> végén egy `.m4b` fájl jön ki, amit bármelyik könyvolvasó (Apple Books,
> Smart AudioBook Player, Listen Audiobook Player) lejátszik.
>
> Az útmutató a `reader/core` kódjából származik: minden útvonal és
> beállításnév valóban így hívódik.

---

## 0. Előkészítés (egyszer)

**1. Indítsd el az appot:**

```bat
cd /d "D:\VoiceAI\apps\auris"
reader\run.bat
```

A böngésző a **http://127.0.0.1:7860**-on nyílik. A jobb felső sarokban
látszik a beszédmotor állapota: **BESZÉDMOTOR KÉSZ** kell, hogy legyen (a
modell hátérben töltődik, várj rá).

**2. Ellenőrizd, hogy minden motor a helyén van** — a Beállítások
(`/settings`) → Beszédmotorok fül:

| Beállítás | Érték | Miért |
|---|---|---|
| `tts_engine` | `auto` | hardver-javaslat; Snapdragonon a Supertonic 3-at adja |
| `supertonic_batch` | `4` | a leggyorsabb mért érték (batch-4) |
| `supertonic_variant` | `int8` | 138 MB, ~400 perc hang/óra — a leggyorsabb |
| `qa_autostart` | **bekapcsolva** | az Ellenőrzés-oldal megnyitásakor magától lefut |

> **Gyorsítás:** ha a leggyorsabb motor kell, válaszd a `supertonic_sherpa`
> motort (Settings → Beszédmotorok). Ez a sherpa-onnx int8 kiépítés, mért
> **RTF 0,14** — a szokásos Supertonicnál ~2× gyorsabb. Klónozni ez sem tud
> (lásd a 3. lépést).

**3. FFmpeg kell az M4B-hez** (már fent van ezen a gépen:
`C:\ffmpeg-arm64`). Ha máshol van, add hozzá a PATH-hoz. Nélküle a
hanggenerálás és a WAV/MP3 export megy, de az **M4B nem**.

---

## 1. Könyv importálása

**A felületen:** Könyvtár (`/`) → **„Könyv hozzáadása"** → EPUB / PDF / DOCX /
TXT / FB2 kiválasztása. Import után a könyv fejezetre bontva jelenik meg, és
a szereplők (párbeszédek hangszerei) automatikusan kikerülnek.

**Parancssorról** (ugyanazt csinálja):

```bat
reader\.venv\Scripts\python.exe reader\auris_cli.py import "konyv.epub" --wait
```

A `--wait` a szereplőelemzésre vár. Ha a könyv nyelve nem magyar, állítsd be a
Könyv adatainál (`/reader/<id>` → Adatok) — **a QA nyelvi választása ebből
származik**, és a beszédfelismerő ehhez igazodik.

---

## 2. Hangok beállítása (Hangstúdió)

**`/voice-studio/<id>`**

- **Egyszerű (ajánlott ezen a gépen):** a motor beépített hangokkal dolgozik.
  A NARRÁTOR blokk → „Hang finomhangolása" → **Beépített hang**: F1–F5 (női),
  M1–M5 (férfi), vagy „Automatikus" (a leírt nem alapján választ).
  Jelöld be **„Ez a narrátor olvassa fel a teljes könyvet"**, ha egy hangon
  akarod az egészet. **Mentés.**

- **Klónozás (csak klónozó motorral):** a referenciahang-feltöltés csak akkor
  jelenik meg, ha a motor támogatja a klónozást. A Supertonic **nem** — ezért
  ott a feltöltés rejtve van, és egy figyelmeztetés jelzi. Klónhoz olyan motor
  kell (pl. XTTSv2-ONNX), amelyik `voice_clone: true`.
  Klónozó motorral a feltöltésnél a **„A feltöltésből a legjobb klónozási
  szakaszt válaszd ki"** jelölőnégyzet alapból be van kapcsolva: a program a
  feltöltött hangból a legtisztább, folyamatosan beszélt szakaszt választja ki
  (nem a köszönést és nem a szobazajt), 24 kHz mono WAV-ban menti, és
  visszajelzi a pontszámot és hogy miért ez lett a kiválasztott.

---

## 3. Hang generálása

**A legegyszerűbb: Produkció oldal (`/production/<id>`)** — ez a
„mennyire kész ez a könyv" vezérlőpont.

- **„Teljes könyv generálása"** gomb → az összes fejezetet legyártja sorban.
- Vagy fejezetenként: az Olvasón (`/reader/<id>`) az egyes fejezeteknél.

**Parancssorról / API-ról** (ugyanaz):

```
POST /api/books/<id>/generate
Body: { }                      <- az összes fejezet
Body: { "chapter_ids": [1,2] } <- csak a választottak
```

Visszaad egy `job_id`-t. A haladás a **Feladatok** oldalon (`/jobs`) vagy
`GET /api/jobs` alatt követhető. A motor `not_loaded` állapotból magától
betölt indulás előtt.

> **Mennyi ideig tart?** Supertonic int8-kal ~**9,6 perc hanganyag / óra
> könyv**; a `supertonic_sherpa` motorral (~400 perc hang/óra) egy 8 órás
> könyv ~15 perc alatt készül el. A generálás alatt a böngésző nyitva maradhat,
> a feladat a szerveren fut.

---

## 4. Minőségellenőrzés (hibrid, alapból automatikus)

**`/qa/<id>`** — ez az Ellenőrzés oldal.

**Amit be kell állítanod (alapból jó, csak ellenőrizd):**

| Mező | Érték | Megjegyzés |
|---|---|---|
| **Beszédfelismerő mód** | **hibrid** (alap) | Parakeet hallgat végig, a gyanús mondatokat Whisper újrahallgatja |
| **Beszédfelismerés** | bekapcsolva | nélküle nincs visszahallgatás |
| **Hibás mondatok automatikus újragenerálása** | bekapcsolva | a jelölt mondatokat maga javítja |

A négy választható mód:

1. **Automatikus** — videokártyán Whisper, processzoron Parakeet + Whisper
2. **Parakeet + Whisper-megerősítés (hibrid)** ← *alapértelmezett*
3. **Csak Parakeet** — a leggyorsabb processzoron
4. **Csak Whisper** — magyarra finomhangolt (`sarpba/whisper-hu-large-v3-turbo-finetuned`)

A választó alatt látszik, **mi futna tényleg** ezen a gépen, és melyik opció
érhető el. A mód váltása ugyanazt a `asr_backend` beállítást írja, amit a
Beállításokban is látsz.

**Automatikus futás:** ha a `qa_autostart` be van kapcsolva (0. lépés), az
Ellenőrzés-oldal megnyitásakor **magától elindul az első ellenőrzés** a
kiválasztott fejezeten — nem kell a gombot megnyomnod.

**Kézileg:** **„Fejezet ellenőrzése"** gomb. A program:

1. visszahallgatja a fejezet hangját beszédfelismeréssel,
2. összeveti a szöveggel (CER/WER),
3. megjelöli a hibás, levágott, túl gyors vagy túl halk mondatokat,
4. az újragenerálás bekapcsolva pedig elkészíti a javított változatokat.

Az összesítésben látod: **Rendben / Gyanús / Hibás / Nincs ellenőrizve /
Jóváhagyva**. A „Hibás és gyanús" szűrővel csak a teendők látszanak. Az
egyes mondatoknál:

- **„Új változat meghallgatása"** → **„Jóváhagyom"** ha tetszik (ez menti a
  jó változatot), vagy
- **„Új változat hangja"** → másik szereplőhanggal is újrageneráltathatod.

> **Mire figyelj:** a CER küszöbök (jelentés: `qa_cer_warn` 0,08, `qa_cer_fail`
> 0,15) fölött a mondat gyanus/hibás. A hosszú szünetek (`long_pause`) és a
> túl halk részek (`loudness`) külön jelzést kapnak. A **„Hangerő-jelentés"**
> gomb a teljes fejezet hangerejét adja (RMS, peak).

**Ha egy mondatnál a felismerő tévedett** (a hang jó, csak a szövegfelismerés
nézte el): a mondatnál hagyd jóvá — a jóváhagyás felülírja a felismerő
ítéletét.

---

## 5. M4B export

**A felületen:** Könyvtár → a könyv kártyája → **„Produkció"**
(`/production/<id>`) → export formátum: **M4B**.

**API-ról** (ugyanazt csinálja):

```
POST /api/books/<id>/export/full
Body: {
  "audio_fmt": "m4b",          <- wav | mp3 | m4b | opus | flac
  "sub_fmt":   "none",         <- srt | ass | none (felirat a fájlba)
  "package":   "none",         <- none | epub3 | audiobookshelf | acx | daw
  "intro": false,              <- „Felolassa: ..." nyitó
  "outro": false               <- „Vége." záró
}
```

Visszaad egy `job_id`-t. Követés és letöltés:

```
GET  /api/export/status/<job_id>       <- állapot, progress
GET  /api/export/download?job_id=...   <- a kész fájl letöltése
```

**Amit az M4B csinál:** a fejezetek hangját egyetlen `.m4b` fájlba rakja,
fejezetjelzőkkel (chapter marks) és könyv-metaadatokkal (cím, szerző), FFmpeg
segítségével. Opcionálisan SRT/ASS feliratot és nyitó/záró kreditet is tesz
be.

> **Kötelező:** FFmpeg **és** ffprobe a PATH-ban. Ha nincs, az M4B hibát ad
> („Az M4B-exporthoz FFmpeg és ffprobe szükséges"), de a többi formátum megy.
> Ezen a gépen:`C:\ffmpeg-arm64` — bent van a PATH-ban.

**Kiadási csomagok** (`package`): ha Audiobookshelfre töltöd fel, add meg a
szerver címét, tokenjét és könyvtárát a Beállításokban, és jelöld be
`abs_upload`. Az ACX csomag automatikusan mintát is kér.

---

## 6. Teljes, egyben futó sorrend (checklist)

```
[ ] 1. run.bat → app indul, BESZÉDMOTOR KÉSZ
[ ] 2. Settings: tts_engine=auto, supertonic_variant=int8, qa_autostart=BE
[ ] 3. Könyv import (Könyvtár → Könyv hozzáadása)
[ ] 4. Nyelv beállítva a könyvnél (ha nem magyar)
[ ] 5. Hangstúdió: narrátor hang + „teljes könyv" jelölő + Mentés
[ ] 6. Produkció: Teljes könyv generálása → kivárni (Feladatok oldal)
[ ] 7. Ellenőrzés (/qa/<id>): mód=hibrid, autostart lefut, vagy „Fejezet ellenőrzése"
[ ] 8. Hibás/gyanús mondatok jóváhagyása („Jóváhagyom")
[ ] 9. Produkció: export = M4B → letöltés
```

---

## GYIK

**„A beszédmotor még nem áll készen" (503 az exportnál).** A modell még
töltődik. Várd meg a BESZÉDMOTOR KÉSZ feliratot a jobb felső sarokban.

**„Egy fejezethang már készül" (409).** Egyszerre egy generálás/futás mehet.
Várd meg, vagy a Feladatok oldalon mondd le.

**Az M4B hibát ad FFmpeg-ről.** Hiányzik a PATH-ról. Windows: Rendszerváltozók
→ PATH → hozzáadás `C:\ffmpeg-arm64\bin`.

**A hangklónozós feltöltés nem látszik a Hangstúdióban.** A motor nem támogatja
a klónozást (pl. Supertonic). Klónozáshoz `voice_clone: true` motor kell; a
Supertonic beépített hangokkal dolgozik.

**A szótár (kiejtési javaslatok) elsőre lassú.** Az első használatkor tölt le
egy magyar szótárat (~1,3 MB) a nyilvános tárházról, és a `reader/lexicon/`
pufferbe teszi. Offline gépen a javaslatok szűkebbek maradnak, de a többi
funkció hibátlanul megy.

**2 teszt ERROR a lefuttatáskor.** Az ismert ARM64 környezeti hiány
(`torchaudio` / `wetext` nincs telepítve) — nem hiba, a funkcionalitást nem
érinti.
