# dubstudio — magyar szinkronkészítő alkalmazás

Filmek és sorozatok magyar szinkronját készíti el a nulláról: kideríti, **ki,
mikor, mit mond**, megállapítja a beszélő nemét, évad-szinten **egy**
magyar hangot rendel minden szereplőhöz, megszintetizálja a dialógust a
zene+effekt sávra, és **mérhető kapukkal** ellenőrzi az eredményt.

```
                  ┌────────────┐
 feliratok ──────►│  prepare   │──► megnyilatkozások + hanglenyomat + nem
 énekbeszéd-sáv ─►└────────────┘    work/<EP>/utterances.json
                          │
                          ▼
                  ┌────────────┐    évad-szintű klaszterezés,
                  │  cluster   │──► szereplő-rekord + hangzár
                  └────────────┘    work/cast-<projekt>.json
                          │
                          ▼
                  ┌────────────┐    terv → szintézis (fix paraméterekkel)
                  │    dub     │──► → keverés → mux → QA riport
                  └────────────┘    work/<EP>/report.html + work/out/*.mkv
```

## Miért épült újra

A korábbi lánc a beszélőt **diarizációs szegmensek átfedéséből** tippelte, és
a hangot **epizódonként, klaszterenként** osztotta. A mérés (egy 10 részes
sorozat egy epizódja) ezt igazolta:

| hiba | mérés a régi láncon |
|---|---|
| ugyanaz a színész más magyar hangon szólal meg | **71 / 99** szomszédos, akusztikusan azonos sorpár (72%) |
| ezen belül nemet is vált | több cos≈1,00 párnál M↔F váltás |
| epizódváltásnál új hang | a hangkiosztás epizódonként történt |
| a hang karaktere soronként ingadozik | a TTS-sebesség soronként 0,7–1,7× között változott |

Az új alkalmazás ezért négy elven működik:

1. **Előbb a forduló, aztán a hang.** A beszélőváltásokat a teljes
   énekbeszéd-sávon pyannote 3.0 szegmentáló + CAM++ hanglenyomat keresi. A
   feliratsor nem saját maga klasztereződik: ahhoz a **beszélőfordulóhoz**
   kerül, amelyik alatt elhangzik. Ez adja a mondatról mondatra váltás elleni
   garanciát.
2. **A hang ellenőrzi az azonosságot.** A fordulók kezdeti címkéjét a CAM++
   embedding és a nem-bizonyíték validálja; csak azonos nemű, kellően hasonló
   fordulók olvadnak össze, és a kevert nemű forduló szétválik. A nyers
   diarizációs címke tehát nem végleges szereplő, csak kiindulópont.
3. **A megnyilatkozás a hang egysége.** A sorokat fordulónként, a szöveg
   folytonossága mentén csoportosítjuk, és a hangot a *megnyilatkozáshoz*
   kötjük — így egy mondaton belül **szerkezetileg** nem szólalhat meg két hang.
4. **A hang zárolt.** A klasztert egy szereplő-rekordhoz kötjük, a rekord
   hangja és hangszíne állandó; a szintézis nem választhat hangot.
5. **Ami nincs mérve, az nincs kész.** Minden lépés mérőszámot ad, és a
   QA-kapu bukása megállítja a folyamatot.

### Zene, dalszöveg és háttérhang

A felirat zenei sorai (`♪ Ez az álom... ♪`, `[Music]`, `(zene)`) nem
dialógusok. A `prepare` `music=True` jelöléssel **megtartja** őket a
sorlistában (ez tartja stabilan a megnyilatkozás-azonosítókat és a
felülvizsgálat kulcsait), de nem kapnak szereplőt, hangot és beszédidőt, a
felülvizsgálat és a terv kihagyja őket, és nem mennek a TTS-be. Mért eset
(E04): 2 dalszöveg-sor; a beszédidő így C001 182,2 s és C009 3,2 s (a
kizárt dalszöveggel 185,0 / 4,7 s volt).

## Hogyan működik a beszélőazonosítás

A `prepare` a teljes énekbeszéd-sávon futtatja a diarizációt, majd a
feliratsorokat a **legnagyobb időbeli átfedés** alapján rendeli a fordulókhoz.
A rövid, zenés vagy halk sorokat legfeljebb `diarization.attach_gap` (alap 2 s)
távolságig csatolja a legközelebbi fordulóhoz; ennél távolabb nem tippel.

A `cluster` ezután öt lépésben stabilizálja a szereplőket:

1. **Kevert nemű forduló szétvágása** — ha ugyanabban a diarizációs címkében
   legalább 3 s erős női és 3 s erős férfi bizonyíték van, a címkét kettévágja.
   Ez volt a fő hibaforrás a párbeszédes jelenetekben.
2. **Önbemutatás összekötése** — ugyanaz a hosszú „A nevem …” szöveg két
   címkében ugyanazt a beszélőt jelenti (visszajátszás, ismételt felvétel).
3. **Feltételes összevonás** — csak azonos nemű címkék olvadhatnak össze;
   erős hanghasonlóság (koszinusz ≥ 0,80) vagy különböző jelenet + közepes
   hasonlóság (≥ 0,55) kell hozzá. Két, egymással párbeszédben álló szereplő
   soha nem olvad össze pusztán azért, mert hasonló a hangja.
4. **Töredék-csatolás** — a 2,5 s alatti, magányos címkék a legközelebbi
   bizonyított, nem co-occuráló azonos nemű szereplőhöz kerülnek.
5. **Nem-ellentmondás javítása** — ha egy sor erős női/férfi jele ellentétes a
   klaszter nemével, a sor a legjobb azonos nemű klaszterbe kerül.

A QA `voice_switch` kapuja nem csak a szomszédos sorok embeddingjét nézi: a
klaszterek középpontját is összeveti. Két valóban különböző, azonos nemű
szereplő (pl. két női szereplő párbeszéde) így nem jelent hamis hibát, viszont
ugyanazon színész két különböző magyar hangja igen.

## Követelmények

* Python 3.12 + a projekt virtuális környezete (a `sherpa-onnx`,
  `onnxruntime`, `numpy`, `scipy`, `soundfile`, `transformers` csomagokkal) —
  a befogadó Auris repóban: `../reader/.venv`.
* `ffmpeg` és `ffprobe` a PATH-ban.
* Modellek (a `projects/*.json` `models` szakasza adja meg):
  * **pyannote 3.0** szegmentáló
    (`models/diar/sherpa-onnx-pyannote-segmentation-3-0/model.onnx`),
  * **CAM++** beszélő-embedding (`models/diar/campplus.onnx`, Apache-2.0),
  * **Supertonic 3** (sherpa-onnx int8) — 5 női + 5 férfi hang,
  * **Piper hu_HU** (`anna`, `berta`, `imre`) + `espeak-ng` az IPA fonémázáshoz.
* Nem-osztályozó: `alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech`
  (HuggingFace cache-ből; első futásnál letölti).

## Hangok (13)

| motor | hangok | nem | fix paraméterek |
|---|---|---|---|
| Supertonic 3 | `SUP_F1`…`SUP_F5`, `SUP_M1`…`SUP_M5` | 7 nő / 6 férfi összesen | `sid`, `speed`, `num_steps`, `silence_scale` |
| Piper hu_HU | `PIP_anna`, `PIP_berta` (nő), `PIP_imre` (férfi) | | `length_scale`, `noise_scale`, `noise_w` |

Ha két szereplő mégis ugyanazt a hangot kapja (elfogytak a hangok), **fix
hangszín-eltolással** (`pitch`, félhang, `rubberband`) különböztetjük meg —
ez is szereplőhöz kötött, tehát állandó.

### Gyakoriság = szerep

A beszédidő (a klaszterek ideje) a **színészi hang gyakorisága**; ebből a
`cluster` lépés rangsort, arányt és szerepet rendel minden szereplőhöz
(`rank`, `share`, `role`, `exclusive` a `cast-*.json`-ban, lásd `usage`):

* **főszereplő** — az összes beszéd legalább `tuning.main_share_min`
  (alap 4%) részéért felelős; neki **kizárólagos** hang jár, amit más
  szereplő nem használhat (`main_shared` kapu = 0);
* **mellékszereplő** — osztozhat a hangokon; előbb a nem főszereplői hangokat
  keresi, és mindegyikük fix `pitch`-el különbözik.

### Évadszintű újrakiosztás (`recast`)

A `cluster` csak az új szereplőket osztja ki, a meglévő hangokat megtartja —
így egy epizód újrafuttatása után a főszereplő hangján maradhattak
mellékszereplők. A `recast` ezért **az egész évad** szereplőit újraosztja:

1. a beszédidőt **epizódonként** összegzi (a `duration` nem íródik felül az
   utolsó epizód idejével);
2. a főszereplők (`share >= tuning.main_share_min`) először **kizárólagos**
   hangot kapnak, nemenként egyedileg;
3. a mellék-, epizód- és statisztaszereplők a maradék hangokon osztoznak,
   DSATUR-szerű színezéssel: a visszatérő mellékszereplők választanak előbb,
   majd a legtöbb jelenetkapcsolattal rendelkező szereplő, kerülve a már egy
   jelenetben használt hangot;
4. az azonos hangot osztozó szereplők fix, eltérő `pitch`-et kapnak; a
   szigorú (pitch nélküli) ütközést a riport külön mutatja.

```bash
../reader/.venv/Scripts/python.exe -m dubstudio recast E01 E02 ... E11
```

A parancs a hangot visszaírja az epizód-JSON-okba is, ezért utána a `dub`
már az új kiosztással dolgozik. Teljes évadnál előbb mind a 11 epizódot
`prepare`-ölni kell; az E05/E06 kétbeszélős sorai miatt ezt a két epizódot a
jelenlegi fordulóvágó kóddal érdemes újrafuttatni.

### Szereplői rétegek

A kiosztás négy réteget különböztet meg (a `recast` és a riport is ezt használja):

| réteg | feltétel | kiosztás |
|---|---|---|
| **fő** | évadidő ≥ `main_share_min` (alap 4%) | nemenként egyedi, kizárólagos hang |
| **mellék** | legalább 2 epizód és ≥ 1% évadidő | a maradék hangokon osztozik; előbb választ |
| **epizód** | legalább 10 s, de nem visszatérő | a maradék hangokon osztozik |
| **statiszta** | 10 s alatt | a maradék hangokon osztozik, pitch-csel |

Ez követi az iparági gyakorlatot: a fő- és visszatérő szerepeket előbb,
lehetőleg állandó hanggal kell kiosztani; az epizód- és statisztaszerepek
osztozhatnak, de egy jelenetben nem szólalhat meg két azonos hang. A
jelenetütközést a `recast` gráfszínezéssel és lokális javítással csökkenti;
a maradék azonos hangot a fix pitch különbözteti meg. A részletes indoklás a
`docs/session-zaro-2026-11-04-szinkron-e07.md` 15. szakaszában van.

## Használat

```bash
cd dubstudio
PY=../reader/.venv/Scripts/python.exe

$PY -m dubstudio voices                        # hangkészlet és elérhetőség
$PY -m dubstudio usage                         # hanggyakoriság: ki milyen gyakran
                                               # beszél, kinek melyik hang
$PY -m dubstudio prepare E05                   # feliratok + hang-identitás + nem
$PY -m dubstudio intake E05                    # nyers MKV: magyar felirat + vokál/M&E
$PY -m dubstudio intake E05 --then-run         # intake után azonnal szinkron is
$PY -m dubstudio cluster E01 E02 ... E11       # évad-szintű szereplők + hangzár
$PY -m dubstudio dub E05                       # terv → szintézis → keverés → mux
$PY -m dubstudio run E05 E06 --season          # a három lépés egyben
$PY -m dubstudio youtube E01 --url https://youtu.be/... --then-run
                                               # letöltés + felirat + szinkron
```

A `youtube` lépés a nyilvános YouTube URL-ről legfeljebb 1080p-s H.264 videót
és angol (kézi vagy automatikus) feliratot tölt le, majd a meglévő
`intake` + `run` láncot futtatja. A feliratot a modul a videóhoz tartozó
összes jelölt-címből, több körben kéri le: a YouTube átmeneti `HTTP 429`
(rate limit) válasza így nem viszi el a teljes letöltést. A média letöltése
formátum-tartalékláncon fut (1080p H.264 → 720p → 360p), és ha van a gépen
`node` vagy `deno`, a yt-dlp JS-futtatókörnyezetként használja — enélkül a
1080p60-as H.264 gyakran `403`-at ad. Felirat nélküli videóhoz előbb külön
SRT-t kell a projekt `subs/E01_eng.srt` helyére tenni. A WebUI ugyanezt a
`/api/dub/youtube` végponton és a „YouTube videó” panelen indítja.

Hasznos kapcsolók: `--limit N` (csak az első N sor, gyors próba), `--no-mux`,
`--out NAME`, `--allow-fail` (bukó kapu esetén is folytatja), `-q`.

### Sorozat vagy film

A projekt JSON `kind` mezője dönti el a módot:

* `"kind": "series"` (alapértelmezett) — a `data_dir` epizódjai `SxxExx`
  szerint; az évad-szintű cast és hangkiosztás öröklődik az új részekre,
  a Web UI könyvtárnézete a hiányzó feliratot/vokált/M&E-t mutatja.
* `"kind": "movie"` — egyetlen `E01` egység; a `data_dir` első támogatott
  videója a film, nincs évad-szintű újrafeldolgozás.

A `series` szakasz opcionális: `{"auto_process": true, "poll_s": 30}`.
Automatikus indítás csak akkor történik, ha a média, a magyar felirat, a
leválasztott énekbeszéd és az M&E is megvan; a kész kimenet megléte lezárja
a részt (QA-bukás után nem indul végtelen újrapróbálkozás).

### Nyers epizód intake (új rész MKV-ból)

Az `intake` a nyers médiafájlból készíti el a szinkronhoz szükséges két
hiányzó anyagot:

1. **magyar felirat** — külső `subs/<EP>_<nyelv>.srt`, majd beágyazott
   szöveges felirat (SubRip/ASS/mov_text/WebVTT); a képfelirat (PGS/VobSub)
   nem fordítható, ilyenkor külső SRT kell. A fordító OpenAI-kompatibilis
   helyi LLM (alapból Ollama `http://127.0.0.1:11434/v1`, `qwen3.5:9b`);
   az időzítés változatlan, a kimenet `subs/<EP>_hun.srt`.
2. **forráshangsáv** — az `original_audio` (alapból 0) sáv kinyerése,
   csúcsnormalizálás, majd `audio-separator` UVR/MDX szétválasztás. A
   `work/<EP>/vocals.wav` a vokál, a `work/<EP>/me.wav` a zene+effekt; a
   második kör a maradék beszédet vonja ki az M&E-ből.

A folyamat állapota `work/<EP>/intake.json`; hiba esetén a figyelő
`retry_s` (alap 900 s) után próbálja újra. A `--no-translate`, `--no-separate`,
`--force`, `--then-run` és `--allow-fail` kapcsolók a CLI-n és a Web UI-on is
elérhetők. Windows ARM64-en a `torchvision` kerék hiányát az intake beépített
minimális shimje hidálja át az `onnx2torch` importjához.

Kimenetek:

* `work/<EP>/report.html` és `report.json` — mérési riport,
* `work/<EP>/dub.wav` — a kész magyar hangsáv,
* `work/out/<...> - HUN DUB.mkv` — videó + magyar szinkron + eredeti hangok + feliratok,
* `work/cast-<projekt>.json` — a szereplő-rekordok és a zárolt hangok.

## QA kapuk

| kapu | mit jelent | alapérték |
|---|---|---|
| `voice_switch` | akusztikusan azonos szomszédos megszólalás más hangon szól | 0 db |
| `one_voice_per_utt` | egy mondaton belül egy hang szól | 0 db |
| `gender_conflict` | erősen mért nemmel ellentétes nemű hang a soron | 0 db |
| `coverage` | minden megszólaláshoz tartozik szereplő és hang | 100% |
| `scene_conflict` | egy jelenetben két szereplő ugyanazt a hangot használja | 0 db |
| `main_shared` | főszereplő, akinek a hangját más szereplő is használja | 0 db |
| `fit` | a szintetizált sor hossza illeszkedik az ablakhoz | ≤2%, ≤0,6 s |

A küszöbök a projekt JSON `gate` szakaszában állíthatók.

## Kézi sor-javítás és a szinkron

A **`/felulvizsgalat`** oldal (Auris „Ellenőrzés”) döntései nem maradnak
jegyzetben: a következő `dub` futás felhasználja őket.

* **Teljes sorra tett szereplő-megjelölés** (`review/lines.json`
  `lines.<uid>#<sor>.marks`, `words: null`) felülírja a sor hangját — a
  klaszterezés döntése helyett a megjelölt szereplő (és hangja, fix
  hangszíne) szól. Ezt a `linereview.labels` adja át a `plan.build`-nek.
* **Szókijelölésre tett mark** kettévágja a sort a szavak arányában: a
  kijelölt rész a megjelölt szereplő hangján szól, a többi marad a gép
  szerinti szereplőnél.
* **Elcsúszás** (`offset`): a sor a megadott másodperccel eltolva kerül a
  hangsávra (a felirat időzítése nem változik).
* A „képen kívül” és „nem tudom” jegyzet; hangot nem rendel.

A kézi döntés **szándékos**, ezért az `one_voice_per_utt` kapu nem számolja
hibának, ha egy megnyilatkozáson belül emiatt szólal meg két hang. A riport
`manual_lines` mezője mutatja, hány sor szól kézi döntés alapján.

### Kevesebb kényszerű összenyomás

A szintézis a magyar sort eddig a feliratsor ablakára nyomta (`atempo`). Ha a
magyar hang hosszabb az ablaknál, most a **következő sor előtti valódi
csendből** kap legfeljebb `BORROW_MAX` = 0,6 s ráhagyást (a következő sor
kezdetéig, 20 ms biztonsági réssel). Rövidebb sort nem lassítunk: a ráhagyás
csak az összenyomást csökkenti, és a gyorsítótár a tényleges hosszt is
tárolja, így a következő futás ugyanazt az ablakot állítja vissza.

## Felépítés

| modul | feladat |
|---|---|
| `config` | projekt/útvonalak, hangmodellek |
| `subs` | SRT beolvasás, szövegtisztítás, magyar TTS-előkészítés |
| `youtube` | YouTube videó + angol felirat letöltése a projekthez |
| `audio` | WAV I/O, resample, `atempo` időillesztés, `rubberband` hangszín |
| `media` | ffprobe/ffmpeg, MKV mux |
| `embeddings` | CAM++ beszélő-embedding |
| `gender` | tanított nem-osztályozó |
| `utterances` | sorok → akusztikusan összefüggő megnyilatkozások |
| `clustering` | évad-szintű klaszterezés + Viterbi-finomítás |
| `registry` | szereplő-rekord, hangzár és hanggyakoriság (fő/mellék) |
| `voices` | a 13 hang és a motorjaik (Supertonic, Piper) |
| `plan` | időzítési terv (a hang itt már nem változhat) |
| `synth` | szintézis fix paraméterekkel + gyorsítótár |
| `mix` | keverés ducking-gal |
| `qa` | kapuk és mérések |
| `report` | HTML riport |

## Tesztek

```bash
cd dubstudio
../reader/.venv/Scripts/python.exe -m unittest discover -s tests -v
```

## Web UI az Aurisban — Szinkron stúdió

A befogadó **Auris** alkalmazás (`../reader/`) `/szinkron` oldala ugyanennek a
pipeline-nak a
felülvizsgáló- és kezelőfelülete. **Nem másolatot használ:** ugyanazt a
`projects/*.json`-t, a `work/<EP>/review.json`-t és ugyanazt a
`python -m dubstudio …` parancsot kezeli, mint a parancssor.

### Mit lehet a Web UI-ban (a CLI mellett)

| Mi | Hogy | Kód |
|---|---|---|
| Beszélő-azonosítás emberi döntése (elfogadás/javítás/elutasítás, név, hang) | a Beszélők szekció kártyáin | `work/<EP>/review.json` |
| Minden beállítás szerkesztése (modellek útvonala, klaszterezés, QA-kapuk, szöveg-ellenőrző LLM, ASR) | a Beállítások szekcióban szakaszonként | `projects/*.json` — mentés előtt **mindig** készül biztonsági mentés a `projects/_backup/`-ba |
| Pipeline-lépések indítása (intake / prepare / cluster / dub / run / asr) kapcsolókkal | a Futtatás szekcióban | háttér-folyamat, log: `work/webui_<idő>_<lépés>_*.log` |
| Futtatás állapota és a napló élő nyoma | a Futtatás szekcióban (3 mp-ként frissül) | `GET /api/dub/run` |
| Futtatás leállítása | Leállítás gomb | `POST /api/dub/run/stop` |
| Hangkatalógus (13 hang, motor, nem, paraméterek) és modell-mappa jelenlét | a Beállítások oldal alján lévő kártyá-sor | ** nem tölt modellt** |
| Próba-futtatás (audit) és hangszín-mérés | a felső eszközsában | `POST /api/dub/audit`, `POST /api/dub/timbres` |
| Sorozat/film könyvtár, új epizód hozzáadása, hiányzó anyagok | a Könyvtár szekcióban | `work/<EP>/`, `subs/<EP>_hun.srt`, `data/` |
| Beépített lejátszó forrás/magyar hangsávval és felirattal | a Lejátszó szekcióban | `work/<EP>/review/player_*.mp4`, `sub_*.vtt` |
| Proxy-előkészítés állapota, gyors hangsávváltás, magyar minta | a Lejátszó szekcióban | `GET /api/dub/player/status`, `POST /api/dub/player/prepare` |
| Nyers epizód anyagainak pótlása (felirat + hang) | a Könyvtár „Hiányzó anyagok pótlása” gombjával | `work/<EP>/intake.json` |
| Új filmprojekt egyetlen fájlból | a Könyvtár „Új filmprojekt” paneljén | `projects/<slug>.json`, `movies/<slug>/` |
| Új részek automatikus figyelése | a Könyvtár szekcióban (opcionális) | `series.auto_process`, `series.poll_s` a projektben |

### API-végpontok

| Végpont | Mit csinál |
|---|---|
| `GET /api/dub/projects` | projektek, epizódok, auditok, médiaszám elérhetősége |
| `GET /api/dub/state` | egy próba teljes állapota (beszélők, döntési modell javaslata, angol sorok) |
| `GET/POST /api/dub/config` | beállítások olvasása / mentése szerkesztéssel + biztonsági mentéssel |
| `GET /api/dub/voices` | a 13 hang katalógusa + modell-jelenlét |
| `POST /api/dub/audit` | új próba futtatása (start/duration/language) |
| `POST /api/dub/timbres` | a TTS-hangok CAM++ beágyazása (hangkiosztás bizonyítéka) |
| `POST /api/dub/speaker`, `POST /api/dub/voice` | emberi döntés rögzítése |
| `GET /api/dub/frame`, `GET /api/dub/clip` | képkocka / kivágás meghallgatása |
| `POST /api/dub/run`, `GET /api/dub/run`, `POST /api/dub/run/stop` | pipeline-lépés indítása / állapota + napló / leállítása |
| `GET /api/dub/library` | sorozat/film könyvtár, epizód-státusz, hiányzó anyagok |
| `POST /api/dub/series/scan`, `POST /api/dub/series/add` | új média keresése / hozzáadása (hardlink vagy másolás) |
| `POST /api/dub/series/process` | a kész epizódok szinkronjának indítása |
| `POST /api/dub/intake` | nyers epizód intake-je (felirat + hang), opcionálisan `then_run` |
| `POST /api/dub/movies` | filmprojekt egyetlen fájlból, majd intake + első szinkron |
| `GET/POST /api/dub/series/watch` | automatikus újrész-figyelés be/ki |
| `GET /api/dub/player`, `GET /api/dub/player/media`, `GET /api/dub/player/subtitle` | lejátszó manifest, MP4 proxy, WebVTT |
| `GET /api/dub/player/status`, `POST /api/dub/player/prepare` | proxy állapota / aszinkron előkészítése |
| `POST /api/dub/player/sample` | rövid magyar ellenőrző klip a kész dub.wav-ból |

A beállítás-szerkesztés szigorú szabályai (a kérés ellen is védi a JSON-t):
- `tuning`/`gate`: csak szám — beágyazott szótár (`asr_cer_max`) is szám-leveles kell legyen;
- `models`: csak szöveg-útvonal;
- `llm`/`asr`: szám, szöveg vagy logikai érték;
- a hang-katalógus lekérés **nem tölt be modellt** (csak mella-létezik-nét néz) — a
  teljes model betöltés ~1 GB lenne.

Egyszerre egy futás futhat; a második indítás 409-et kap. A leállítás
`terminate` → 10 mp-es várás → `kill` sorrendben.

### Kutatás: mit vettünk át más repókból

A hasonló nyílt forráskódú szinkron-projekteket (pyVideoTrans, Dub Studio,
youtube-auto-dub, Edge-TTS-Subtitle-Dubbing, mazinger, VideoLingo, Subdub,
Linly-Dubbing, OmniPlay) átnéztük; a tanulságok és a velünk tett/nem tett
beépítési döntés: **`docs/hasonlo-repok.md`**.

A 2026-11-07-i, kifejezetten AI-videószerkesztőkre, TTS- és hangklónozó
motorokra, valamint Snapdragon X Elite / Windows ARM64 kompatibilitásra
irányuló kutatás: **`docs/github-ai-video-tts-2026-11-07.md`**. Fő döntés: a
meglévő sherpa-onnx + Supertonic + Piper + ffmpeg lánc marad; CUDA nincs, a
QNN/NPU csak külön exportált modellekkel lenne opció.

## Képi beszélő-ellenőrzés (ASD)

A `work/visual_asd.py` a tanított **Light-ASD** modellel (CVPR 2023, 1 M
paraméter, tisztán CPU) mondja meg, melyik arc szája jár a hallható beszéddel:
25 fps, YuNet arckövetés, 112×112-es kivágás, MFCC 13. A korábbi,
korreláció-alapú döntést kivettük — a mérés szerint a jel a zajszinten van
(15 hosszú soron 0 találat), míg a Light-ASD ugyanazokon 0,68–0,95-ös
valószínűséggel nevezte meg a beszélőt.

A kimenet háromféle lehet: „arc #k beszél”, „nincs beszélő arc a képen”
(képen kívüli beszélő) és „nem eldönthető” — ez **bizonyíték, nem kapu**.
A kimenet soronként bekerül a `report.json` / `report.html` `asd` szakaszába
(arc, `p`, arcméret, beszéd-intervallum, kimenet), kaput nem állít és
hang-alapú döntést nem ír felül.

A döntés kézzel címkézett ground truthon mérve (15 hosszú sor, S02E04):
**AVA súly 6/9**, TalkSet 5/9 — ezért az AVA az alapértelmezés, és a
„beszél” küszöb 0,65 helyett 0,45 (a hallgató arcok 0,00–0,15-öt kapnak).
Mérési táblázat, a mérőlánc javításai (sötét jelenet detektálása, vágásnál
identitást váltó követés, jelenlét-maszk) és a súlyok letöltése:
**`docs/asd.md`**.

## Tesztek és ellenőrzés

```bash
cd dubstudio
../reader/.venv/Scripts/python.exe -m unittest discover -s tests -v
```

A Web UI réteg (beállítások, futtatás) unit-tesztjei az Aurisban:

```bash
cd ../reader
./.venv/Scripts/python.exe -m unittest tests.test_dub_webui -v
```
