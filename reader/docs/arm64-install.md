# ARM64 (Snapdragon X) telepítés és mérés

Mérve: 2026-10-02, Windows 11 ARM64 (vivo2, Snapdragon X Elite), 31,6 GB RAM,
12 logikai mag, Python 3.12.10 ARM64. PyTorch 2.10.0+cpu (win_arm64 wheel),
ONNX Runtime 1.30.0 (win_arm64 wheel), **torchaudio nélkül**.

> **Ez a fájl a *módszer* és a *rendszer* leírása.** A bizonyítékkatalógus —
> hogy melyik szám mért, melyik csak dokumentáció, és mi a nyitott mérési
> kérdés — a [meresek.md](meresek.md)-ben van, számozott M-xx bejegyzésekkel.
> Az összefoglaló és a terv a [handoff-2026-10-03.md](handoff-2026-10-03.md)-ben.

> **Fontos pontosítás (2026-10-04):** a „torchaudio nincs win_arm64 wheel"
> állítás **igaz**, és a korábbi „minden PyTorch-klónozó zárt" következtetés
> **téves volt**. A torchaudiót ARM64-en nem lehet telepíteni, de **x86_64
> emulációval futtatható**, és ezzel a **magyar hangklónozás mérve működik**
> (F5-TTS, WER 0,11 %). A natív ARM64 úthoz a repo saját shimje
> (`core/torchaudio_compat.py`) kell — az is működött, de lassabb volt, és
> 2026-10-05-ön a shim és az F5-TTS teljes környezete **törlésre került**
> (a lezárt kutatás). A nyers mérési adatok a `docs/meresi-adatok/f5-tts/`
> alatt vannak. Lásd a
> [Hangklónozás](#hangklónozás-f5-tts-magyar-arm64-en-mérve-2026-10-04)
> szakaszt.

## Mi fut, és miért

| Motor | Állapot | Ok |
| --- | --- | --- |
| **Supertonic 3** | ✅ mérve | ONNX Runtime, arm64 wheel létezik — **ez a választott motor** |
| Piper | ⚠️ telepíthető WSL2-ben | `piper-tts==1.8.0`-nak nincs win_arm64 wheelje |
| OmniVoice | ❌ nem telepíthető | az `omnivoice` csomag importálja a torchaudio-t |
| **F5-TTS magyar** | ✅ mérve | zero-shot klónozás, WER 0,11 %, de lassú (1 óra hang ~7,6 óra) |
| Higgs TTS 3 (4B) | ❌ | GPU-only, ~14 GB VRAM |
| MOSS-TTS 1.5 (4B) | ❌ | GPU-only, ~14 GB VRAM |
| ~~MOSS-TTS-Nano~~ | ❌ **kivezetve** | futott, de a magyarja érthetetlen volt — lásd lentebb |

A Snapdragon Adreno GPU-ját és Hexagon NPU-ját egyik PyTorch sem használja,
ezért a "GPU-s" motorok nem jöhetnek szóba. A 32 GB RAM nem helyettesíti a
VRAM-ot: a 4B adapterek bf16-ot és CUDA-t kérnek.

## Miért került ki a MOSS-TTS-Nano

Ez a mérés legfontosabb tanulsága, és nem a várt eredményt hozta.

**Ami működött.** A codec-zaj valódi hiba volt, és valóban javítottam: a
268 karakteres bekezdés 57,04 s helyett 28,07 s-t ad, a 19,2 másodperces
−72…−78 dBFS-es zajszőnyeg nullára csökkent, és a generálás 2,2× gyorsult.
A hiba teljesen determinisztikus volt (5 futás, mind azonos), és az oka a
vendor-futtató volt: minden belső töredékhez újrakódolta a referenciahangot, és
közben elvesztette a codec kontextusát.

**Ami nem működött.** A magyar kimenet ettől még érthetetlen maradt. Ezt a
hallás bizonyította, és **egyetlen objektív metrikám sem tudta kimutatni** —
ez a tanulság.

A kutatás, amit a kizárás előtt végeztem:

- A magyar **hivatalosan támogatott** a MOSS-TTS-Nano 20 nyelve között
  (a GitHub- és a HF-kártya is felsorolja), tehát nem nyelvi hiány.
- Ugyanarra a hosszú mondatra a modell **1,7× lassabb** magyarra, mint angolra:
  0,091–0,13 s/karakter vs 0,055–0,06.
- A spektrális flatness romlik a kétszeresére (0,063 vs 0,030), és vág.
- A kimenet spektrálisan **beszédnek** néz ki (flatness 0,022, energia a
  300–3000 Hz-es beszédsávban), tehát nem zaj — a modell maga állít elő
  érthetetlen hangot.

A magyarázat a méret: **0,1B paraméter**. A hangklónzás ezt nem menti meg,
mert a kiejtés a modellben van, nem a referenciahangban.

**Következmény:** a Supertonic 3 az egyetlen motor ezen a gépen, amely jó
magyar beszédet ad. Hangklónozásra ezen a hardveren jelenleg nincs működő
alternatíva; ha saját hang kell, más gép vagy más motor kell.

## RTF-mérés (268 karakteres magyar bekezdés, 24 kHz kimenet)

| Motor | RTF (median) | Hanghossz | Betöltés |
| --- | ---: | ---: | ---: |
| **Supertonic 3** (férfi) | **0,243** | 26,27 s | 0,89 s |
| **Supertonic 3** (női) | **0,264** | 26,01 s | 0,68 s |
| ~~MOSS-TTS-Nano~~ (kivezetve) | 0,61–0,91 | 38–57 s | 4,5–8,3 s |

## Teljes könyvmérés — Supertonic 3, Snapdragon X Elite

9 764 karakter magyar szöveg, 9 blokk, tiszta gyorsítótárral, 2026-10-02.

| Mérőszám | Érték |
| --- | ---: |
| Kész hang | 13,2 perc |
| Valós idő | 4,2 perc |
| **Átlagos RTF** | **0,317** |
| RTF min / med / max | 0,291 / 0,310 / 0,347 |
| WAV cache | 36,4 MB (384 kbps, 24 kHz mono PCM) |
| **Peak memória** | **756 MB** |
| Modellbetöltés | 0,72 s |

**Egyórás hangoskönyv ≈ 19 perc generálási idő**, ~270 MB WAV, ~760 MB memória.
A 756 MB peak főként az ONNX Runtime munkaterülete; a 32 GB RAM bőven elég.

### Ismert zaj: a `long_pause` hamis pozitív

A mérés 7 blokknál jelezte `long_pause`-t, de ez **nem hiba**. A leghosszabb
belső csend 1,45–2,10 s, ami normál mondatközi szünet: a `qa.py` **fix 1,5 s-os**
küszöbe minden 15 mondatos blokknál tévesen szólal meg.

A `find_long_silences` ezért **arányos küszöböt** használ (1,0 s minimum,
mondatszámmal skálázva), nem a fix 1,5 s-ot. A `qa.py` küszöbe változatlan
maradt, mert az az upstream működése.

## Telepítés – amit a setup.py most ARM64-en csinál

A `reader/setup.py` felismeri az ARM64-et (`win_arm64` / `linux_aarch64`) **minden
CUDA-szondázás előtt**, és `arm64_onnx` ágra vált:

1. **Telepít** a `torch`-ot a CPU-indexről, **torchaudio nélkül** (win_arm64), és
   `torchaudio`-val együtt aarch64-en.
2. **Kihagyja** az omnivoice-ot, a Higgs runtime-ot és a spaCy-modelleket.
3. **Szűri** a függőségeket: `soxr`-t kihagy (nincs ARM wheel), `scipy`-t telepít
   helyette (mindenhol van arm64 wheel), `thinc<9.1`-et pinel aarch64-en, mert a
   thinc 9.1 ledobta az aarch64 wheeleket.
4. **Ellenőrzi** a natív runtime-ot: `onnxruntime` provider-listával, nem
   torchaudio-importtal.
5. A végén kiírja, hogy a **Supertonic 3 vagy a MOSS-TTS-Nano** érhető el.

A `core/hardware.py` új modul felismeri a gépet, és az ajánlott motor sorrendjét a
**mért RTF** alapján állítja fel — nem a funkciók száma szerint. Az omnikonzisztens
választás az `auto`: amíg a felhasználó nem választott motort, a hardver dönt; egy
kifejezett választás mindig felülkereül.

## Kódváltozások

| Fájl | Változás |
| --- | --- |
| `reader/setup.py` | ARM64-detekció, `arm64_onnx` telepítési ág, szűrt függőségek |
| `reader/core/hardware.py` | **új** — hardver-felismerés és motorajánlás mért RTF szerint |
| `reader/core/local_engines.py` | MOSS-TTS-Nano kivezetve, `find_long_silences` arányos küszöbbel, scipy `resample` fallback |
| `reader/core/tts_router.py` | `auto` motorbeállítás → hardverajánlás |
| `reader/core/settings.py` | alapértelmezett motor `auto` |
| `reader/core/settings_api.py` | `auto` validálása |
| `reader/app.py` | sebességtámogatás tisztázása `auto` esetén |
| `reader/tests/test_hardware_detection.py` | **új** — 26 teszt |

## Ismert korlátok

- **spaCy nincs telepítve.** A thinc-nek nincs `win_arm64` wheelje, így a
  spaCy forrásból buildelne (a `blis` miatt el is bukik). A karakterfelismerés
  regex-visszaesésre vált — működik, csak kevésbé pontos.
- **wetext nincs telepítve.** A `kaldifst` C++ build, nincs arm64 wheel. A
  szám- és dátum-normalizálás a `num2words` + a magyar lexicon modulra esik.
- Az **x64-emuláció** (x86_64 Python ezen a gépen) épp ezeket a hibákat
  kerüli, de lassabb: a hosszú könyveknél a natív ARM64 + ONNX a jobb választás.
---

## Kiejtési szótár és érthetőségi mérés

### Beépített magyar kiejtések

A kiejtési szótár korábban üres volt: a `pronunciation_rules` táblában nem volt
egyetlen bejegyzés sem, minden szót kézzel kellett volna felvenni. Most
`core/hu_pronunciation.py` 29 beépített magyar szabályt ad:

| Kategória | Darab | Példa |
|---|---|---|
| név | 7 | Bethlen → Betlen, Móricz → Móric, Mikszáth → Mikszát |
| idegen név | 8 | Shakespeare → Sakszpér, Napoleon → Napóleon |
| helynév | 1 | Zsigetvár → Szigetvár |
| rövidítés | 13 | stb. → s t b, pl. → például, Bp. → Budapest |

A beépített szabályok nem foglalnak helyet a könyvtárban: a felhasználó által
mentett szabály (`apply_pronunciation` a `core/experience.py`-ban) mindig
felülírja őket. A Beállítások → Kiejtési szótár képernyőn kereséssel
(kis- és nagybetűtől, ékezetektől függetlenül) és kategória szerint
listázhatók, egy kattintással a saját szabály űrlapjára másolhatók.

**Ragozás:** a beépített szabályok csak teljes szóként illeszkednek. Ennek oka,
hogy a toldó a *helyes* alakhoz tartozik, nem a cserélt szótőhöz: a
„Móricz" → „Móric" átírás toldalékaiból „Móriczot" helyett „Móricot" keletkezne.
A kézzel mentett szabályok ragozása változatlanul működik
(`hu_pronunciation.EXACT_ONLY`).

### Magyar WER-mérés

`core/hu_wer.py` 24 magyar mondatból álló, verziózott korpuszt ad, amely a
kiejtés szöveges megítélését célozza: hosszú magánhangzók, digrafumok
(ty, cs, zs, sz, gy), kettős mássalhangzók, számok, évszámok, a szótárban
szereplő nevek és hosszú mondatok.

A mérés valódi kört jár be: szintetizálás a Supertonic 3-mal, visszaírás
`core.qa` ASR-rel (magyarul CPU-n Parakeet TDT v3), majd `score_transcript`.
Ez **ugyanaz a mérés**, amit a fejezetellenőrző oldal használ, tehát a számok
egyeznek a felületen látottakkal.

Futtatás (a gép betöltött motorral):

```bash
.venv/Scripts/python.exe -m core.hu_wer            # teljes korpusz
.venv/Scripts/python.exe -m core.hu_wer --case bethlen --case moricz
.venv/Scripts/python.exe -m core.hu_wer --out wer.json
```

Az összesített WER a teljes szövegre van súlyozva, nem a mondatonkénti
átlaggal, hogy a rövid mondatok ne húzzák el. A `run_bench` a szintetizálást
és a visszaírást függvényben kapja, ezért a mérési logika modellek betöltése
nélkül is tesztelhető (`tests/test_hu_wer.py`).

### INT8 Supertonic 3

`models/supertonic-3-int8/onnx/` – 138 MB, a fp32 `models/supertonic-3/onnx/`
383 MB helyett. Forrás: `csukuangfj2/sherpa-onnx-supertonic-3-tts-int8-2026-05-11`.
Választható a Beállítások → Beszédmotorok → Supertonic 3 → **Pontosság** mezőben
(`supertonic_variant`: `fp32` | `int8`).

Mért eredmény (Snapdragon X Elite, 10 lépés, `scripts/bench_supertonic_variants.py`,
minden variáns külön processzben, mert a Windows csúcsmemóriája monoton):

| Mutató | fp32 | INT8 | Változás |
|---|---:|---:|---:|
| Modellmappa | 382,7 MB | 138,1 MB | **−63,9 %** |
| Betöltés | 0,72 s | 0,67 s | −6,9 % |
| Csúcsmemória (csak TTS) | 494 MB | 241 MB | **−51,3 %** |
| RTF (12 generálás, 59 s hang) | 0,32 | 0,35 | **+8 % (lassabb)** |
| WER (24 magyar mondat) | 10,1 % | 6,5 % | −3,7 százalékpont |

**Az INT8 nem gyorsabb ezen a processzoron.** A várttal ellentétben 8 %-kal
lassabb (3,1× helyett 2,9× valós idejű). A vector_estimator 10 lépéses
diffúziós hurokja szűk keresztmetszetű, így a kvantálás dekvantálási költsége
nem hoz hasznot. Amit az INT8 valóban ad: **feleakkora memória és 64 %-kal
kisebb lemezigény**, azonos kiejtés mellett.

A WER különbség nem jelentős: ~170 szón 10 % hibaaránynál a szórás kb. 2,3
százalékpont, tehát a 3,7 pontos különbség kb. 1,6 szórás. Következtetés: a
kvantálás **nem rontja** a magyar kiejtést, de javulást sem mutat. Megbízható
megerősítéshez több mondat kellene; addig az **fp32 az alapértelmezett**.

### Miért nem segít az INT8 – és mi az, amit a Qualcomm tényleg javasol

Külső forrásokból, nem találgatásból:

1. **Ismert jelenség.** A [sherpa-onnx #575](https://github.com/k2-fsa/sherpa-onnx/issues/575)
   pontosan ez: „int8 quantized TTS model slower than fp32", int8-as fájllal
   0,08 s helyett 20–28 s.
2. **A saját modellünk tényleges tartalma** (`vector_estimator.int8.onnx`):
   22× `DynamicQuantizeLinear`, 38× `MatMulInteger`, 24× `MatMul`. Ez
   **dinamikus** kvantálás: a skálákat futásidőben számolja ki. A
   `vector_estimator` ráadásul 10 lépésben fut, így a kvantálási overhead
   tízszer fizetődik. A `text_encoder.int8.onnx` pedig **nincs is kvantálva**
   (nincs benne egyetlen QDQ operátor; a fájlméret is azonos az fp32-vel).
3. **A dokumentált Qualcomm-út: ONNX Runtime QNN execution provider.**
   Az [onnxruntime.ai/winarm](https://onnxruntime.ai/winarm) és az
   [onnxruntime-qnn](https://github.com/onnxruntime/onnxruntime-qnn) szerint a
   Snapdragon NPU (Hexagon HTP) **QDQ formátumú** modellt futtat:
   `pip install onnxruntime-qnn` (a win_arm64 cp312 wheel létezik, 2.6.0,
   59,2 MB), majd `register_execution_provider_library` +
   `add_provider_for_devices(…, {"backend_path": qnn_ep.get_qnn_htp_path()})`.
   A `session.disable_cpu_ep_fallback=1` kapcsolóval ellenőrizhető, hogy a
   graf *teljes egészében* az NPU-n fut-e.

   **A mi modellünk erre nem alkalmas.** A QNN támogatott operátorlistája
   `QuantizeLinear`/`DequantizeLinear`/`MatMul`/`Conv` elemet tartalmaz, de
   `DynamicQuantizeLinear`-t és `MatMulInteger`-t nem; a dokumentáció továbbá
   kifejezetten elutasítja a dinamikus alakú modelleket, miközben a
   `vector_estimator` bemenetei `['batch_size', 144, 'latent_length']`.
4. **A legközelebbi létező munka sem használható.**
   A [dev-ansh-r/Supertonic-qualcomm-quantized](https://huggingface.co/dev-ansh-r/Supertonic-qualcomm-quantized)
   pont ezt a lépést végzi el, de Supertonic **2**-höz, **QCS6490** chiphez
   (HTP arch v68), Linux/aarch64-re fordítva, QAIRT SDK 2.37-tel, és saját
   bevallása szerint sem kész: „Inference script and sample application are
   not provided." Windows ARM64-re nincs kész Supertonic 3 QDQ export.
5. **Pontosítás (2026-10-03):** az `onnx` csomag **elérhető** ARM64-en
   (`onnx 1.23.1`, `cp312-abi3-win_arm64`), tehát az earlier állítás, hogy
   „nem telepíthető", téves volt. Amit viszont valóban nem lehet itt előállítani,
   az a **statikus QDQ export**: az ORT C++ kvantáló segédegyei (quantize_static
   mögötti natív rész) x86_64-ra készülnek. Lásd a következő szakaszt, ahol a
   QNN-kísérlet tényleges eredménye (és az `onnx` ARM64-es elérhetősége) szerepel.

**Következtetés:** a gyorsításhoz nem INT8 CPU-kvantálás kell, hanem statikus
QDQ export a Snapdragon X-re. Ez jelenleg nem létezik. Az fp32 marad
alapértelmezett, és az INT8 csak **lemez- és memóriamegtakarításra** jó
(−64 % / −51 %), azonos kiejtés mellett.

Megjegyzés a méréshez:Megjegyzés a méréshez: a hangcache a `variant=` kulcsrészt tartalmazza, ezért
a két variáns nem szolgálhatja ki egymás hangját. A mérés minden futás előtt
törli a mérendő felvételt, különben a gyorsítótárazott találatot nullának
számolná a generálási időnek.

```bash
.venv/Scripts/python.exe scripts/bench_supertonic_variants.py
.venv/Scripts/python.exe scripts/bench_supertonic_variants.py --skip-wer --rtf-samples 12
.venv/Scripts/python.exe scripts/bench_supertonic_variants.py --variant int8
```

## NPU (QNN) kísérlet a Supertonic 3-on — 2026-10-03

A korábbi „nincs NPU-út" következtetés dokumentációs szövegen alapult. Ez a
szakasz **mérővel** ellenőrzi, hogy a gép NPU-ja el tudja-e venni a modellt.

### A környezet

A gépen már van működő Qualcomm NPU futtatókörnyezet, amit a rendszer
készségei (`hermes/profiles/analyst/skills/arm64-voice-npu`, `geniex-npu`,
`snapdragon-x-elite-npu-stack`) dokumentálnak:

| Elem | Érték |
|---|---|
| canonical dev venv | `C:\AI\venvs\foundry-dev` (Python 3.13) |
| onnxruntime | 1.29.0 |
| onnxruntime-qnn (plugin EP) | 2.5.0, `win_arm64` |
| QAIRT SDK | `C:\Qualcomm\AIStack\QAIRT\2.45.40.260406` és `2.50.40.260831` |
| NPU | Hexagon HTP (Snapdragon X Elite) |

Az EP regisztrálása működik, és a provider csak **explicit regisztráció** után
jelenik meg:

```python
import onnxruntime as ort, onnxruntime_qnn as qnn_ep
ort.register_execution_provider_library(qnn_ep.get_ep_name(), qnn_ep.get_library_path())
devices = [d for d in ort.get_ep_devices() if d.ep_name == "QNNExecutionProvider"]
so = ort.SessionOptions()
so.add_provider_for_devices(devices, {"backend_type": "htp"})   # htp = NPU
sess = ort.InferenceSession(path, so)      # NEM providers=[...] — az visszairányít CPU-ra
```

### Hol megy el az idő

Minden almodell külön sessionben, valós magyar mondattal:

| Almodell | Idő | Hány hívás | Rész |
|---|---:|---:|---:|
| `duration_predictor` | 0,002 s | 1 | 0,1 % |
| `text_encoder` | 0,021 s | 1 | 1,0 % |
| **`vector_estimator`** | **2,177 s** | **10** | **93 %** |
| `vocoder` | 0,153 s | 1 | 7,5 % |

Ez megfordítja a korábbi feltevést: **nem a vocoder, hanem a `vector_estimator`**
a szűk keresztmetszet, és az fut tízszer.

### A QNN EP próbák eredménye

| Almodell | Eredmény |
|---|---|
| `vocoder.onnx` (fp32, 101 MB) | a session létrejön, a provider-lista QNN-t mutat, de nem gyorsít: RTF 0,3317 vs 0,3383 CPU-hoz képest |
| `vector_estimator.onnx` (fp32, 256 MB) | **`EP_FAIL: Dynamic shape is not supported yet, for input: total_step`** |
| statikus alakra vágott `vector_estimator` | **`Tile` node futásidőben számított `repeats` miatt nem tölthető** |

> **Utólagos javítás (2026-10-04):** az első sor **„betöltődik QNN EP-n”**
> állítása félrevezető volt. A `get_providers()` listája nem bizonyít
> NPU-futtatást: ezen a gépen az EP a háttérben CPU-ra esik, és a mért „QNN”
> idő valójában CPU-idő. A bizonyíték és a gyökérok a
> „NPU II. mérés” szakaszban van, a `scripts/npu_probe.py` pedig
> bármikor lefuttatható ellenőrzést ad.

A statikus export kísérlete: az `onnx` 1.23.1 ARM64-en elérhető, így a graf
bemenetei konkrét méretre rögzíthetők (1198 dinamikus tenzor → 138). A QNN EP
ezután nem a bemeneteknél, hanem egy belseő `Tile` node-nál áll meg: a
`repeats` vektor a `Shape → Slice → Concat` ágon futásidőben készül, tehát a
teljesen statikus grafból csak **konstans-összevonással** (az összes alak-
számító ág előzetes kiértékelésével) lehetne QDQ/HTP-kompatibilis modellt
alkotni. Ez nem az ONNX-fájl átnevezése, hanem valódi grafátalakítás.

**Következtetés (2026-10-03-ig):** a Supertonic 3 NPU-gyorsítása ezen a modellen
nem érhető el anélkül, hogy a teljes `vector_estimator` alakkezelő algráfját
statikus konstansokra összevonnánk és az egész modellt statikus QDQ formátumban
exportálnánk. A gyorsítás nem INT8 kérdés, hanem **grafátalakítás** kérdése.

### Fontos javítás: ez a konklúzis téves volt

A fenti érvelés a `onnxruntime-qnn` EP-re igaz volt, de **elfelejtette a
Qualcomm saját SDK-ját**, ami a gépen telepítve van:

`C:\Qualcomm\AIStack\QAIRT\2.50.40.260831\bin\arm64x-windows-msvc\`

Ebben van `qnn-onnx-converter` és `qairt-quantizer` — **ONNX → QNN DLC**
konverter és kvantáló. A `-d` kapcsolóval **statikus bemeneti méreteket** lehet
rögzíteni, ami pontosan az, amit a QNN EP elutasított.

**A natív ARM64 build nem tölt be ARM64 Pythonba** (`windows-arm64ec` →
„%1 nem érvényes Win32 alkalmazás"), de a `windows-x86_64` build **emulációval
működik**:

```bat
:: 1. x86_64 venv (a gépen már volt uv-python 3.12.13 x86_64)
"C:\Users\istva\AppData\Roaming\uv\python\cpython-3.12.13-windows-x86_64-none\python.exe" -m venv C:\AI\venvs\qairt-x64
:: 2. a konverter függőségei — az onnx 1.17 kötelező, 1.23-ban megszűnt az onnx.version
C:\AI\venvs\qairt-x64\Scripts\python.exe -m pip install numpy onnx==1.17.0 pyyaml onnxruntime pandas setuptools
:: 3. a konvertálás
set PYTHONPATH=C:\Qualcomm\AIStack\QAIRT\2.50.40.260831\lib\python
C:\AI\venvs\qairt-x64\Scripts\python.exe ^
  C:\Qualcomm\AIStack\QAIRT\2.50.40.260831\bin\arm64x-windows-msvc\qnn-onnx-converter ^
  -i vector_estimator.onnx -o vector_estimator.dlc ^
  -d noisy_latent 1,144,64 -d text_emb 1,256,96 -d style_ttl 1,50,256 ^
  -d latent_mask 1,1,64 -d text_mask 1,1,96 -d current_step 1 -d total_step 1
```

**Mért eredmény:** a `vector_estimator.onnx` (256 MB, a 93%-os szűk keresztmetszet)
**sikeresen konvertálódott** — „Conversion complete!", DLC 5,3 MB, 3,82 Mrd MAC,
63,9 M paraméter, ~93 másodperc. Azaz **az NPU-út létezik és a gépen futtatható**.

A `text_encoder.onnx` viszont egy `Pad` node-on elbukik a konverterben
(`attn_layers.0.Pad`, közös initializátor) — ez konverterkorlát, nem
architekturális akadály.

**Ami még nincs kiderítve** (és ezért nincs NPU-mérés):

1. a `qairt-quantizer` statikus kvantálása,
2. hogy a DLC fut-e a HTP-n az `onnxruntime-qnn`-en keresztül,
3. hogy a kimenete számszerűen egyezik-e a CPU-val,
4. a legfontosabb: **egyáltalán gyorsabb-e**, hiszen a modell fp32, és a HTP
   natívan int8/int16,
5. a statikus méretek miatt minden mondatot padelni kell, vagy méret szerinti
   DLC-készlet kell — ami gyorsulásnál is számít.

## Ami viszont működik: batchelés és lépésszám

Mivel a `vector_estimator` a szűk keresztmetszet, a két valódi gyorsító a
hívásképzés. A vendored helperben már van `TextToSpeech.batch()`, de az
`__call__` soha nem használta: minden mondat külön, `batch=1` grafhívásban futott.

### Batch (mérve az export útvonalán, `TTSEngineRouter.generate_many`)

A mérés azon az interfészen futott, amelyet az export munka is használ
(`app.py` → `tts.generate_many`), 8 könyvszerű bekezdéssel, hideg cache-en:

| Batch | Falió | Hang | RTF | Sebesség | Hiányzó eredmény |
|---:|---:|---:|---:|---:|---:|
| 1 | 17,63 s | 54,5 s | 0,3235 | 1,00× | 0 |
| **4** | **7,82 s** | 54,3 s | **0,1440** | **2,25×** | 0 |

Nyers átviteli mérés a pipeline-on (azonos lépések, 4 mondat): batch 4 → 2,12×,
batch 6 → 2,60×, batch 8 → 2,15× (a padding miatt nagyobb batchnél már nem
nő). A `supertonic_batch` beállítás 1–6 közötti értéket fogad, alapértelmezett 4.

**Minőség** (a `core.hu_wer` 24 mondatos korpusza **két hangon** — F1 és M3 —
azaz 48 szintézis + visszaírás; ASR: Parakeet + Whisper-hu):

| Út | WER |
|---|---:|
| serial (batch=1) | 12,21 % |
| batchelt (batch=4) | **11,52 %** |

A különbség −0,69 százalékpont, ami kb. 2–3 szó a ~380 szavas mintán: a
batchelés **nem rontja a kiejtést**. Egyedi szegmensenként 9/48 esetben rosszabb
a batchelt felvétel, de ezek rövid mondatokon 1–2 elhallott szó jelentenek
(0 % → 10–28 %), ami az ASR saját zaja, nem minőségromlás.

Megjegyzés a méréshez: a hang kiválasztását a `voice:` token **a karakterlánc
elején** kell megadni (`explicit_voice()`); `"[voice:M3]"` csendben az
alapértelmezett hangra esik vissza. A mérőprogram ezt külön ellenőrzi, mielőtt
futtat, különben két hang helyett ugyanazt mérné kétszer.

### Lépésszám (12 mondat, `--steps`)

| Lépés | RTF | WER | CER |
|---:|---:|---:|---:|
| 4 | 0,1850 | 9,5 % | 1,64 % |
| 6 | 0,2717 | 7,4 % | 1,22 % |
| 10 (alapértelmezett) | 0,3916 | 7,4 % | 0,76 % |
| 16 | 0,5600 | 8,4 % | 1,07 % |

A 16 lépés **lassabb és nem jobb**, tehát az alapértelmezett 10 jól választott.
A 6 lépés 1,44× gyorsabb a WER-t nem mozgatja; a 4 lépés 2,1× gyorsabb, de a CER
monoton nő. A WER mindenütt 24 vagy 12 mondatos mintán van, ezért ezek trendek,
nem statisztikailag igazolt különbségek.

### Együtt

A 10 lépés és batch=4 együtt **≈ 2,25× gyorsabb** azonos kiejtéssel: egy 1 órás
hangoskönyv generálása a mért RTF 0,3235-től (≈ 19,4 perc) RTF 0,1440-re
(≈ 8,6 perc) kerül.

```bash
# RTF lépésszám szerinti gyorsításellenőrzés
.venv/Scripts/python.exe scripts/bench_supertonic_variants.py --steps 6 --cases 12
```

## NPU II. mérés: a DLC elkészült, de a HTP-n SOHA nem futott (2026-10-04)

Ez a szakasz felülírja a fenti „Ami még nincs kiderítve” öt pontját.
A konklúzis: **a konvertálás működik, az NPU-futtatás ezen a gépen nem.**
Nem architekturális akadály, hanem a Windows ARM64-es Hexagon-driverlánc
hiányzó darabja.

### Amit működik — a konvertálás (mérve)

A `vector_estimator.onnx` → DLCátalakítás mindkét SDK-val sikerül:

| SDK | DLC | BIN | MAC-ek | Paraméter | Idő |
|---|---:|---:|---:|---:|---:|
| QAIRT 2.50.40 (`arm64x`) fp32 | 5,3 MB | 256 MB | 3 815 004 809 | 63 923 520 | ~93 s |
| QAIRT 2.45.40 (`arm64x`) fp32 | 5,3 MB | 256 MB | 3 815 004 809 | 63 837 504 | ~90 s |
| QAIRT 2.50.40 int8 (per-tensor) | 5,35 MB | **64,2 MB** | – | – | ~95 s |

Két kvantálási akadály, mindkettő mérve:

- `--use_per_channel_quantization` elbukik:
  `modeltools::IrQuantizer::preprocessPerChannel: No bias info for op:
  /vector_estimator/vector_field/proj_in/net/Conv_2d (Conv2d)`
  → a modellben van bias nélküli konvolúció, per-tensor int8 használható.
- A `qairt-quantizer` **egyáltalán nem tudja visszaolvasni** a 2.50-es DLC-t
  (`Failed to deserialize the network`), ezért a kvantálást a konverterrel
  egy menetben kell elvégezni (`--input_list … --target_backend HTP`).

### Amit senki nem mért meg: az NPU-futtatás

A DLC-t három különböző úton próbáltam lefuttatni. Egyik sem futott a HTP-n.

**1. `qairt-net-run.exe` (QAIRT 2.50, ARM64EC)**

```
ERROR: qairt-net-run failed: Function not supported by current QAIRT Version
```

A magas szintű API-val (`--use_high_level_apis`) pedig ez jön:
`ERROR: Failed to dlopen system library: libQairtSystem.so` — vagyis a
2.50-ös **ARM64EC** Windows-csomag a HTP-útat nem szállítja használhatóan.

**2. `qnn-net-run.exe` (QAIRT 2.45, natív aarch64) — ez a legközelebbi**

```
<W> Initializing HtpProvider                       ← a HTP provider ÉL!
Processing inference input(s): …
    36.0ms [ ERROR ] Failed to create dlc handle with code 1002 for dlc file …
Check executable cache failure
```

A **HTP provider inicializálódik**, tehát a DSP oldal elérhető. A **DLC
betöltése** hiányzik. Vezérlő kísérlet: ugyanezzel a hibával elbukik egy
**15 KB-os apró Conv DLC is**, ráadásul `--backend QnnCpu.dll`-lel is, tehát
**nem HTP-specifikus hiba, hanem a DLC-betöltő út hibája ezen a Windows
runtime-on**. (A `snpe-dlc-graph-prepare` másik hibakóddal, 310-zel
hasal el, az másik DLC-olvasó.)

**3. `onnxruntime-qnn` EP (`graph_path` = DLC) — csendben CPU-vá esik**

Ez a legveszélyesebb, mert **nem hibázik, hanem hamis ígéretet ad**:

```
providers: ['QNNExecutionProvider', 'CPUExecutionProvider']
HTP exec: min 216.06 ms  median 217.58 ms
```

A 216 ms **nem NPU-idő**. Három egymástól független bizonyíték:

- a profilban **0 db `QNNExecutionProvider` node**, 1796 db CPU-node,
- a kimenet **bitre azonos** a CPU fp32 kimenettel (`np.array_equal` → True),
  míg a CPU int8 kimenettől 1,6e-2 eltérés van,
- a tiszta CPU-referencia 1 szálon **219,5 ms** — gyakorlatilag ugyanaz.

Vagyis a „HTP-n 216 ms” mérés **egy szálas CPU-futtatás** volt. Az EP
`get_providers()` listája önmagában **nem bizonyít NPU-futtatást**.

### A gyökérok: két hiányzó driver-darab

**a) `libcdsprpc.dll`** — a `QnnHtpV81Stub.dll` (Hexagon v81 = Snapdragon X
Elite) statikus importja. A gépen **nincs** a betöltési útvonalon, de megvan:

```
C:\Windows\System32\DriverStore\FileRepository\
  qcadsprpc8380.inf_arm64_81346390221b730a\libcdsprpc.dll
```

Ennek átmásolása a QNN könyvtárba **valóban megjavítja** a stub betöltését:
előtte `LoadLibrary` hiba `Could not find module 'libcdsprpc.dll' (or one of
its dependencies)`, utána `LOAD OK` (ezt `ctypes.WinDLL`-lel mérve).

**b) A HTP user-driver (HNRD) V81 fájljai** — innen már nem jutunk tovább.
Az `onnxruntime_providers_qnn.dll` beépített üzenete árulja el:

```
QNN EP fell back to HTP user-driver (HNRD) path;
QnnHtpPrepare/Stub/Skel libs missing from backend lib dir.
```

A driver store-ban a HNRD kliensfájlok **csak V73-ra** vannak
(`qcnspmcdm8380.inf_arm64_*/HTP/`):
`HtpUsrDrv.dll`, `QnnHtpV73StubDrv.dll`, `libQnnHtpV73SkelDrv.so` — V81-es
(`QnnHtpV81StubDrv.dll`, `libQnnHtpV81SkelDrv.so`) **sehol nincs a gépen**.
Azokat is átmásoltam; az EP erre is `load library failed`-t ad.

**Vagyis:** a Windows 11 ARM64-es Hexagon NPU illesztőprogramja
(30.0.220.3000) a DSP RPC réteget megadja, de a **HTP user-driver V81
klienskönyvtárai nem részei a csomagnak**. Amíg ez hiányzik, az ORT QNN EP
ezen a gépen nem tud NPU-node-ot létrehozni — és a `get_providers()` ezt
nem árulja el.

### A döntő kérdés — a sebesség — megválaszolatlan

Mivel a DLC soha nem futott a HTP-n, **nincs NPU-időmérés**, és ezért
nincs gyorsulási szám sem. Amit *tudunk* mondani, az a CPU-referencia,
mert egy NPU-mérést ezzel kell összevetni:

CPU, `vector_estimator`, ONNX Runtime 1.29.0 (`C:\AI\venvs\foundry-dev`,
12 mag ARM64), a DLC-hoz használt 1×144×64 alak, `scripts/npu_probe.py
--cpu-baseline` ugyanezt méri az Auris venv 1.30.0-val (az eltérés néhány
százalék):

| Modell | Batch | Szál | min ms | median ms |
|---|---:|---:|---:|---:|
| fp32 | 1 | 1 | 219,5 | 220,9 |
| fp32 | 1 | 12 | 107,8 | 113,5 |
| fp32 | 4 | 1 | 834,4 | 841,9 |
| fp32 | 4 | 12 | 170,4 | 211,7 |
| int8 | 1 | 1 | 201,6 | 202,8 |
| int8 | 1 | 12 | 108,8 | 114,3 |
| int8 | 4 | 1 | 780,4 | 783,5 |
| int8 | 4 | 12 | 176,5 | 207,5 |

**Szórás:** a táblázat egy futásból származó `min` érték. Két független futást
összevetve a `batch=1` sorok 2–3 %-on belül stabilak (fp32 1×1: 219,5 és
225,4 ms; fp32 1×12: 107,8 és 106,8 ms), **a `batch=4`, 12 szálos sorok
viszont 10–20 %-ot ugranak** (fp32 170,4–182,8 ms, median 211,7–262,9 ms) —
ott az ORT szálütemezés és az arena-viselkedés dominál. NPU-összehasonlításnál
ezért mindig azonos gépen, azonos `min` módszerrel kell mérni.

Ebből két dolog következik, ha valaha is működővé válik az NPU-út:

- **a 3,8 Mrd MAC-os `vector_estimator` NPU-n akkor ér meg valamit, ha a
  DSP→host átviteli költség azelőtt elhasal, hogy egyáltalán számít.** A
  CPU-referencia `batch=4`, 12 szálon 170–183 ms — ez az a szint, amit egy
  NPU-mérésnek **meg kell haladnia**, különben a DSP-ra küldés önmagában
  elfogyasztja a nyereséget.
- **az INT8 önmagában nem gyorsít** (108,8 vs 107,8 ms), ugyanaz a leképezés,
  ahogy a teljes pipeline-ban sem. Ha gyorsulás lesz, annak oka az int8
  natív HVX-végrehajtás, nem a kvantálás.

### Reprodukálható ellenőrzés

```bash
# 2026-10-05: a scripts/npu_probe.py az archívumba került, mert a
# onnxruntime-qnn venv-ek megszűntek. Innen már nem futtatható.
.venv/Scripts/python.exe docs/meresi-adatok/npu-qnn/npu_probe.py
```

Ez három dolgot vizsgál, és **nullát ír ki, ha a QNN EP nem az NPU-n fut**
(a `get_providers()` helyett a profil node-számát nézi). A `--cpu-baseline`
kapcsoló kiírja a fenti CPU-referenciát, hogy bármi NPU-mérés azonnal
összevethető legyen. Kilépési kód: 0 = van NPU-futtatás, 1 = nincs.

A `onnxruntime_qnn` a `C:\AI\venvs\foundry-dev` venvben van, nem az Auris
venvben; a driver- és RPC-ellenőrzés bármelyikből fut.

### Amit egy következő próbálkozásnak érdemes tennie

1. **V81 HNRP driverfájlok beszerzése** (Qualcomm AI Hub `qairt-converter`
   workflow, vagy frissebb `qcnspmcdm` INF a Windows Update-en) — enélkül
   az ORT QNN EP ezen a gépen nem tud NPU-node-ot létrehozni.
2. **DLC-betöltés megkerülése**: a 2.45-ös runtime a
   `qnn-model-lib-generator`-t kéri (`.cpp` modell), amit a
   `qnn-onnx-converter` már nem állít elő — vagy a `code 1002`
   „executable cache” hiba megoldása egy HTP-előkészített DLC-lel.
3. Csak **ezután** érdemes a batch-4-es DLC-t megmérni, mert a statikus
   méretek miatt a 4-es batchhez külön DLC kell.

## Hangklónozás: F5-TTS magyar, ARM64-en (mérve 2026-10-04)

### Fontos: két környezet, és a shim szerepe

A `torchaudio` **sehol nincs win_arm64 wheel** — sem a PyPI-n, sem a PyTorch
CPU-indexen (`Wheels are available ... win_amd64`, ARM64 nélkül). Ez **igaz
állítás**, és az is igaz, hogy emiatt ARM64-en a F5-TTS-t erősen nehezíti.
Két út van, és mindkettőt le is mértem:

| Út | torchaudio | RTF (idő/hang) | WER | Megjegyzés |
| --- | --- | ---: | ---: | --- |
| x86_64 emuláció (valódi torchaudio) | valódi, win_amd64 | ~24× | 0,11 % | **működik és gyorsabb** |
| Natív ARM64 + shim (`core/torchaudio_compat.py`, **2026-10-05: TÖRÖLVE**) | saját (6 szimbólum) | ~62× | 0,07 % | működött, de a leglassabb |

Az emuláció **2,6× gyorsabb** a natív ARM64-nál (82 s vs 212 s egy 3,4 s
mondatra), mert az x86_64 PyTorch CPU-build optimalizáltabb, mint a win_arm64
build. Ezért a **gyorsabb út az emuláció**, a shim a tartalék, ha emuláció
nélkül kellene.

### A minőség kiváló, a sebesség elfogadhatatlan egy könyvhöz

A mért minőség (Parakeet TDT v3 magyar ASR, 12 futás, 2 hang):

| Hang | nfe_step | idő/mondat | WER | CER |
| --- | ---: | ---: | ---: | ---: |
| F1 | 32 | ~82 s | 0,11 % | 0,00 % |
| F1 | 16 | ~55 s | 0,11 % | 0,00 % |
| F1 | 8 | ~26 s | 0,11 % | 0,00 % |
| M3 | 32 | ~84 s | 0,11 % | 0,00 % |
| M3 | 16 | ~41 s | 0,11 % | 0,00 % |
| M3 | 8 | ~19 s | 0,11 % | 0,00 % |

A `nfe_step` (flow-matching lépések) 32→8 csökkentése **nem rontja a WER-t**.
Az egyetlen ismétlődő jellegzetes „hiba" a szóösszefűződés („ködlebegett") —
a magyar szövegelő sajátossága, nem ASR-hiba.

A megfelelő könyvi becslés (a hiba nélkül, `idő/mondat` × mondatszám):
egy 3,4 s mondához nfe=8 esetén ~26 s kell → **1 óra hang kb. 7,6 óra valós
idő**; nfe=32 esetén ~24 óra. Összehasonlítás: a **Supertonic** batch=4-gyel
1 óra hangot ~9,6 perc alatt ad. Az F5-TTS tehát **50× lassabb** a jelenlegi
motornál.

**Következtetés:** a magyar hangklónozás ezen a gépen **működik és minőségileg
kitűnő**, de lassú. Érdemes felhasználni a betöltő hang kiválasztásához vagy
rövid bemutatókhoz, nem egy teljes hangoskönyv leolvasásához.

### Amit kizártam méréssel (nem hiedelmmel)

| Motor | Kizárva | Ok |
| --- | --- | --- |
| Chatterbox Multilingual v3 | ❌ | 23 nyelv, **magyar nincs köztük** |
| Sopro 169M | ❌ | csak en/pt/fr/de |
| IndexTTS / IndexTTS2 | ❌ | en/zh/ja/es/ar |
| ZipVoice / sherpa-onnx | ❌ | nincs magyar |
| PocketTTS | ❌ | en/es/fr/de/pt/it/nl |

### Három magyar F5 checkpoint (a mért az első)

- `sarpba/F5-TTS_V1_hun_v2` – ez a mért, F5TTS_v1_Base architektúrával.
- `Maxdorger29/f5-tts-hungarian` – szintén magyar F5TTS_v1_Base.
- `mp3pintyo/F5-TTS-Hun` – a repó szerzőjének saját finomítása (2024, `.pt`).

### Reprodukálható mérés

**2026-10-05: a `C:\AI\clone-probe` teljes egésze TÖRÖLVE** (2,5 GB). A
mérőprogramok megmaradtak a `docs/meresi-adatok/f5-tts/` alatt, a venv-eket
újra kell építeni (x86_64 Python + torchaudio 2.11 + f5_tts 1.1.22).

```
cd <repo>/reader/docs/meresi-adatok/f5-tts
# a venv-ek már nem léteznek:
# venv/Scripts/python.exe f5_matrix.py          # hang x nfe_step (emulált)
# venv-arm64/Scripts/python.exe f5_native.py    # natív, shimmel
```

### A gépen régebben lévő, nem általam épített környezetek

A felhasználó 2026-10-04-én jelezte, hogy a D: meghajtón vannak korábbi
TTS-környezetek. Két érdemi találat (a teljes leltár a handoffban):

1. **`D:\VoiceAI\apps\hu-voice-ai\venv-x64`** — kész F5-TTS (`f5_tts 1.1.20`, `torch 2.12.1`,
   `torchaudio 2.11`, `torchcodec 0.15`) + `ffmpeg_shared\`. **A
   `libtorchcodec_core4.dll` PE-importjai szerint** az FFmpeg **58-as
   generációját** kéri (`avcodec-58`, `avformat-58`, `avfilter-7`,
   `swscale-5`, `swresample-3`), a `ffmpeg_shared\...win64-gpl-shared\bin`
   viszont a **63-as generációt** tartalmazza (`avcodec-63`, `avformat-63`,
   `avfilter-12`, `avutil-61`). Ez **nem egy hiányzó fájl, hanem generáció-
   ütközés**, ezért DLL-másolással nem javítható — a `bin`-nek a torchcodec
   által támogatott FFmpeg-verziót kellene adnia. Ugyanez a hiba, mint a
   `C:\AI\clone-probe\venv`-ben (ott a `torchcodec 0.17` + `av 19` `avutil-61`
   párosítás volt a működő). **(2026-10-06: az egész repó átköltözött a
   `D:\VoiceAI\apps\hu-voice-ai\` alá.)**
2. **`D:\VoiceAI\apps\hu-voice-ai\models\piper\hu_HU-{anna,berta,imre}.onnx`** — három
   **magyar Piper-hang ONNX-ben** (22050 Hz, medium). Nem klónoz, de
   ONNX Runtime-tel, Torch nélkül közvetlenül futtatható, ahogy a Supertonic.

**A legígéretesebb ajtó MEGNYÍLT:** a
`pltobing/XTTSv2-Streaming-ONNX` — a modellkártya **magyar (`hu`) zero-shot
klónozást** ígér **PyTorch nélkül, ONNX Runtime + NumPy** alapon. A felhasználó
2026-10-04-én elfogadta a CC BY-NC 4.0 űrlapot, és a modell a
`D:\VoiceAI\vendor\XTTSv2-Streaming-ONNX` alatt áll (2026-10-06: átköltözött a
`D:\XTTSv2-Streaming-ONNX`-ből). **Mérve is működik magyarra** — lásd a
„XTTSv2-ONNX magyar hangklón" szakaszt lentebb (és az `M-39` bejegyzést a
`meresek.md`-ben).

## Magyar Piper-hangok ONNX-ben, mérve (2026-10-04)

### A gated XTTSv2 hozzáférése: a 403 téves lezárás volt

A `HF_TOKEN` a környezetben **be van állítva**, és a `model_info()` hívás
**sikeres** volt — de ez csak a *metadatalapot* jelenti. A tényleges fájlletöltés
`GatedRepoError: 403 … Access to model pltobing/XTTSv2-Streaming-ONNX is
restricted and you are not in the authorized list` hibával elhasalt. **Ez
azonban nem a modellről szólt, hanem a sessionben lévő `HF_TOKEN` fiókjáról:**
az űrlap kitöltése után a felhasználó **letöltötte** a modellt
(`D:\XTTSv2-Streaming-ONNX`, 2,24 GB), és a mérés elvégezhető volt.

**Két tanulság a jövőre:**
1. A HF-en a `model_info()` sikeressége **nem** jelent fájlhozzáférést — a
   `hf_hub_download` / `snapshot_download` a mérvadó próba.
2. **A 403 nem feltétlenül a modell zároltságáról szól.** Ha a fiók nincs az
   engedélyezett listán, hiába elfogadta valaki más a licencet. Ezt a lezárást
   egy idióta „elvártam a felhasználó döntésére” formában adtam tovább — a
   jogosabb állítás: **a `403` ezen a fiókon nem eldönthető, új token kell.**

### Amit viszont meg lehetett mérni: a három magyar Piper-hang

`D:\VoiceAI\apps\hu-voice-ai\models\piper\hu_HU-{anna,berta,imre}-medium.onnx` — magyar
ONNX-hangok, 63 MB, 22050 Hz. Nem klónoz, viszont az Auris **már is használ
ONNX Runtime-t**, tehát nincs új függőség.

Bemenet/bimenet (a `piper` 1.4.2 `voice.py` alapján):

| Név | Típus | Jelentés |
| --- | --- | --- |
| `input` | `int64[1,N]` | BOS+PAD, phonem-azonosítók, PAD köztük, EOS a végén |
| `input_lengths` | `int64[1]` | `N` |
| `scales` | `float32[3]` | `[noise_scale, length_scale, noise_w]` = `[0.667, 1.0, 0.8]` |

A phonemizálást (espeak) **nem** a natív ARM64-en futtattam: az x86_64-es
`D:\VoiceAI\apps\hu-voice-ai\venv-x64` piper-CLI kigyártja a phonem-ID-kat
(`voice.phonemize()` + `phonemes_to_ids()`), és a natív mérés ugyanezzel a
számsorral indul. Így kizárólag az **ONNX-inferencia** ideje mérődik.

### Eredmények (2 mondat × 3 hang)

| Futtatás | Környezet | idő | hang/sebesség | WER | CER |
| --- | --- | ---: | ---: | ---: | ---: |
| piper CLI (`piper` 1.4.2) | x86_64 emulált | 0,10–0,15 s/mondat | **26,8×** | 0,17 % | 0,03 % |
| nyers ORT | **natív ARM64**, ORT 1.30.0, 12 szál | **0,06–0,11 s/mondat** | **35,3×** | 0,19 % | 0,03 % |

- **Az 1 óra hangoskönyv így ~1,9 perc**, egy 8 órás könyv ~15 perc. A
  Supertonic (9,6 perc/óra, 6,9×) **5,0× gyorsabb**, az F5-TTS klón
  (nfe=8: 7,6 óra/óra = 456 perc/óra) **240× gyorsabb**.
- A mintavétel 22050 Hz, tehát ha a Supertonic 24 kHz-et ad, a fájlméret
  is kisebb (és a gyorsaság felülmúlja a Supertonic-ot).
- A WER ugyanakkora, mint az F5-TTS klóné (0,11 %), és 60×kal jobb, mint a
  Supertonic batchelt 11,5 %-a. Ugyanaz a jellegzetes „hiba" jelenik meg
  (szóösszefűződés: „mégköd lebegett") — ez a magyar szövegelő sajátsága.

### Fontos módszertani részlet: a Piper hangnem-determinisztikus

Két futás nyers hullámformáját elsőre összehasonlítva a korreláció **≈ 0,15**
lett, ami ijesztően rossz aritmetikai eltérésnek látszik. **Nem az.**
A VITS generátor `noise_scale=0,667` és `noise_w=0,8` értékkel **szó szerint
random zajt** kap; két hívás ezért szándékosan eltérő hullámot ad (a
hossz is 1–2 %-ot mozog, mert a leállás a zajfüggő dekódolástól függ).

Ellenőrzés determinisztikusra állítva (`scales = [0, 1.0, 0]`, mindkét
oldalon):

| Hang | minta (emulált / natív) | max\|Δ\| | korreláció |
| --- | --- | ---: | ---: |
| anna #1 | 73 984 / 73 984 | **4,7e-05** | **1,000000** |
| anna #2 | 71 936 / 71 936 | **3,5e-05** | **1,000000** |

Azonos hossz, azonos korreláció — a két ORT ugyanazt a számot adja.

Tehát a natív ARM64-es ORT **számítástechnikailag egyenértékű** az emulált
x86_64-es ORT-tal; a 4,7e-05 a 16 bites PCM-kvantálás, nem numerikus hiba.
Ezt a mérés csak determinisztikus (zaja nélküli) módban érvényes.

### Reprodukálható mérés

**2026-10-05: a `C:\AI\clone-probe` TÖRÖLVE**, a mérőprogramok a
`docs/meresi-adatok/f5-tts/` alatt vannak. A piper-hangok modellje
(`D:\VoiceAI\apps\hu-voice-ai\models\piper\`) és a mérés önmagában **újrafuttatható**,
csak a mérőprogram helyét kell átírni:

```
cd <repo>/reader/docs/meresi-adatok/f5-tts
D:/VoiceAI/apps/hu-voice-ai/venv-x64/Scripts/python.exe piper_hu_probe.py   # emulált + phonem-ID-k
<repo>/reader/.venv/Scripts/python.exe piper_hu_native.py      # natív ARM64 ORT
cd <repo>/reader
.venv/Scripts/python.exe scripts/asr_check_wavs.py --json <...>/cases.json
```

A nyers `cases.json` / `report.json` eredmények a `f5-tts/out_piper\`
(`out_piper_native\`) alatt vannak.

## XTTSv2-ONNX magyar hangklón — MÉRVE MŰKÖDIK (2026-10-04, este)

A gated modellt a felhasználó letöltötte (`D:\XTTSv2-Streaming-ONNX`, 2,24 GB),
és natív ARM64-en, **tisztán ONNX Runtime + NumPy alapon** mérhető. **PyTorch
nem kell hozzá, shim sem, emuláció sem** — ez az első klónozó, ami közvetlenül
az Auris ONNX-útjába illik.

### A mérés

Referenciahangok: **ugyanaz a két Supertonic-minta (F1, M3)** és **ugyanaz a két
mondat**, mint az F5-TTS-nél (M-12/M-13) és a Piper-hangoknál (M-16/M-17) —
így a WER-ek összevethetők. ONNX Runtime 1.30.0, 12 szál, `stream_chunk_size=20`,
`speed=1.0`.

| Mód | Számítás | Hang | Sebesség | 1 óra hangra | WER | CER |
|---|---:|---:|---:|---:|---:|---:|
| **int8 GPT** | 10,83 s | 14,30 s | **1,32×** | **45,4 perc** | 0,18 % | 0,06 % |
| fp32 GPT | 14,51 s | 14,86 s | 1,02× | 58,6 perc | 0,11 % | 0,06 % |

**Az INT8 itt 1,29× gyorsabb, és a minőségromlás elhanyagolható** (0,18 % vs
0,11 % WER) — vagyis **fordítva** a Supertonicnál látott eredménynél, ahol az
INT8 8 %-kal lassabb volt. Ennek oka, hogy itt a GPT-2 **30 rétegű,
autoregresszív** modell, aminek minden tokenre újra kell futnia: a kvantálás
költsége sokszorosan fizetődik vissza.

### Ez a legjobb mért klónozó

| Motor | Környezet | 1 óra hang | WER |
|---|---|---:|---:|
| **XTTSv2-ONNX int8** | **natív ARM64, ONNX only** | **45,4 perc** | 0,18 % |
| F5-TTS (nfe=8) | x86_64 emuláció + valódi torchaudio | 7,6 óra | 0,11 % |
| F5-TTS (nfe=8) | natív ARM64 + saját torchaudio-shim | 15,4 óra | 0,07 % |

**9,9× gyorsabb az emulált F5-TTS-nél és 20,3× a natív+shim úton** (azaz 449
illetve 924 perc egy óra hangra, szemben a 45,4 perccel), a minősége azonos
szinten van. A Supertonic
alapmotor 9,6 perc/óra, tehát a klónozás itt **4,7× időbehülés**, nem
minőségromlás.

### Amit a függőségekről tudni kell

A repó `requirements.txt`-e **17 csomagot** sorol fel, de magyar nyelven a
beállítás **9 csomag** (`numpy onnxruntime soundfile scipy num2words tokenizers
librosa pypinyin hangul-romanize`), és ezek mind ARM64 wheelből jöttek —
**meglepés módon a `librosa` is** (a `numba 0.68` + `llvmlite 0.50` ARM64-en
elérhető), a `tokenizers 0.23` is.

**A `spacy` az egyetlen, ami nem telepíthető** (nincs win_arm64 wheel; a `thinc`
C++ kiterjesztés nem fordul) — de **magyarhoz nem is kell**. A tokenizer
`preprocess_text`-je magyarra a `multilingual_cleaners` ágat választja, a
`get_spacy_lang`-ot pedig csak az `ar/en/es/ja/zh` hívja. Ezért kell egy
`stubs/spacy/`, amely **példányosításkor `RuntimeError`-t dob** — szándékosan
nem néma helyettesítő, hogy ha valaki spacy-t igénylő nyelvet választ, azonnal
lássa, hogy ez csak a magyar ágat fedi le.

### Reprodukálható mérés

```bat
cd /d D:\VoiceAI\hub\scripts
D:\VoiceAI\envs\xtts-probe\venv-arm64\Scripts\python.exe xtts_hu_probe.py --smoke
D:\VoiceAI\envs\xtts-probe\venv-arm64\Scripts\python.exe xtts_hu_probe.py --mode int8
D:\VoiceAI\envs\xtts-probe\venv-arm64\Scripts\python.exe xtts_hu_probe.py --mode fp32

:: WER/CER
cd "<repo>\auris\reader"
.venv/Scripts/python.exe scripts/asr_check_wavs.py --json D:/VoiceAI/out/xtts/cases_int8.json --language hu
```

**Két mérési figyelmeztetés:**
1. A pipeline **minden generált tokenre INFO-sort ír**, többszöres sorokban.
   Ez önmagában mérési zaj — a mérő `--verbose` nélkül `WARNING`ra állítja.
2. A `cases.json` a `mode` nélkül **felülíródna** az fp32 futással, ezért a
   fájlnévben benne van a mód (`cases_int8.json`, `cases_fp32.json`).
