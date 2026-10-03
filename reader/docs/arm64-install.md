# ARM64 (Snapdragon X) telepítés és mérés

Mérve: 2026-10-02, Windows 11 ARM64 (vivo2, Snapdragon X Elite), 31,6 GB RAM,
12 logikai mag, Python 3.12.10 ARM64. PyTorch 2.10.0+cpu (win_arm64 wheel),
ONNX Runtime 1.30.0 (win_arm64 wheel), **torchaudio nélkül**.

## Mi fut, és miért

| Motor | Állapot | Ok |
| --- | --- | --- |
| **Supertonic 3** | ✅ mérve | ONNX Runtime, arm64 wheel létezik — **ez a választott motor** |
| Piper | ⚠️ telepíthető WSL2-ben | `piper-tts==1.8.0`-nak nincs win_arm64 wheelje |
| OmniVoice | ❌ nem telepíthető | az `omnivoice` csomag importálja a torchaudio-t |
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
5. **A kvantáló eszközök is korlátozottak:** az ORT quantizációs segédegyei a
   dokumentáció szerint csak x86_64-on futnak (az `onnx` csomag ARM64-en nem
   telepíthető), tehát a statikus QDQ exportot itt nem lehetne előállítani.

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
