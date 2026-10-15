# Mérési jegyzőkönyv — mit mértünk, és mit *nem*

**Ez a fájl a bizonyítékkatalógus.** Az [arm64-install.md](arm64-install.md)
a *hogyan*-t dokumentálja, témánként (hogyan működik az ONNX→DLC, mi a
batchelés); a [handoff-2026-10-03.md](handoff-2026-10-03.md) az *útmutatót*
és a nyitott listát. Itt minden mérés egy azonos sémában szerepel, hogy ne
kelljen újrafelderíteni: **mi futott a gépen, mi csak dokumentáció, és mi az
elvetett hipotézis.**

Utolsó állapotellenőrzés: **2026-10-04** — az artifact-leltár (§5) ekkor
lett újraellenőrizve a lemezen, programszerűen.

## 0. Hogyan olvasd

### 0.1 Bizonyítéki szintek

Minden bejegyzés (M-xx) ezt a jelölést viseli. **A jelölés a szám erejét
adja meg, nem a témát.**

| Jelölés | Jelentés | Mire épül |
|---|---|---|
| ✅ **MÉRVE** | A gépen futott, számmal és megőrzött artifacttal | mérőprogram futtatása + kimeneti fájl/log |
| ⚠️ **MÉRVE, KIS MINTA** | Futott, de az `n` túl kicsi ahhoz, hogy a szám túlélje a szórást | mérés, de kis korpusz; a konkrét `n` mindig a bejegyzés mellett áll |
| 🔶 **KÖZVETLEN** | Nem futtattuk: fájl-, driver- vagy metaadatszemlélet | PE-importok, driver store, modellkártya, fájlméret |
| ⬜ **ELMÉLET** | Külső dokumentáció állítása, amit itt nem teszteltünk | issues, gyártói dokumentáció |
| ⏸️ **BLOKKOLT** | Nem futott, külső akadály miatt | licenc/hozzáférés/hiányzó komponens |

**Az elvetés nem külön szint, hanem a bejegyzés állapota:** egy motor akkor
„kizárt”, ha a hozzá tartozó bejegyzés ✅ (mérve kiderült, hogy nem jó —
lásd M-31) vagy ⬜ (csak a modellkártya mondja — lásd M-33). Így a kizárás
okmányolva marad, nem hangzik el.

### 0.2 Az alapelv

**Ne írj számot mérés nélkül.** Ha egy szám itt nem szerepel, vagy ⬜/🔶
jelölésű, akkor azt nem mértük ezen a gépen — hiábalható, de nem
bizonyítható belőle állítás. Ez a fájl azért létezik, mert a
2026-10-02→04 időszakban **többször ismétlődött**, hogy egy dokumentációs
állítást mérés cáfolt meg (ld. §4).

### 0.3 A mérés tárgyának neve

Az M-szám a témát, nem a pillanatnyi állapotot azonosítja. Ha egy mérés
újrafut, **ugyanazt az M-számot** kapja, és a §8 naplóba kerül.

## 1. A gép és a mérési környezetek

Rögzített tények — ezek nem mérési eredmények, hanem a mérés feltételei.

| Elem | Érték |
|---|---|
| Gép | Windows 11 ARM64, vivo2, Snapdragon X Elite (Hexagon v81), 12 logikai mag, 31,6 GB RAM |
| GPU/NPU | Adreno GPU-t egyik PyTorch sem használja; a Hexagon NPU illesztőprogramja hiányos (ld. M-26) |
| Repó venv | `auris/reader/.venv` — **natív ARM64** Python 3.12.10, ORT 1.30.0, numpy 2.5.3, soundfile 0.14.0, scipy 1.18.1 |
| Nincs a repó venvben | `torchaudio` (M-15), `wetext` (M-35), `transformers`, `tokenizers`, `librosa`, spaCy (M-35) |
| Emulált F5 venv | ~~`C:\AI\clone-probe\venv`~~ — x86_64 Python + valódi torchaudio 2.11.0, torch 2.14.1+cpu, f5_tts 1.1.22 — **2026-10-05: TÖRÖLVE** |
| Emulált piper venv | `D:\<projekt>\apps\hu-voice-ai\venv-x64` — x86_64, piper 1.4.2, ORT 1.27.0 |
| NPU venv | ~~`C:\AI\venvs\foundry-dev`~~ — ORT 1.29.0 + onnxruntime-qnn 2.5.0 (`win_arm64`) — **2026-10-05: TÖRÖLVE** |
| Hanghub | `D:\<projekt>\` — a modellek/motorok/hangok registryje és állapotellenőrzője (`doctor.py`) |
| ASR | `core/qa.py`: magyarra Parakeet TDT v3 (`nemo-parakeet-tdt-0.6b-v3`, `onnx_asr`) + Whisper (`sarpba/whisper-hu-large-v3-turbo-finetuned`) |

**Két megkötés, ami minden RTF-számra érvényes:**

1. **A gyorsítótárazás hamis 0-t ad.** Minden RTF-mérés előtt törölni kell a
   mérendő felvételt; a `bench_supertonic_variants.py` ezt asserttel védi.
2. **A Windows csúcsmemóriája monoton nő**, ezért variánsonként **külön
   processz** kell a memóriaméréshez.

## 2. A mérések

### 2.1 Supertonic 3 — az alapmotor

| ID | Állapot | Mit mértünk | Eredmény | Bizonyíték |
|---|---|---|---|---|
| **M-01** | ✅ | RTF egy 268 karakteres magyar bekezdésen | Férfi 0,243 / női 0,264 | `arm64-install.md` „RTF-mérés" |
| **M-02** | ✅ | Teljes könyv: 9 764 karakter, 9 blokk | 13,2 perc hang / **4,2 perc** valós; RTF **0,317** (min/med/max 0,291/0,310/0,347); 36,4 MB WAV; **756 MB peak**; betöltés 0,72 s | u.o. „Teljes könyvmérés" |
| **M-03** | ✅ | Batch 1 → 4 az export útvonalon, 8 szegmens | RTF 0,3235 → **0,1440** (2,25×), hang 54,5 → 54,3 s | u.o. „Batch" |
| **M-04** | ✅ | Ugyanez hangonként, 24 mondat | F1 0,3123→0,1758 (1,78×); M3 0,3316→0,1698 (1,96×) | u.o. |
| **M-05** | ⚠️ | Batch minőség: 48 futás (24 mondat × 2 hang) | WER 12,21 % → **11,52 %**; CER 1,3 % (azonos) | u.o. — **a különbség kb. 2–3 szó, nem szignifikáns** |
| **M-06** | ⚠️ | Lépésszám 4/6/10/16, 12 mondat | RTF 0,1850/0,2717/0,3916/0,5600; WER 9,5/7,4/**7,4**/8,4 % | u.o. „Lépésszám" — trend, nem igazolt különbség |
| **M-07** | ✅ | INT8 vs fp32 (szám, betöltés, memória, RTF) | modell 382,7→**138,1 MB**; betöltés 0,72→0,67 s; peak 494→**241 MB**; RTF 0,32→0,35 (**+8 % lassabb**) | u.o. „INT8 Supertonic 3" |
| **M-08** | ⚠️ | INT8 WER | 10,1 % → 6,5 % | u.o. — **kb. 1,6 szórás, tehát nem igazolt javulás** |
| **M-09** | ✅ | INT8 batchelt módban | 0,1638 vs fp32 0,1758 — **az INT8 ekkor előzi az fp32-ot** | `handoff` 2.4 |
| **M-10** | ✅ | Almodell-időeloszlás, egy mondat | **`vector_estimator` 2,177 s = 93 %** (10 hívás); vocoder 0,153 s = 7,5 %; text_encoder 0,021 s; duration_predictor 0,002 s | u.o. „Hol megy el az idő" |
| **M-11** | ✅ | `long_pause` hamis pozitív | 7 blokk jelezte; a valós leghosszabb belső csend 1,45–2,10 s, a `qa.py` **fix 1,5 s-os** küszöbe szólal meg | u.o. „Ismert zaj" |

### 2.2 Magyar hangklónozás — F5-TTS

> **LEZÁRT ÉS TISZTÍTVA (2026-10-05).** A motor kikerült a gépről, mert a
> legjobb mért futása is **449 perc/óra hang** (az XTTS int8 45,4), azaz
> 9,9× lassabb. A számok és a nyers adatok megmaradtak (§5); a környezet
> archív: `docs/meresi-adatok/f5-tts/`.

| ID | Állapot | Mit mértünk | Eredmény | Bizonyíték |
|---|---|---|---|---|
| **M-12** | ✅ (sebesség) ⚠️ (WER) | F5-TTS magyar, **x86_64 emuláció**, 12 futás, 2 hang × 3 nfe_step | hang_idő/idő: F1 0,0418/0,0615/**0,1336**; M3 0,0360/0,0734/**0,1562** (nfe 32/16/8) | archív: `docs/meresi-adatok/f5-tts/out_matrix/bench.json` + `matrix.log` |
| **M-13** | ✅ (sebesség) ⚠️ (WER) | Ugyanez **natív ARM64 + shimmel**, 3 futás | nfe=32: 212,26 s + 207,03 s (3,4 s hangra); nfe=8: 53,13 s | archív: `docs/meresi-adatok/f5-tts/out_native/cases.json` + `native.log` |
| **M-14** | ✅ | A torchaudio-shim felépítése és három valódi hibája | 6 pótolt szimbólum; a `transformers` `find_spec`-et hív (`__spec__` kell); a vocos nem a publikus API-t importálja (`torchaudio.functional.functional`); a vocos checkpoint `mel_spec.*.fb` / `.window` kulcsokat vár → `MelSpectrogram` torch.nn.Module | archív: `docs/meresi-adatok/f5-tts/` (a shim és a 21 teszt 2026-10-05: TÖRÖLVE) |
| **M-15** | ✅ | Van-e win_arm64 torchaudio wheel | **Nincs** — sem PyPI, sem a PyTorch CPU-index (a `uv` üzenete is csak `win_amd64`-ot sorol) | `arm64-install.md` „Fontos pontosítás" |

**A mért minőség (M-12): WER 0,11 %, CER 0,00 %** (Parakeet TDT v3, 2 mondat).
**A mért minőség (M-13): WER 0,07 %** (3 futás; a 4. timeout miatt nincs).

> **A WER-ek 2–3 mondatból származnak.** A 0,07–0,11 % minőségi jelzés, nem
> könyvszintű becslés. Ne idézz belőle „a F5-TTS hibátlanul szólal magyarul".

### 2.3 Magyar Piper-hangok (ONNX)

| ID | Állapot | Mit mértünk | Eredmény | Bizonyíték |
|---|---|---|---|---|
| **M-16** | ✅ (sebesség) ⚠️ (WER) | piper 1.4.2 CLI, **x86_64 emulált**, 3 hang × 2 mondat | 0,10–0,15 s/mondat → **26,8× valós idő**; WER 0,17 %, CER 0,03 % | `out_piper\report.json`, `out_piper\cases.json` |
| **M-17** | ✅ (sebesség) ⚠️ (WER) | Ugyanez **natív ARM64**, nyers ONNX Runtime 1.30.0, 12 szál, 12 futás | 0,06–0,11 s/mondat → **35,3×** (átlag a hányadosokon), **31,4×** (aggregált 18,2 s hang / 0,58 s); WER 0,19 %, CER 0,03 % | `out_piper_native\report.json`, `cases.json` |
| **M-18** | ✅ | **Numerikus konzisztencia** emulált ↔ natív, determinisztikus módban | `scales=[0,1,0]`: azonos minta-szám (73 984 / 73 984; 71 936 / 71 936), **max\|Δ\| 4,7e-05 és 3,5e-05**, korreláció **1,000000** | u.o. + `out_piper\anna_det_*.wav` |
| **M-19** | ✅ | A VITS hangnem-determinisztikussága | Zajos módban (`noise_scale=0,667`) két futás korrelációja **≈ 0,15**, a hossz 1–2 %-ot mozog — ez **nem** numerikus hiba, hanem a generátor random zaja | u.o. |
| **M-20** | ✅ | Könyvbecslés M-17-ből | **1 óra hang ≈ 1,9 perc**, 8 órás könyv ≈ 15 perc. Vs. Supertonic 9,6 perc/óra → **5,0× gyorsabb**; vs. F5-TTS nfe=8 (7,6 óra/óra) → **240× gyorsabb** | M-17 számai, arithmetikai extrapoláció |

**Bemeneti szerződés** (a `piper` 1.4.2 `voice.py` alapján): `input`
`int64[1,N]` (BOS+PAD, phonem-ID-k PAD-del, EOS a végén), `input_lengths`
`int64[1]`, `scales` `float32[3]` = `[noise_scale, length_scale, noise_w]`
= `[0.667, 1.0, 0.8]`.

> **A natív mérés tiszta ONNX-inferencia, nem szövegből indul.** A phonem-ID-kat
> az x86_64-es piper CLI gyártja, és ugyanazt a számsort adja az ORT-nek — így
> a mérésből a phonemizálás ideje **kimarad**. A teljes „szöveg → hang" út
> ezen a gépen tehát **egy lépésben nincs mérve**.

### 2.4 NPU / QNN

> **ÚJRA MEGNYITVA (2026-10-12) — a korábbi „LEZÁRT" verdikt téves
> architektúra-feltevésen alapult.** A Snapdragon X Elite HTP-je **V73**,
> nem V81. A V73 HTP driver-stack **megvan a gépen**
> (`DriverStore\...\qcnspmcdm8380.inf_arm64_*\HTP\QnnHtpV73StubDrv.dll`,
> `libQnnHtpV73SkelDrv.so`, `HtpUsrDrv.dll`, `QnnHtpPrepareDrv.dll`), és a
> `qnn-platform-validator` a V73 stubot **sikeresen betölti**. Az ONNX
> Runtime QNN EP **tényleg fut a HTP-n** (lásd M-47/M-48).
> A korábbi mérőkörnyezetek törlése érvényben marad, de az SDK-k
> (`C:\Qualcomm\AIStack`) és a Windows ML QNN EP csomagok megvannak.

| ID | Állapot | Mit mértünk | Eredmény | Bizonyíték |
|---|---|---|---|---|
| **M-21** | ✅ | ONNX→DLC konverzió, QAIRT 2.50 fp32 | `Conversion complete!`; DLC **5,3 MB**, BIN 256 MB, **3 815 004 809 MAC**, 63 923 520 paraméter, ~93 s | archív: `docs/meresi-adatok/npu-qnn/convert*.log` |
| **M-22** | ✅ | Ugyanez QAIRT 2.45 fp32 | DLC 5,3 MB, 3 815 004 809 MAC, 63 837 504 paraméter, ~90 s | `convert245.log` |
| **M-23** | ✅ | int8 kvantálás a konverterrel egy menetben | per-tensor: BIN **256 MB → 64,2 MB** | `convert_int8.log` |
| **M-24** | ✅ | Per-channel kvantálás | **elbukik**: `preprocessPerChannel: No bias info for op .../Conv_2d (Conv2d)` — a modellben bias nélküli konvolúció van | u.o. |
| **M-25** | ✅ | A `qairt-quantizer` visszaolvassa a 2.50-es DLC-t | **nem**: `Failed to deserialize the network` → a kvantálást a konverterrel kell egy menetben végezni | u.o. |
| **M-26** | ✅ | **A DLC futtatása a HTP-n — három úton, egyik sem** | (a) `qairt-net-run` 2.50 ARM64EC: `Function not supported by current QAIRT Version`; (b) `qnn-net-run` 2.45: `Initializing HtpProvider` **él**, de `Failed to create dlc handle with code 1002` — **még egy 15 KB-os apró DLC és a `--backend QnnCpu.dll` is elbukik**; (c) ORT QNN EP: **csendben CPU-ra esik** (lásd M-27) | `htp*.log`, `htp_check.py` |
| **M-27** | ✅ | Az ORT QNN EP tényleges végrehajtási helye | A profilban **0 db `QNNExecutionProvider` node** (1796 CPU-node); a kimenet **bitre azonos** a CPU fp32 kimenettel (`np.array_equal` → True); a „HTP 216 ms" egy szálas CPU-futtatás (tiszta CPU-referencia 219,5 ms) | `arm64-install.md` „NPU II. mérés" |
| **M-28** | ✅ | CPU-referencia a DLC-hoz (1×144×64 alak), 8 konfiguráció | fp32 1×1 szál 219,5 ms; fp32 1×12 szál 107,8 ms; fp32 4×12 szál 170,4 ms; int8 1×12 szál 108,8 ms (**az INT8 önmagában nem gyorsít**) | archív: `docs/meresi-adatok/npu-qnn/{npu_probe.py, cpu_run.log}` |
| **M-47** | ✅ | **A QNN EP TÉNYLEGESEN fut a Hexagon NPU-n (HTP V73)** — ONNX Runtime 1.31.0 + `onnxruntime-qnn` 2.6.0 (plugin-EP), Windows ARM64 | `htp_arch="73"` + plugin-EP regisztráció. A session létrejön **`session.disable_cpu_ep_fallback=1`** mellett (nem eshet CPU-ra), és az ORT-profilban a node-ok **mind `QNNExecutionProvider`**. A QNN HTP profil (`qnn_profile.csv`): **„Number of HVX threads used: 4"**, „QNN accelerator (execute) time" 1065–2587 µs, „Create device" OK, backend interface **2.39.0** (`HTP_QTI_AISW`, id 6), rendszer-interface 1.14. **Ez a korábbi „csendben CPU-ra esik" állítás cáfolata** | `D:\<projekt>\hub\scripts\npu_qnn_test.py`, `ort_profile_*.json`, `qnn_profile.csv` |
| **M-48** | ✅ | **CPU vs NPU GFLOP/s** — 4-rétegű 2048² MatMul-lánc (fp32 CPU 12 szál vs fp16 QNN HTP V73), 30 iteráció mediánja | **N=1024, L=6** (12,6 MFLOP): CPU 0,166 ms / 75,8 GF/s ↔ NPU 1,532 ms / 8,2 GF/s; **N=2048, L=4** (33,6 MFLOP): CPU 0,585 ms / 57,4 GF/s ↔ NPU 2,510 ms / 13,4 GF/s; **N=4096, L=3** (100,7 MFLOP): CPU 2,219 ms / 45,4 GF/s ↔ NPU 4,740 ms / 21,2 GF/s. **Az NPU sűrű fp16 MatMul-on lassabb a 12 szálas CPU-nál** (NPU/CPU 0,108 → 0,233 → 0,468, a mérettel javulva); az RPC-overhead kicsi modelleknél dominál. Az NPU előnye kvantált, konvolúció/figyelem-nehéz modelleknél várható | `D:\<projekt>\hub\scripts\npu_bench.py`, `npu_bench.json` |

> **M-28 szórása:** a `batch=1` sorok 2–3 %-on belül stabilak, a
> **`batch=4`, 12 szálos sorok 10–20 %-ot ugranak** (170,4–182,8 ms;
> median 211,7–262,9 ms) — ott az ORT szálütemezés és az arena-viselkedés
> dominál. NPU-összehasonlításhoz mindig azonos gépen, azonos `min`
> módszerrel kell mérni. **A küszöb, amit egy NPU-mérésnek meg kell haladnia:
> a CPU batch=4, 12 szálon 170–183 ms**, különben a DSP→host átvitel önmagában
> elfogyasztja a nyereséget.

**Gyökérok (🔶 KÖZVETLEN — fájl- és driverszemlélet):** a `QnnHtpV81Stub.dll`
statikus importja a `libcdsprpc.dll`, ami a driver store-ban
(`qcadsprpc8380.inf_arm64_*`) **van, de nincs a betöltési útvonalon** —
átmásolva `LoadLibrary` szinten betöltődik. A nagyobb akadály: az ORT QNN EP a
**HTP user-driver (HNRD)** úton jár, aminek **V81 kliensfájljai
(`QnnHtpV81StubDrv.dll`, `libQnnHtpV81SkelDrv.so`) sehol nincsenek** a
gépen; a driver store-ban csak V73-asok vannak.

### 2.5 Kizárások és elérhetetlenségek

| ID | Állapot | Tárgy | Megállapítás |
|---|---|---|---|
| **M-29** | ✅ | `pltobing/XTTSv2-Streaming-ONNX` (magyar zero-shot klón, tiszta ONNX Runtime) | **MEGMÉRVE — a felhasználó 2026-10-04-én elfogadta a CC BY-NC 4.0 űrlapot és letöltötte `D:\XTTSv2-Streaming-ONNX`-re (2,24 GB).** A korábbi `403` csak a session `HF_TOKEN`-jére vonatkozott, nem a modellre. Részletek: M-39. |
| **M-39** | ✅ (sebesség) ⚠️ (WER) | XTTSv2-ONNX **magyar hangklón, natív ARM64**, ONNX Runtime 1.30.0, 12 szál, ONNX-only (nincs PyTorch) | **int8: 1,32× valós idő** (10,83 s számítás / 14,30 s hang) → **1 óra hang ≈ 45,4 perc**; **fp32: 1,02×** → 58,6 perc/óra. WER **0,18 %** (int8) és **0,11 %** (fp32), CER 0,06 % |
| **M-40** | ✅ | A gated repó függőségei magyar nyelven | A 17 csomagból a magyar úthoz **9 kell**. A `spacy` **nem telepíthető** ARM64-en (nincs wheel), de a `preprocess_text` magyarra a `multilingual_cleaners` ágat választja, `get_spacy_lang`-ot csak az `ar/en/es/ja/zh` hívja → **a spaCy magyarhoz felesleges**. `stubs/spacy/` kell, ami példányosításkor `RuntimeError`-t dob (nem néma helyettesítő) |
| **M-41** | ✅ | **sherpa-onnx Supertonic 3 int8** (csukuangfj2 csomag: teljes szöveg-előfeldolgozás ONNX-ben), natív ARM64, 4 szál | **RTF 0,1497 = 400,7 perc hang/óra** (2 mondat, 1,19 s számítás / 7,92 s hang, **44,1 kHz**) — az Auris serial úthoz (0,7624) képest **5,1×**, a batch4-hez (0,3348) **2,2×** gyorsabb. Minőség: az M-39-protokoll ASR-checkerje **WER 0,00 % / CER 0,00 %** a 2 generált mondaton. Mérő: `D:\<projekt>\hub\scripts\sherpa_supertonic_probe.py` |
| **M-42** | ✅ | **Silero VAD QA-integráció** (`core/qa.py` `vad_speech_spans`, opcionális sherpa-onnx függőséggel) | **2,0 s mesterséges szünet → 1,97 s** (long_pause OK); az `analyze_audio` VAD-útja RMS-alapúval konzisztens F1-en (0,54/0,55, 0,51/0,55 s). Ugyanebben a körben: whisper-small int8/fp32 magyar **WER 44–89 %** (auto-detekt angolul hallucinál; `language="hu"`-val közel-misszek, CER ~16 %) → **WER-munkára nem használható**, a jelenlegi faster-whisper út marad. Tesztek: `tests/test_qa_vad.py` (5 teszt, reader-venvben 3 fut/2 skip, sherpa-venvben 5/5) |
| **M-43** | ✅ | **sherpa-onnx Supertonic 3 int8 — TELJES mérés a 24 mondatos hu_wer korpuszon** (M-41 kiegészítése, 2→24 mondat), natív ARM64, 4 szál | **serial RTF 0,1401 = 428,4 perc hang/óra; batch-út RTF 0,1419 = 422,8 perc/óra** (betöltés 0,7 s, 44,1 kHz). Az Auris serial (0,7624) → **5,4×**, az Auris batch4 (0,3348) → **2,4×** gyorsabb. Minőség (Parakeet, M-39 protokoll): **WER 0,11 % / CER 0,03 %** (24 mondat). Megjegyzés: a sherpa-onnx `OfflineTts.generate` csak egy szöveget fogad → **nincs valódi batch-nyereség** (a 0,1401 vs 0,1419 zajszintű; a „batch" itt újrahasznosított threadpool). Legrosszabb: double-name 0,50 %, bigyear 0,25 % (név/szám, nem motorhiba). Mérő: `D:\<projekt>\hub\scripts\sherpa_supertonic_full.py`, kimenet `D:\<projekt>\out\sherpa-supertonic-full\` |
| **M-44** | 🔶 **KÖZVETLEN** | **Nem-Whisper magyar ASR jelölt-landskép** — a `core/qa.py` már most is a **Parakeet TDT v3** (`nemo-parakeet-tdt-0.6b-v3`, onnx_asr, magyar a 25 európai nyelv közt — a sherpa-onnx docs megerősíti) használja CPU jelöltként, **WER 0,11 % / CER 0,00 %** (M-12, integrált). A többi jelölt a HF-en: Qwen3-ASR-0.6B/1.7B, jonatasgrosman/wav2vec2-large-xlsr-53-hungarian (magyar-specifikus), Cseti/Qwen3-ASR_Hungarian_v1 — **mindhattő transformers+torchot kéri, ami ARM64-en nem fut** (M-35); nemotron-3.5-asr-streaming, mms-1b-all, VibeVoice-ASR — nem mérve. **Következtetés:** nincs új, mért nummerő nem-Whisper magyar jelölt a Parakeet-hez képest; a Parakeet a kiválasztás. |
| **M-45** | 🔶 **KÖZVETLEN** | **Sherpa-Supertonic klón-korlát** — a sherpa-onnx `OfflineTtsSupertonicModelConfig` **csak a `voice_style` (voice.bin) mezőt tárta**, **nincs speaker-encoder / referenciaklón API**. A voice.bin fejléc `int32(10,0,50,0)` (10 beépített style, 50×256-as style_ttl). A README: **„open-weight fixed-voice setting"** → az open-weight Supertonic 3 **nem támogatja** a custom referenciahang-klónozást; a klónozó Voice Builder szolgáltatás **megszűnt** (M-34). Következtetés: a sherpa-Supertonic csak a beépített hangokon fut (F1/M3/F3...), a custom hangklón nem ez az út. |
| **M-49** | ✅ | **Minőségellenőrzés a valódi generált hangoskönyvön** („Én és te", minta-klon, XTTSv2-ONNX int8) | 3. fejezet (965,1 s): **CER 2,73 % / WER 11,94 %** (Parakeet TDT v3, magyar), flag `long_pause` (1,55 s), RMS −16,2 dB, peak 0,0 dB, ASR **RTF 0,070** (180 s-os ablakokban, mert a Parakeet ONNX encoder egyben nem bírja a 16 perces fejezetet). A QA 15 unit tesztje **mind OK** (a VAD-tesztek is, mert a `sherpa_onnx` telepítve van) | `D:\<projekt>\hub\scripts\qa_real_audiobook.py`, `qa_report.json` |
| **M-50** | ✅ | **Három független ASR ugyanazon a generált fejezeten** — Parakeet TDT v3 vs Whisper Small vs Whisper-HU finemodell (`04_main-4`, az első 120 s, azonos forrásszöveg) | **Parakeet: CER 5,34 % / WER 15,57 %, RTF 0,099** ↔ **Whisper Small: CER 4,83 % / WER 24,53 %, RTF 1,47** ↔ **`sarpba/whisper-hu-large-v3-turbo-finetuned`: CER 3,14 % / WER 8,49 %, RTF 2,20**. Vagyis a **magyar finemodell a legpontosabb** (a Parakeetnél ~1,8× jobb WER), de ~20×-el lassabb — ezért a `hybrid` a jó felépítés: Parakeet elöl, Whisper csak a gyanús mondatokra. A 10 leggyakoribb hiányzó szóból **8 közös** Parakeet és Whisper Small közt (`sír`, `olyasmiért`, `egó`, `valószínűleg`, `tálcán`, `megad`, `ismeszél`, `vajban`) → **a hiba a felolvasásban van, nem az ASR-ben**. Javítás: a `core/qa.py` Whisper-útja 30 s feletti hangnál `ValueError`-ral bukott (transformers 5.x long-form) — `return_timestamps=True`-ra javítva, a `words` csak `word_timestamps` kérésre töltődik; `asr_backend` `whisper` → `hybrid` | `auris/reader/scripts/asr_backend_compare.py`, `asr_compare_3way.json` |
| **M-51** | ✅ | **Teljes könyv-minőségellenőrzés — mind a 15 generált fejezet** („Én és te”, minta-klon, XTTSv2-ONNX int8, Parakeet TDT v3-as elő-felismerés 180 s-os ablakokban) | **503,0 perc hang** (30 180,8 s) végighallgatva, egyetlen `qa_report.json`-ba írva: **átlag CER 3,39 % / WER 12,85 %** (WER 10,43–15,15 %, CER 2,71–4,21 % — szűk eloszlás, **nincs kiugró/hibás fejezet**). **Az egyetlen jelzés minden fejezeten a `long_pause`** (2,73–3,57 s) — ez az ismert téves riasztás (1,5 s-os rögzített küszöb, a normál mondatközi szünet 1,5–2,1 s); a clipping, a csendes-fejezet és a tempó (`too_fast`/`too_slow`) ellenőrzés **egyik sem jelzett egy fejezeten sem**. Az ASR-összes idő 1748,9 s → **ASR-RTF 0,0579** (17,3× gyorsabb a valós időnél), tehát a teljes könyv hangjának ellenőrzése ~29 percet vett igénybe | `D:\<projekt>\hub\scripts\qa_real_audiobook.py`, `qa_report.json` (15/15 fejezet, `status: warn` = csak long_pause) |
| **M-52** | ✅ | **Élő hibrid QC-lábat valódi hangon, a gyártási kódfutáson** — `qa_hybrid_probe.py`: `asr_backend_for("hu")` → `hybrid`, `Transcriber.for_language("hu")` → `HybridTranscriber(confirms=True)`, majd a M-50-es protokoll 120 s-os szeletén végigfut a `transcribe` → `confirm` ketteőjük | **A gyanús szegmens tényleg megerősítésre kerül:** Parakeet CER 5,34 % / WER 15,57 % (17,4 s, RTF 0,145) → Whisper `confirm` CER **3,14 % / WER 8,49 %** (174,4 s, RTF 1,453) → a jobb olvasat számít, **final CER 3,14 % / WER 8,49 %**. Vagyis a `core/qa_api._evaluate` hibrid útja valódi hangon **csökkenti a WER-t 15,57 %-ról 8,49 %-ra** (−46 %). Korábban ez csak mock teszttel volt fedve; a `qa_real_audiobook.py` csak a Parakeet-előt futtatta | `D:\<projekt>\hub\scripts\qa_hybrid_probe.py`, `hybrid_probe.log` |
| **M-53** | ✅ | **Minőség–sebesség egyensúly a választható ASR-módokon, + élő módszerváltás a kész alkalmazásban** (`scripts/asr_balance_bench.py`, 4×60 s a 00/04/08/13 fejezet elejéről, azonos hivatkozószöveggel, M-50 protokoll) | **Élő kapcsolás a valódi settings fájlban** (a Beállítások legördülő ugyanezt írja): `auto`→`hybrid` (`HybridTranscriber`), `parakeet`→`ParakeetTranscriber`, `whisper`→`Transcriber`, `hybrid`→`HybridTranscriber`; az eredeti érték visszaállítva. **Mérés:** betöltés parakeet 8,2 s / whisper 31,4 s; szegmens-idő parakeet 3,6–3,9 s vs whisper 74–77 s. **Parakeet (és hibrid): WER 18,73 % / CER 6,46 %, RTF 0,062 → a teljes könyv ellenőrzése 31,6 perc. Whisper: WER 13,06 % / CER 6,05 %, RTF 1,266 → 637 perc (10,6 óra).** A 0,08-as megerősítő küszöb ezen a mintán 0/4 szegmenset ért el → a hibrid itt parakeet-sebességet ad, és csak a tényleg rossz szegmensnél fizet Whisper-költséget (ott mérve: WER 15,57 % → 8,49 %, M-52). **Következtetés: a hibrid az egyensúly** — a whisper-pontosság ~20×-es időáron. Új unit teszt: `TranscriberSwitchTest` (a módváltás osztályváltást ad, a visszaváltás nem tölt újra) | `auris/reader/scripts/asr_balance_bench.py`, `asr_balance_bench.json` |
| **M-54** | ✅ | **Küszöb-sweep: hol húzzuk a hibrid megerősítést?** — mondat-szintű mérés 3 valódi fejezeten (4/7/11), 60 mondat, mindegyik Parakeet-tel ÉS Whisperrel feliratkoztatva; a mondatok pontos hangidőtartamát az `alignment.align_words` adja | **Váratlan fő eredmény: mondatonként a Whisper ROSSZABB, mint a Parakeet** (WER 34,18 % vs 10,58 %) — a 30 s-os ablakpadlás + a rövid magyar klipen való hallucináció miatt. Megerősítőként is csak a trigger-ek 38–73 %-a javít, a többi pazarlás. Sweep (küszöb → megerősítés %, WER, teljes könyv perc): 0,04 → 26,7 %, 9,82 %, 397; 0,06 → 18,3 %, 9,86 %, 285; **0,08 (gyártási alap) → 13,3 %, 10,35 %, 212**; 0,12 → 5,0 %, 10,58 %, 98. Parakeet önmagában: 10,58 %, 33 perc. **Következtetés: a küszöböt NE húzzuk lejjebb** — a legjobb minőségnyereség is csak −0,76 pp WER 3–12×-es időáron; a 0,08 jó biztonsági háló, a sebesség felé a 0,12 sem veszített minőségből (60 mintán). A küszöb a felületen állítható (`qa_cer_warn` = „Gyanús, ha a hibaarány eléri (%)” → `confirm_above`). **Közben kijavítva egy valódi gyártási hibát:** az `align_words` prefix-keresése globálisan keresett, így egy illesztetlen szót a fejezet későbbi elhangzására tett (a fejezet eleji „sír” → 139,68 s), a monoton kitöltés pedig a teljes hátra lévő idővonalat 140 s-ra torlódta (minden „mondat” 140 s-os lett, a túlhosszú szelettől pedig az ONNX encoder is elbukott: broadcast 3083×8083). A keresés most a szó saját hiányos szakaszában marad (korábbi és következő illesztett szó hallott-indexe közt); regessziós teszt: `test_prefix_match_stays_inside_its_own_gap` | `auris/reader/scripts/asr_threshold_sweep.py`, `asr_threshold_sweep.json` |
| **M-46** | ✅ | **Sherpa-onnx Supertonic beillesztés a friss-klónbali** (repó-javítás) | **Eredeti válasz: NEM volt teljesen futatható a friss-klón.** Javítva: (1) `sherpa-onnx` a `ARM64_BASE_DEPS` + `requirements.txt` (win_arm64 wheel ✓, linux_aarch64 ✓ — pipelnövelt); (2) `SherpaSupertonicEngine` beillesztett a motor-választóba (`tts_router.ENGINE_NAMES` + `_create`, `hardware.ENGINE_ORDER_CPU` e LENN a leggyorsabb az ARM64-nál, `ENGINE_INFO`, UI-select a `settings.html/js`); (3) modellletöltés a HF-en `reader/models/supertonic-sherpa-int8/`-re (~139 MB, egyszer); (4) a Docker compose **CPU-ra** (`TORCH_VARIANT=cpu`, nincs a NVIDIA-gpu deploy). **Mérve a gép:** `sherpa_onnx 1.13.8` natív ARM64, modell-letöltés OK, teljes `generate` OK (6,37 s hang, 24 kHz, betöltés 0,5 s). Tesztek: **640, 0 bukás** (a 2 error a ismert torchaudio/wetext ARM64 baseline). `_usables` a `sherpa_onnx` hiányot felbontja → fallback a Supertonic-ra. |
| **M-30** | 🔶 **KÖZVETLEN** | `D:\hu-voice-ai\venv-x64` torchcodec hibája | **Nem hiányzó fájl, hanem generáció-ütközés:** a `libtorchcodec_core4.dll` PE-importjai az FFmpeg **58-as** generációját kérik (`avcodec-58`, `avformat-58`, `avfilter-7`, `swscale-5`, `swresample-3`), a `ffmpeg_shared\...win64-gpl-shared\bin` a **63-ast** tartalmazza. DLL-másolással nem javítható. |
| **M-31** | ✅ | MOSS-TTS-Nano — a legfontosabb negatív eredmény | **Kivezetve**: a codec-zaj valódi hiba volt (268 karakteres bekezdés 57,04 s → 28,07 s, a −72…−78 dBFS-es zajszőnyeg nullára, 2,2× gyorsabb, 5 futásban teljesen determinisztikus), **de a magyar kimenet érthetetlen maradt**. Egyetlen objektív metrika sem tudta kimutatni — ezt a tanulság. |
| **M-32** | ✅ | Miért érthetetlen a magyarja | 1,7× lassabb magyarra, mint angolra (0,091–0,13 vs 0,055–0,06 s/karakter); a spektrális flatness kétszeresére romlik (0,063 vs 0,030); a kimenet spektrálisan **beszédnek** néz ki (flatness 0,022, energia a 300–3000 Hz-es sávban), tehát nem zaj. A modell 0,1B paraméter — a kiejtés a modellben van, nem a referenciahangban. |
| **M-33** | ⬜ **ELMÉLET** | Nyelvi kizárások (modellkártya alapján, **nem mérve** ezen a gépen) | Chatterbox Multilingual v3 (23 nyelv, magyar nincs köztük); Sopro 169M (en/pt/fr/de); IndexTTS/2/2.5 (en/zh/ja/es/ar); PocketTTS (en/es/fr/de/pt/it/nl); ZipVoice / sherpa-onnx (nincs magyar) |
| **M-34** | ⬜ **ELMÉLET** | Supertonic Voice Builder | A szolgáltatás megszűnt; az open-weight modell nem támogat klónozást |
| **M-35** | ✅ | ARM64 wheel-hiányok, ténylegesen telepítve hiányzó csomagok | `torchaudio` (M-15), `wetext` (kaldifst, nincs arm64 wheel), spaCy/thinc (a thinc 9.1 ledobta az aarch64 wheeleket), `piper-tts` 1.8.0, `soxr` |
| **M-36** | ✅ | Ami **elérhető** ARM64-en | `onnxruntime-qnn` 2.5.0 (`win_arm64`), `onnx` 1.23.1 (`cp312-abi3-win_arm64`), `onnxruntime` 1.30.0, `torch` 2.10.0+cpu |
| **M-37** | ✅ | A QAIRT konverterek Python-követelményei | A natív `arm64x` build **nem tölt be ARM64 Pythonba** (ARM64EC binárisok); a `windows-x86_64` build emulációval működik. A 2.50-höz `onnx==1.17.0` kell (1.23-ban megszűnt az `onnx.version.version`), a 2.45 csak 3.6/3.8-as modult szállít |
| **M-38** | 🔶 **KÖZVETLEN** | A HF-kártyán említett, de nem mért F5 checkpointok | `Maxdorger29/f5-tts-hungarian` (672 MB, `model_last_final.safetensors`) és `mp3pintyo/F5-TTS-Hun` (`model_122000-hun.pt`, 1349 MB) — **egyik sincs kimérve** |

### 2.6 A hangklónozás összehasonlítása (minden mérve)

| Motor | Környezet | Sebesség | 1 óra hang | WER |
|---|---|---:|---:|---:|
| **XTTSv2-ONNX int8** | **natív ARM64, ONNX only** | **1,32×** | **45,4 perc** | 0,18 % |
| XTTSv2-ONNX fp32 | natív ARM64, ONNX only | 1,02× | 58,6 perc | 0,11 % |
| F5-TTS (nfe=8) | x86_64 **emuláció** + valódi torchaudio | 0,13× | 7,6 óra | 0,11 % |
| F5-TTS (nfe=8) | natív ARM64 + saját torchaudio-shim | 0,065× | 15,4 óra | 0,07 % |

**A mért klónozók közül az XTTSv2-ONNX a legjobb, két okból:**
1. **A legjobb minőség-arányú:** WER 0,18 %, az F5-TTS 0,11 %-áéval egy szinten,
   de nem néma helyettesítő — mindkettő magyar.
2. **10–20× gyorsabb az F5-TTS-nél** (9,9× az emulált, 20,3× a natív+shim úton —
   449 és 924 perc/óra hang, szemben a 45,4 perccel), és ellentéten az F5-TTS-sel
   **nem kell hozzá sem emuláció, sem shim** — a repo ONNX Runtime + NumPy.

A Supertonic (nem klónoz, de az alapmotor) 9,6 perc/óra, tehát az XTTS
4,7×-kal lassabb nála; a Piper-hangok (nem klónoznak) 1,9 perc/óra, 24×-kal
gyorsabbak. **A klónozás ára itt 4,7× idő**, nem minőségromlás.

### 2.7 Iparági kutatás — NPU és QC-ASR (2026-10-12, internet)

> **Mit változtat ez a korábbi verdikten.** A „zárt NPU-út, HNRD V81 driverre
> várunk” állítás **két okból is elavult**: (1) az X Elite HTP-je V73, és a
> driver-stack a gépen van (M-47); (2) **a Qualcomm 2025. decemberében
> publikus Hexagon NPU drivert adott ki**. Az NPU-út nincs szunnyadó —
> mérhető.

| Tárgy | Megállapítás | Forrás |
|---|---|---|
| **Publikus NPU-driver** | Qualcomm **Hexagon NPU Driver 1.0.0.12 Public Release** (2025-12-22), Snapdragon X sorozathoz, a Qualcomm Software Centerből. A 2024-es útmutató korábban a QPM-es, előnézeti 1.0.0.10-et írta le. **Nem keverendő** a Windows ML QNN EP minimális eszközillesztőjével (`30.0.140.0`) — a gépen a NPU-driver **30.0.220.3000** (2026-01-08), tehát ez a küszöb **teljesül** | [qualcomm.com/developer/blog/2025/12/hexagon-npu-driver-update-snapdragon-pcs](https://www.qualcomm.com/developer/blog/2025/12/hexagon-npu-driver-update-snapdragon-pcs); [támogatott EP-k](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/supported-execution-providers) |
| **Windows ML** | **GA 2025-09-23**; ONNX Runtime-alapú, ARM64-en is támogatott, a QNN EP-et NPU-ként sorolja fel. EP-k regisztrálása/enumereálása: `ExecutionProviderCatalog.GetDefault()` + `RegisterCertifiedAsync()`. A `dotnet add package Microsoft.WindowsAppSDK.ML`. A **Speech Recognition API (Whisper)** viszont a 2026-os doksi szerint **kísérleti**, C++/Python-only | [Windows ML GA](https://blogs.windows.com/windowsdeveloper/2025/09/23/windows-ml-is-generally-available-empowering-developers-to-scale-local-ai-across-windows-devices/); [EP választás](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/select-execution-providers) |
| **QNN EP frissítés Windows Update-en** | A **KB5096135** (2026-05) QNN EP-javításokat hoz Windows 11 24H2/25H2-re | [KB5096135](https://support.microsoft.com/en-us/servicing/os/windows/ai-components/2026/05/kb5096135-qualcomm-qnn-execution-provider-update-2-2605-2-0) |
| **DirectML kerülőút** | Az `onnxruntime-directml` wheeljei **csak `win_amd64`** — natív ARM64 Pythonhoz **nincs** (ORT issue #27123). x64-emulációs Pythonból az **Adreno GPU** gyorsítható, de az **nem NPU**. A Microsoft a DirectML-t „sustained engineering”-ként jelöli, új Windows-appokhoz a Windows MLt ajánlja | [PyPI onnxruntime-directml](https://pypi.org/project/onnxruntime-directml/); [ORT install](https://onnxruntime.ai/docs/install/) |
| **A csendes CPU-visszaesés ellen** | A hivatalos recept: `session.disable_cpu_ep_fallback="1"` — ha a modell nem vihető HTP-re, **hibát dob**, nem fut észrevétlenül máshol. A `get_available_providers()` csak a csomagot igazolja, nem a tényleges HTP-futtatást; azt a **profil** (node `provider` + QNN backend CSV) bizonyítja | [QNN EP docs](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html) |
| **A QNN EP elérhetőségi hibái** | #26163: `QNN_COMMON_ERROR_INCOMPATIBLE_BINARIES` (kevert QNN-verziók); #22617: a QNN EP nem jelenik meg. 2025–26-ban **nincs olyan javítás**, ami minden EP/driver/modell kombót megoldana | [issue #26163](https://github.com/microsoft/onnxruntime/issues/26163), [issue #22617](https://github.com/microsoft/onnxruntime/issues/22617) |
| **Whisper a QNN-en** | Egy 2025-ös ORT-vita szerint X Elite-en a **Whisper Tiny/Base QNN-en futott**, a **Medium/Large gráfinalizálási hibát** adott. A Qualcomm AI Hub `whisper_windows_py` mintája is **Whisper-Base**-re épül, nem Medium/Large-ra | [ORT discussion #24531](https://github.com/microsoft/onnxruntime/discussions/24531); [AI Hub whisper_windows_py](https://aihub.qualcomm.com/apps/whisper_windows_py) |
| **Whisper ONNX repók** | `onnx-community/whisper-{base,small,medium,large-v3}-ONNX`: fp32/fp16/**int8**/q4/q4f16/bnb4 + cache-es és merged decoder-variánsok (pl. small int8 encoder+decoder ≈ 406 MB; large-v3 fp32 `encoder_model.onnx_data`-t is használ). **„glonnx” nem létezik** — a fájlok `.onnx`-ek | [whisper-small-ONNX](https://huggingface.co/onnx-community/whisper-small-ONNX/tree/main/onnx) |
| **Intel INT4 Whisper** | A `Intel/whisper-large-onnx-int4` valódi (INT4, ~1,9 GB, Apache-2.0, `MatMulFpQ4`, ORT≥1.16), de **Intel AVX512 CPU-ra** készült. A `whisper-large-onnx-int4-inc` pontos repóneve **401-et** adott, nem ellenőrizhető | [optimized-whisper-intel](https://github.com/egorsmkv/optimized-whisper-intel) |
| **Parakeet/Canary nyelvtámogatás** | `nemo-parakeet-tdt-0.6b-v3` és `canary-1b-v2` = **25 európai nyelv, köztük a magyar**. A „Parakeet csak angol”, „Canary csak 4 nyelv” állítás **elavult** (a v2/v3-ra nézve). Az `onnx-asr` ezeket és a Whispert is támogatja | [onnx-asr](https://github.com/istupakov/onnx-asr); [sherpa Parakeet v3 nyelvek](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/nemo-transducer-models.html) |
| **Valós NPU-mérés hang-ASR-n** | **OpenWritr** (trsdn/openwritr-windows): Snapdragon X Elite-en, Parakeet — teljes dekódolás 3 s hangra **128 ms (23×)**, 23 s-ra **626 ms (37×)**; **a Parakeet encoder HTP-n fut**, a mel-előfeldolgozás és a TDT decoder CPU-n → részleges offload. A projekt Whisper-utat is implementált. Projektgazda-mérés, nem független labor | [openwritr-windows](https://github.com/trsdn/openwritr-windows) |

**Amit a kutatás NEM igazolt:** hogy bármilyen Whisper ONNX *automatikusan* a
Hexagon NPU-n futna; hogy a DirectML EP Pythonból a Hexagonhoz rendelne; hogy
a Whisper Medium/Large HTP-n menne. **Amit viszont igen:** a QNN EP a mi
gépünkön fut (M-47), a publikus NPU-driver kiadva, a Windows ML GA, és van
független, X Elite-es Parakeet-NPU mérés.

## 3. Amit NEM mértünk — a nyitott mérési lista

Ez a legfontosabb szakasz. **Ezekre nincs számunk**, és senki ne
hivatkozzon rá, mintha lenne.

| # | Kérdés | Miért nincs válasz | Mi kellene |
|---|---|---|---|
| 1 | **Az NPU gyorsítása valódi hang-modelleken** | A QNN EP **már fut a HTP V73-on** (M-47), és van sűrű-MatMul mérés (M-48) — de az Auris TTS/ASR modelljeit (XTTS, Supertonic, Parakeet) **nem próbáltuk HTP-re kvantálni/futtatni**. Az M-48 szerint sűrű fp16 MatMul-on az NPU lassabb a CPU-nál, tehát a gyorsulás kvantált int8/konvolúciós modelltől függ | XTTS/Supertonic/Parakeet QDQ-kvantálás + QNN EP-futtatás és mérés |
| 2 | Az XTTSv2-ONNS **könyvszintű** viselkedése | Csak 2 mondat mérve (M-39), hosszú szöveg, dialógusváltás, számozott lista nincs | Ugyanaz a mérőprogram bővebb korpusszal |
| 2b | Az XTTSv2-ONNX **stream_chunk_size** és **speed** optimalizálása | Csak az alapértelmezett `chunk=20`, `speed=1.0` futott | `--chunk` és `--speed` sweep a mérőprogrammal |
| 3 | A másik két magyar F5 checkpoint minősége/sebessége | M-38: le sem töltve | `f5_matrix.py` átállítása a másik repo-checkpointra |
| 4 | **A szöveg→phonem lépés** ARM64-en | A Piper bemenete phonem-ID, nem szöveg; az Auris szövegelője `wetext`, ami ARM64-en nincs (M-35) | Magyar g2p ARM64-en, **vagy** a phonemizálás egy x86_64 segédprocesszbe |
| 5 | A Piper **könyvszintű** viselkedése | Csak 2 mondat mérve: hosszú szöveg, számozott lista, dialógusváltás, csendkezelés | Ugyanaz a mérőprogram bővebb korpusszal |
| 6 | A Piper **batchelt** futtatása | Nem mérve | Hosszú könyvhől vágott blokkok |
| 7 | **150 mondatos WER-korpusz** | A `core/hu_wer` 24 mondatot tartalmaz (ellenőrizve 2026-10-04) | A korpusz bővítése; enélkül a 0,07–0,19 %-os WER-ek nem statisztikailag érdemiek |
| 8 | MOSS-TTS-Nano WER/CER | A hallás döntött, ASR nem futott rajta | az `asr_check_wavs.py` a F5-próba 3 WAV-jára futott; azok 2026-10-05-től nincsenek a gépen, újra kell generálni |
| 9 | Batch 5 és 6 **minősége** | Csak RTF mérve (2,60× / 2,15×) | `--skip-wer` nélküli futás |
| 10 | A 4B-es GPU-modellek | Kizárva VRAM/hardware miatt, nem mérve | — |
| 11 | spaCy nélküli karakterfelismerés pontossága | A regex-visszaesés működik, de nincs szám | `evaluate_character_detection.py` egy referenciakorpuszszal |
| 12 | A F5-TTS sebesség **hosszabb mondatoknál** | A mérés 3,4–3,8 s mondatokon futott; a 7,6 óra/óra extrapoláció | Hosszabb szövegblokkok a `f5_matrix.py`-ban |

## 4. Tévesnek bizonyult állítások — ne fussunk bele újra

| A téves állítás | Mi a valóság | Bizonyíték |
|---|---|---|
| „A `torchaudio` nincs win_arm64 wheelje, **tehát minden PyTorch-klónozó zárt**" | A premissza igaz, a **következtetés téves**: x86_64 emulációval a magyar F5-TTS működik | M-12 |
| „A `get_providers()` QNN-t mutat, **tehát** NPU-futtatás van" | A profilban **0 QNN node**; a kimenet bitre a CPU-é, a „216 ms" egy szálas CPU-futtatás | M-27 |
| „Az INT8 kvantálás gyorsít" | Ezen a processzoron **+8 % lassabb** serial módban; batchelt módban csak akkor előzi az fp32-ot, 0,1638 vs 0,1758 | M-07, M-09 |
| „Az NPU-út ezen a modellen lehetetlen" | A **konverzió működik**, DLC készül két SDK-val is. Csak a **futtatás** hiányzik | M-21…M-23 |
| „A Snapdragon X Elite Hexagon NPU-ja V81" | **Téves.** Az X Elite HTP-architektúrája **V73**; a V81 más (későbbi) platformoké. Ezért kerestük hiába a V81 drivert, és ezért tűnt zártnak az NPU-út — a V73 driver-stack végig a gépen volt | M-47; `DriverStore\...\qcnspmcdm8380\HTP\QnnHtpV73StubDrv.dll` |
| „Az ORT QNN EP csendben mindig CPU-ra esik" | **Nem.** `session.disable_cpu_ep_fallback=1` mellett a session létrejön (nem dob), az ORT-profilban minden node `QNNExecutionProvider`, a QNN HTP profil pedig 4 HVX-szálas akcelerátor-végrehajtást mutat | M-47 |
| „Az `onnx` nem telepíthető ARM64-en" | `onnx 1.23.1` (`cp312-abi3-win_arm64`) elérhető; a statikus QDQ export viszont nem állítható elő | M-36 |
| „A `D:`-os venv-ből `avcodec-58` hiányzik" | **Nem hiányzó fájl**, hanem FFmpeg-generáció-ütközés (a DLL 58-at kér, a `bin` 63-at tartalmaz) | M-30 |
| „RTF 0,04 = gyors" | Ez **idő/hang**: 0,04 → **25× lassabb**. A Supertonic RTF-je 0,144, *az* a gyors | M-12 |
| **„A gated modell le van zárva, tehát nem mérhető"** | a `403` a session `HF_TOKEN`-jének jogosultságáról szólt, nem a modellről. Az űrlap kitöltése után nyíltan letölthető, és **mérve működik magyarra** | M-39 |
| „A HF `model_info()` siker ⇒ van hozzáférés" | Csak a metaadat látszik; a `hf_hub_download` 403-at dob | M-29 |
| „A vocoder a szűk keresztmetszet" | A **`vector_estimator`** (93 %); a korábbi feltevés fordított volt | M-10 |
| `matrix.log` záró sora: „1 óra hangra (nfe=32, F1): **2,5 perc**” | **Hibás képlet** (`3600 * (at/tt) / 60`, a fordított aránnyal) — ez a sor egy régebbi `f5_native.py`-verzióból származik. A helyes érték **23,9 óra** (nfe=32), illetve **7,5 óra** (nfe=8) | M-12; a képlet javítva a `f5_native.py`-ban |

## 5. Artifact-leltár — hol a bizonyíték

**Utolsó ellenőrzés: 2026-10-05, a tisztítás UTÁN.** Az „állapot" oszlop a
tényleges tartalom, nem az eredeti feltevés.

> **2026-10-05: az F5-TTS és az NPU kutatás kitisztult a gépről.** A nyers
> mérési adatok (json, log, probe program) a repóban vannak a
> `docs/meresi-adatok\` alatt, hogy a számok továbbra is ellenőrizhetők
> legyenek. A hangminta-WAV-ok szándékosan **nincsenek** meg: azok
> 4,5 percnyi audio, nem bizonyíték. A teljes indoklás:
> `D:\<projekt>\hub\docs\dont.md`.

| Útvonal | Tartalom | Állapot (2026-10-05) |
|---|---|---|
| `docs/meresi-adatok/f5-tts/` | M-12/M-13/M-16/M-17 nyers adatai | ✅ **archív**: `matrix.log`, `native.log`, `f5hu.log`, `out_matrix/bench.json` + `cases.json`, `out_native/cases.json`, `out_piper/report.json` + `cases.json` + `phonemes.json`, `out_piper_native/report.json` + `cases.json`, és a 8 mérőprogram |
| `docs/meresi-adatok/npu-qnn/` | M-21…M-28 nyers adatai | ✅ **archív**: 20 `convert*.log`/`htp*.log`/`cpu_run.log`/`prep*.log`, `npu_probe.json`, `probe_auris.out`, `probe_foundry.out`, 4 QAIRT `*_net.json` (a konverter pontos parancssorával), a 6 probe program, `qai-hub-cli.txt` |
| ~~`C:\AI\clone-probe\`~~ | F5 + Piper mérőkörnyezet (2 venv, 5 kimenet-mappa) | ❌ **TÖRÖLVE** (2,5 GB) — lásd `docs/meresi-adatok/f5-tts/` |
| ~~`C:\AI\npu-work\`~~ | DLC-k és logok | ❌ **TÖRÖLVE** (567 MB) — lásd `docs/meresi-adatok/npu-qnn/` |
| ~~`C:\AI\venvs\{qairt-x64, qairt38-x64, foundry-dev, npu-dev\}`~~ | QNN/QAIRT venv-ek | ❌ **TÖRÖLVE** (1,2 GB) |
| ~~`auris/reader/core/torchaudio_compat.py`~~ | F5 natív shim | ❌ **TÖRÖLVE** — csak az F5 út importálta, semmi más |
| ~~`auris/reader/tests/test_torchaudio_compat.py`~~ | 21 shim-teszt | ❌ **TÖRÖLVE** a shimmel együtt |
| ~~`auris/reader/exports/test-samples/`~~ | 192 generált A/B minta | ❌ **TÖRÖLVE**; a két referenciahang a hubban van (lásd lent) |
| `D:\<projekt>\voices\supertonic\{F1,M3}\` | **a klónreferenciák** (2026-10-05, Supertonic serial fp32) | ✅ F1 254 218 B / 5,30 s, M3 217 790 B — 24 kHz mono PCM16 |
| `D:\<projekt>\apps\hu-voice-ai\models\piper\` | M-16/M-17 forrásmodelljei | ✅ 3 magyar ONNX + `.json` (a `anna` mellett három `_*_b1/_fixed/_static` variáns is) — 2026-10-06: átköltözött `D:\hu-voice-ai\`-ből a hub `apps\`-ába |
| **`D:\<projekt>\vendor\XTTSv2-Streaming-ONNX\`** | M-39 forrásmodelljei (a felhasználó töltötte le `D:\XTTSv2-Streaming-ONNX`-re; 2026-10-06: átköltözött a hub `vendor\`-ába) | ✅ **2,24 GB**: `gpt_model.onnx` 1,52 GB, `gpt_model_int8.onnx` 383 MB, `conditioning_encoder.onnx` 185 MB, `hifigan_vocoder.onnx` 71 MB, `speaker_encoder.onnx` 32 MB, 4 `.npy` embedding, `vocab.json`, `metadata.json`, 3 python modul, 14 referencia/minta WAV+MP3+FLAC |
| `D:\<projekt>\envs\xtts-probe\` | M-39 mérő**környezete** | ✅ `venv-arm64` (natív ARM64), `stubs\spacy\` (loud stub) — a mérőprogram a hubban; az env 2026-10-06-ban átköltözött, a régi `C:\AI\xtts-probe\` törölve |
| **`D:\<projekt>\hub\scripts\xtts_hu_probe.py`** | az XTTS mérőprogram (új hely, `--ref-dir`/`--out`) | ✅ 2026-10-05: smoke újrafutott, 1,25× valós idő, 3,30 s hang |
| **`D:\<projekt>\hub\registry\` + `scripts\doctor.py`** | a hub leírása és az állapotellenőrző | ✅ 10 modell, 5 env, 2 hang — `doctor.py` exit 0 |
| `C:\Qualcomm\AIStack\`, `C:\Qualcomm\VoiceAI\` | QAIRT 2.50+2.45, ASR 2.7.1.0 | ✅ **MEGTARTVA** (felhasználói döntés, 8,3 GB) |

## 6. Reprodukciós parancsok

Minden kapcsoló ellenőrizve létezik (2026-10-04; a 6.1–6.3 parancssorok
útvonalai 2026-10-05-ön frissítve).

### 6.1 A repó venvből (natív ARM64)

```bat
cd /d "C:\Users\<user>\Dev\portfolio\Projects\audiobook narrator\auris\reader"

:: unit tesztek (2026-10-05: 634 teszt, 0 bukás; 2 ismert modulhiány-hiba: torchaudio, wetext.
:: A 21 shim-teszt a torchaudio_compat.py-vel együtt eltűnt, ezért 655 helyett 634.)
.venv\Scripts\python.exe -m unittest discover -s tests -t tests -p "test_*.py"

:: WER-korpusz (24 mondat) és RTF-variánsok
.venv\Scripts\python.exe -m core.hu_wer
.venv\Scripts\python.exe -m core.hu_wer --case bethlen --case moricz --out wer.json
.venv\Scripts\python.exe scripts\bench_supertonic_variants.py
.venv\Scripts\python.exe scripts\bench_supertonic_variants.py --skip-wer --rtf-samples 12 --json-out r.json
.venv\Scripts\python.exe scripts\bench_supertonic_variants.py --variant int8 --steps 6 --cases 12

:: NPU/HTP diagnosztika (0 = van NPU-futtatás, 1 = nincs).
:: 2026-10-05: a scripts\npu_probe.py az archívumba került
:: (docs\meresi-adatok\npu-qnn\), mert a venv-jei megszűntek.
:: Új venv nélkül nem fut: onnxruntime-qnn kell hozzá.
.venv\Scripts\python.exe docs\meresi-adatok\npu-qnn\npu_probe.py

:: tetszőleges WAV-készlet ASR-ellenőrzése
.venv\Scripts\python.exe scripts\asr_check_wavs.py --json D:\<projekt>\out\xtts\cases_int8.json --language hu
```

### 6.2 A lezárt kutatások mérőprogramjai — archívumban

Ezek **2026-10-05-ön törlésre kerültek**; a programok és a nyers adatok a
`docs\meresi-adatok\` alatt vannak, de a venv-jeik nem futnak többé.
Az alábbi parancsok **történeti referencia**ként maradnak itt, nem
futtathatók állapotban.

```bat
:: M-12: F5-TTS emulált (x86_64) — hang x nfe_step mátrix
cd /d docs\meresi-adatok\f5-tts && f5_matrix.py
:: M-13: F5-TTS natív ARM64 + torchaudio-shim
f5_native.py
:: M-16: magyar Piper emulált + a phonem-ID-k kiírása
D:\<projekt>\apps\hu-voice-ai\venv-x64\Scripts\python.exe piper_hu_probe.py
:: M-17: magyar Piper natív ARM64 ORT-tal (a phonem-ID-kból)
"<repo>\auris\reader\.venv\Scripts\python.exe" piper_hu_native.py
```

### 6.3 Az XTTSv2-ONNX klónmérő (M-39)

A mérőprogram **2026-10-05-ön a hubba költözött**
(`D:\<projekt>\hub\scripts\xtts_hu_probe.py`), mert onnan más projekt is
használhatja. Az env **2026-10-06-ban** a `D:\<projekt>\envs\xtts-probe\venv-arm64`
alá költözött: mérés igazolta, hogy a venv Windowson **áthelyezhető** (a
`python.exe` és az importok a `pyvenv.cfg`-ből relatívan oldódnak fel; csak a
console-scriptek törnek el, azok `python -m pip install --force-reinstall
--no-deps <csomag>`-gal újratermelhetők). A régi `C:\AI\xtts-probe\` törölve.

```bat
cd /d D:\<projekt>\hub\scripts

:: gyors füstpróba: egy mondat, int8 GPT
D:\<projekt>\envs\xtts-probe\venv-arm64\Scripts\python.exe xtts_hu_probe.py --smoke

:: teljes mátrix: 2 referencia-hang × 2 mondat, int8 (az ajánlott)
D:\<projekt>\envs\xtts-probe\venv-arm64\Scripts\python.exe xtts_hu_probe.py --mode int8

:: ugyanez fp32 GPT-vel (a WER kissé jobb, a sebesség 30 %-kal rosszabb)
D:\<projekt>\envs\xtts-probe\venv-arm64\Scripts\python.exe xtts_hu_probe.py --mode fp32

:: kapcsolók: --threads N  --chunk N  --speed F  --verbose
::           --ref-dir DIR  --out DIR  (új, 2026-10-05)
```

A függőségek (`numpy onnxruntime soundfile scipy num2words tokenizers
librosa pypinyin hangul-romanize`) mind ARM64 wheelből jöttek; **egyedül a
`spacy` nem telepíthető**, ezért `stubs\spacy\` kell — és a magyar út nem
használja (lásd M-40).

### 6.4 A pipeline függőségiMinimuma magyarhoz

A repó `requirements.txt`-e 17 csomagot sorol fel, de magyarul **nem kell** a
többség: a `gruut`, `g2pkk`, `bangla`, `mecab`, `cutlet`, `jieba`, `spacy`
csak más nyelvekhez kellene. A modul-szintű importok közül három „kell",
de csak a más nyelvek ágához tartozik (`pypinyin`, `hangul_romanize`,
`spacy.lang.*`).

## 7. A jegyzőkönyv és a kapcsolódó doku-k kapcsolata

| Fájl | Szerep | Frissítendő, ha |
|---|---|---|
| `meresek.md` (**ez**) | a bizonyíték katalógusa, az M-számok, a nem-mért lista, a téves állítások | új mérés készült, vagy egy M-szám státusza változott |
| `arm64-install.md` | a *módszer* és a *rendszer* leírása témánként (hogyan működik az ONNX→DLC, mi a batchelés) | a megoldás változik |
| `handoff-2026-10-03.md` | az *útmutató* és a nyitott lista (hol van minden, mi a következő lépés) | a leltár vagy a terv változik |
| `docs/meresi-adatok/` | a F5 és az NPU kutatás **nyers adatai** (json, log, mérőprogram) | új archivált kutatás készül |
| `D:\<projekt>\hub\docs\dont.md` | **mi szűnt ki a gépről és miért** (a tisztítás indoklása) | újabb törlés vagy migráció történik |

**Szabály: egy mérés számát a `meresek.md`-be írd be, a magyarázatát az
`arm64-install.md`-be.** Ha egy szám két helyen szerepel, a `meresek.md`
az ürtruthoz kötődik, az `arm64-install.md` a környezetéhez.

## 8. Változásnapló

| Dátum | Változás |
|---|---|
| 2026-10-04 | **Létrehozva.** A `arm64-install.md` (823 sor) és a `handoff-2026-10-03.md` (402 sor) minden mért állítását 38 számozott bejegyzésbe rendezve, 6 bizonyítéki szinttel. A nem mért kérdésekből 12-es nyitott lista, a tévesnek bizonyult állításokból 10-es nyilvántartás. Kijavítva a `f5_native.py` fordított idejű extrapolációját (600× eltérés) és a `piper_hu_probe.py` `SyntaxWarning`-ját. Ellenőrizve: 192 exportált WAV, `CORPUS` = 24 mondat, minden dokumentált kapcsoló létezik. |
| 2026-10-04 (később) | **M-39, M-40 és a 2.6 szakasz: az XTTSv2-ONNX magyar klón MEGBÍZVA.** A felhasználó elfogadta a CC BY-NC 4.0 űrlapot és letöltötte (`D:\XTTSv2-Streaming-ONNX`, 2,24 GB). M-29 ⚏→✅, a 3. szakasz 2. pontja lezárult, a 4. szakasz egy új sorral bővült. 40 bejegyzés. Mért: int8 1,32× (45,4 perc/óra hang), fp32 1,02× (58,6 perc/óra), WER 0,18 % / 0,11 %. |
| 2026-10-05 | **A kutatás lezárva és a gép tisztítva.** A Supertonic 3 + XTTSv2-ONNX klón a nyertes; az F5-TTS és a Hexagon NPU kutatás kikerült a gépről. Törölve: `C:\AI\clone-probe` (2,5 GB), `C:\AI\npu-work` (567 MB), `C:\AI\venvs` (1,2 GB), `core/torchaudio_compat.py` + 21 teszt, a 192 generált export-WAV és az `audio_cache`. Megtartva: `C:\Qualcomm\AIStack` (8,1 GB, felhasználói döntés), `C:\Qualcomm\VoiceAI`, az XTTS modell és mérőenv. A nyers mérési adatok (json/log/proba) a `docs/meresi-adatok/` alatt vannak (11 MB, hangminták nélkül). Új: `D:\<projekt>\` hanghub — registry + `doctor.py`, a referenciahangok (`voices/supertonic/F1,M3`) és az XTTS-mérőprogram. **Az M-számok változatlanok: minden mérés megmaradt, csak a környezete került archivumba.** Újramérve: Supertonic batch 2,29×/2,32×, XTTS smoke 1,25×, unit tesztek 634 (0 bukás, ugyanaz a 2 ismert ERROR). |
| 2026-10-06 | **Egységesítés: minden hang-AI a `D:\<projekt>` főrepo alá került.** XTTS modell → `vendor\XTTSv2-Streaming-ONNX`, XTTS mérőenv → `envs\xtts-probe` (a `C:\AI\xtts-probe` törölve, 356 MB), a `D:\hu-voice-ai` repó → `apps\hu-voice-ai` (git-történettel együtt), az `auris` az `apps\auris` junction-nal látszik. Igazolva: XTTS smoke exit 0 az új helyről, piper szintézis (2,15 s WAV), `doctor.py` exit 0 (10 modell, 5 env), registry JSON-ok érvényesítve (közben kijavítva a `["hu", "en", ...]` nem-JSON rövidítést, és a `doctor.py` most már mind a négy registry fájlt parseolja). **Az M-számok és a mért számok változatlanok, csak az útvonalak.** |
| 2026-10-12 | **M-47, M-48, M-49: az NPU-út újra megnyílt, és a QA lefutott a valódi hangoskönyvön.** (1) **Az X Elite HTP-je V73, nem V81** — a V73 HTP driver-stack (stub driver + skel + HtpUsrDrv) végig megvolt a DriverStore-ban, ezért a korábbi „HNRD V81 hiányzik" gyökérok téves volt. (2) A QNN EP (ORT 1.31.0 + `onnxruntime-qnn` 2.6.0 plugin-EP, `htp_arch="73"`) `disable_cpu_ep_fallback=1` mellett **tényleg a HTP-n fut** — M-47. (3) Megmértük a CPU↔NPU GFLOP/s-t: sűrű fp16 MatMul-on **az NPU lassabb** a 12 szálas CPU-nál (M-48). (4) A minőségellenőrzés (Parakeet TTS-átirat + hangjelzések) **lefutott a generált fejezeteken**: 3. fejezet CER 2,73 % / WER 11,94 %, `long_pause` flag, ASR RTF 0,070 (M-49); a QA 15 unit tesztje mind OK. Új eszközök: `hub/scripts/{npu_qnn_test.py, npu_bench.py, qa_real_audiobook.py}`, env: `D:\<projekt>\envs\qnn-test`. **A korábbi F5/NPU lezárás nem érvényes az NPU-ra.** |
| 2026-10-12 (folyt.) | **M-50…M-52: a QC teljesen működik a teljes könyvön, és megvan a legjobb ASR.** (1) **Három ASR összevetve** ugyanazon a fejezeten (M-50): a magyar Whisper-finemodell (`sarpba/whisper-hu-large-v3-turbo-finetuned`) **WER 8,49 %**-ot hoz, szemben a Parakeet 15,57 %-ával és a Whisper Small 24,53 %-ával — a `core/qa.py` alap ASR-ja `hybrid` lett (Parakeet elöl, Whisper megerősítésre), és a Whisper-út 30 s feletti hangon `return_timestamps`-javítást kapott. (2) **Lefutott a teljes 15 fejezet** (M-51): átlag WER 12,85 % / CER 3,39 %, csak `long_pause` flag-ek, ASR-RTF 0,0579. (3) **A hibrid QC-láb élőben is bizonyított** (M-52): a `qa_hybrid_probe.py` a gyártási úton a gyanús szegmenst Parakeet CER 5,34 %-ról Whisper-megerősítéssel **3,14 %-ra** javította (WER 15,57 % → 8,49 %). (4) **Iparági kutatás** a 2.7 szakaszban: publikus Hexagon NPU-driver (2025-12), Windows ML GA, QNN EP javítás KB5096135, Whisper QNN-es határai, OpenWritr X Elite-es Parakeet-NPU mérés. Az infografikák NPU-szakasza és a teszt-oldal frissítve. |
| 2026-10-13 | **M-53: az egyensúly megmérve, és a módszerváltás működik a kész appban.** Négy 60 s-os valódi szeleten: Parakeet/hibrid WER 18,73 %, RTF 0,062 (teljes könyv 31,6 perc) vs Whisper WER 13,06 %, RTF 1,266 (10,6 óra) → **a hibrid a javasolt egyensúly**. A `settings.asr_backend` négy értéke tényleg másik transcriber-osztályt ad (élő teszt a valódi settings fájlban), plusz új `TranscriberSwitchTest` a tesztekben. A Beállítások "Beszédfelismerő" szekciója megkapta a mért számokat. Betöltési hiba javítva: a `scripts/`-ből futtatáshoz `sys.path` bejegyzés kell (a `qa_real_audiobook.py`-hoz hasonlóan). |
| 2026-10-13 (folyt.) | **M-54: küszöb-sweep mondat-szinten + az illesztő javítása.** 60 valódi mondat (3 fejezet) három tanulsággal: (1) mondatonként a Whisper-HU finemodell **rosszabb** a Parakeetnél (WER 34,18 % vs 10,58 %) — a „Whisper a legpontosabb” megállapítás csak 60–120 s-os szeletekre igaz (M-50); (2) a 0,08-as megerősítő küszöb jó biztonsági háló, lejjebb húzva +3–12× időáron alig javuló minőség; (3) kijavítva egy valódi gyártási hibát: az `align_words` prefix-keresése globálisan ugrált (az olvasó felolvasáskijelzője és a mondat-idők is sérülhettek); regessziós teszt, most 698 teszt (0 bukás). |
| 2026-10-14 | **Az Ellenőrzés oldal is mutatja és állítja a beszédfelismerő módot.** Új `GET/POST /api/books/<id>/qa/asr-backend` végpont (`qa_api.py`): a könyv nyelvére értelmezi a kért módot (`asr_backend`), megmondja mi futna tényleg (`asr_backend_for`), és jelzi melyik opció érhető el a gépen (Parakeet/Whisper csomag és nyelvi támogatás szerint). Az Ellenőrzés oldal választója a négy módot kínálja, a kikapcsolható opció feliratot kap, alatta a hatályos mód + modell látszik; a váltás a Beállítások `asr_backend` mezőjével ugyanazt a kulcsot írja. 5 új API-teszt (14/14 OK a `test_asr_backends.py`-ben, a QA tesztek is zöldek). | Natív ARM64 env (`D:\<projekt>\envs\sherpa-arm64`, sherpa-onnx 1.13.8). A sherpa Supertonic 3 int8 megmérve (M-41: RTF 0,1497, WER 0,00 %), és a Silero VAD beépült a `core/qa.py`-ba opcionális függőségként (`VOICEAI_QA_VAD=off` kikapcsolja, `VOICEAI_SILERO_VAD` a modell-út) — M-42. A whisper-small magyarra nem használható WER-re (M-42ben dokumentálva). Unit tesztek: 639 (0 bukás, ugyanaz a 2 ismert ERROR, 3 skip). Új hub-mérők: `hub/scripts/qa_sherpa_test.py`, `hub/scripts/sherpa_supertonic_probe.py`. |
