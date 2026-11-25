# GitHub-kutatás: AI-videószerkesztők, TTS- és hangmodellek (2026-11-07)

Ez a jegyzet a 2026-11-07-i kutatás eredménye, amelyet a szinkronstúdió
sorozat/film moduljához és a beépített felülvizsgáló lejátszóhoz készítettünk.
A cél nem az volt, hogy egy teljes AI-videószerkesztőt vagy felhős
dubbing-szolgáltatást vegyünk át, hanem hogy a **már működő, ARM64/ONNX
láncra** adaptálható mintákat és korlátokat azonosítsuk.

## 1. AI-videószerkesztők és dubbing-pipeline-ok

| Repó | Mit érdemes belőle átvenni | Licenc / megjegyzés |
|---|---|---|
| [dngrs-dev/dublaro](https://github.com/dngrs-dev/dublaro) | Tiszta pipeline-szemlélet: külön szakasz a beszélő-azonosítás, a fordítás, a TTS és a keverés; a szakaszok között fájl-alapú, újraindítható állapot. | A repó licence a beépítés előtt ellenőrizendő; a mintát **nem** másoltuk, csak a felépítést követtük. |
| [wassgha/rescript](https://github.com/wassgha/rescript) | A transcript-szerkesztő élmény: a szöveg, az időzítés és a beszélő egy felületen javítható; a lejátszó a szerkesztés közben szól. | **PolyForm Noncommercial** — kereskedelmi újrahasznosítás nem megengedett; csak UX-referencia. |
| [OpenCut-app/OpenCut](https://github.com/OpenCut-app/OpenCut) | Modern, idősávos szerkesztő elrendezés, billentyűparancsok, előnézeti panel. | MIT; a kód közvetlen átvétele nem szükséges. |
| [OpenCut-app/opencut-classic](https://github.com/OpenCut-app/opencut-classic) | Régebbi, egyszerűbb szerkesztő-UI; jó kiindulás a „kevés dolog, nagy gombok” elvhez. | A repó licence ellenőrizendő. |
| [etro-js/etro](https://github.com/etro-js/etro) | Böngészős videó-kompozíció és effekt-réteg; hasznos, ha később előnézeti overlay kell. | Könnyű, de nem dubbing-motor. |
| [Huanshere/VideoLingo](https://github.com/Huanshere/VideoLingo) | Teljes videó → felirat → fordítás → TTS → mux folyamat; a szakaszok külön futtathatók. | Nagy Python/PyTorch lánc; Snapdragonon nem ez a biztos út. |

**Döntés:** nem húztunk be nehéz PyTorch-alapú videószerkesztőt. A meglévő
`prepare → cluster → dub → QA → mux` lánc marad, mert ez már bizonyítottan fut
a gépen. A UI a Rescript/OpenCut mintájából csak a **lejátszó + egyszerű
döntések** elvet veszi át.

## 2. TTS- és hangklónozó motorok

| Motor / modell | Szerep a projektben | Licenc | ARM64 / Snapdragon X Elite |
|---|---|---|---|
| [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) | ONNX-alapú TTS/ASR futtató; a Supertonic és Piper modellek hordozója. | Apache-2.0 | ARM64 CPU ONNX Runtimemal megy; ez a fő út. |
| [supertone-inc/supertonic](https://github.com/supertone-inc/supertonic) | 5 női + 5 férfi magyar hang, gyors, on-device. | Kód: MIT; **modellek: OpenRAIL-M** | ONNX Runtime CPU; GPU nem szükséges. Az OpenRAIL-M használati korlátai külön figyelendők. |
| [OHF-Voice/piper1-gpl](https://github.com/OHF-Voice/piper1-gpl) | Tartalék/alternatív magyar hangok (`anna`, `berta`, `imre`). | GPL-3.0 (a kód); a modellek saját licence szerint | CPU-n működik; a projektben a meglévő Piper útvonal marad. |
| [ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp) | Helyi ASR-ellenőrzés, CPU-n. | MIT | ARM64 natív; jó alternatíva, de a jelenlegi QA a meglévő ASR-t használja. |
| [ONNX Runtime QNN EP](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html) | Snapdragon NPU-gyorsítás lehetősége. | MIT | **Csak QNN-re exportált/kompatibilis modellekkel**; nem drop-in a mostani ONNX modellekhez. |
| [ONNX Runtime CPU EP](https://onnxruntime.ai/docs/execution-providers/CPU-ExecutionProvider.html) | A biztos, hordozható végrehajtás. | MIT | Windows ARM64 wheel; a jelenlegi lánc ezt használja. |

**Döntés:** marad a **sherpa-onnx + Supertonic + Piper + ffmpeg** kombináció.
CUDA nem opció Snapdragonon; a QNN/NPU csak külön modell-exporttal lenne
érdemes, ezért egyelőre nem része a pipeline-nak. A hangklónozást nem
automatizáljuk a felhasználó engedélye nélkül; a meglévő hangkatalógus
szereplőnkénti kiosztása marad.

## 3. Több hangsávos lejátszás böngészőben

A kutatás megerősítette, hogy a `HTMLMediaElement.audioTracks` széles körben
nem megbízható, és a böngészők az MKV több hangsávját jellemzően nem
választják szét. Ezért a felülvizsgáló lejátszó **nem** a nyers MKV-t tölti be,
hanem a szerver oldalon készülő proxyt:

- a kiválasztott hangsávot **egy** hangsávként tartalmazó MP4,
- H.264 esetén videó-copy, egyébként `libx264` újrakódolás,
- AAC 160 kbps, `+faststart`, Range-kérésekkel,
- a feliratok WebVTT-vé alakítva; a képfeliratok (PGS/VobSub) kimaradnak,
- a kész proxy a `work/<EP>/review/` alatt gyorsítótárazódik.

Az [hls.js](https://github.com/video-dev/hls.js/blob/master/docs/API.md)
szegmentált HLS-t tudna adni, de helyi, fájlos lejátszásnál a Range-képes MP4
egyszerűbb és kevesebb mozgó alkatrész. A proxy első elkészítése nagy
médiánál percekig tarthat, ezért a felület jelzi, hogy „sávok betöltése…”.

## 4. Amit a sorozat/film modulhoz átvettünk

- **Sorozat:** a könyvtár az `SxxExx` / `E12` mintából ismeri fel az
  epizódokat; a státusz a média, a magyar felirat, a leválasztott
  énekbeszéd és az M&E meglétéből számol. Csak a hiánytalan rész indítható.
- **Új epizód:** a média hozzáadása után a meglévő évad-szintű cast és
  hangkiosztás öröklődik; a feldolgozás a meglévő `run --season` paranccsal
  fut. A figyelő (opcionális) csak kész anyagok mellett indít.
- **Film:** egyetlen `E01` egység, nincs évad-szintű újrafeldolgozás; a
  filmfájl bármely támogatott konténer lehet.
- **Korlát:** a nyers új MKV-ból a jelenlegi lánc nem fordít magyar feliratot
  és nem választja szét a forrás hangját. Ehhez külön intake/fordítás/separáció
  kellene; a modul ezért a kész `subs/EXX_hun.srt`, `work/EXX/vocals.wav` és
  `work/EXX/me.wav` (vagy a projektben beállított minták) mellett automatikus.

## 5. Források

- https://github.com/dngrs-dev/dublaro
- https://github.com/wassgha/rescript
- https://github.com/OpenCut-app/OpenCut
- https://github.com/OpenCut-app/opencut-classic
- https://github.com/etro-js/etro
- https://github.com/Huanshere/VideoLingo
- https://github.com/k2-fsa/sherpa-onnx
- https://github.com/supertone-inc/supertonic
- https://github.com/OHF-Voice/piper1-gpl
- https://github.com/ggml-org/whisper.cpp
- https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html
- https://onnxruntime.ai/docs/execution-providers/CPU-ExecutionProvider.html
- https://github.com/video-dev/hls.js/blob/master/docs/API.md
