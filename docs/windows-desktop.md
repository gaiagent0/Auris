# Auris Windows ARM64 (Snapdragon X Elite)

Az asztali kiadás **Windows 11 ARM64 (Snapdragon X Elite)** rendszeren
használható. A telepítő saját Python-környezetet, CPU-s AI-függőségeket és
FFmpeg/ffprobe programokat tartalmaz. A használathoz nem kell Git,
Python-telepítés vagy parancssor.

## Telepítés és első indítás

1. Indítsd el az Aurist. Kövesd a magyar telepítő lépéseit. Alapértelmezett
   programhely: `%LOCALAPPDATA%\Programs\Auris`. Rendszergazdai jogosultság
   nem szükséges.
2. Indítsd az Aurist az asztali vagy Start menüben létrehozott ikonnal.
3. Az első indításkor válaszd a **Gyors indulás** (Supertonic 3, kb. 0,4 GB)
   vagy a **Saját hangok és hangklónozás** (XTTSv2-ONNX, a telepítéssel
   együtt jár) lehetőséget, majd kattints a **Letöltés és beállítás** gombra.
   Az állapot és az aktuális fájl letöltése az ablakban követhető.
4. Kész beállítás után nyisd meg a könyvtárat.

A Supertonic modell letöltéséhez és a hiányzó Microsoft WebView2
beállításához internet kell. A már telepített motorok helyi felolvasása
internet nélkül is működik; webes cikkimport és külső API-k használata
továbbra is hálózati kapcsolatot igényel.

## Hardver: minden motor CPU-n fut

Az Auris kizárólag Snapdragon X Elite / Windows ARM64 rendszerre készült;
minden beszédmotor ONNX Runtime-on, a processzoron fut, videokártya vagy
külön illesztőprogram nem kell. A mért sebességek: a sherpa-Supertonic RTF
0,14 (az auto-ajánlott motor), a Supertonic 0,25, a klónozó XTTSv2-ONNX 0,58.

Más motort később a **Beállítások** oldalon választhatsz; a motorválasztó
alatt látszik, melyik motor futtatható ezen a gépen. Az asztali kezdeti
beállítás újra megnyitható a Súgó asztali telepítésről szóló részéből.

## Könyvek, modellek és frissítés

Minden felhasználói adat a `%LOCALAPPDATA%\Auris` mappában van:

- `data`: könyvtáradatbázis, beállítások, hangprofilok és mentések;
- `uploads`: feltöltött forrásfájlok;
- `audio_cache`: elkészített hangok;
- `exports`: exportált hangoskönyvek;
- `models`: letöltött modellek;
- `runtime`: az opcionális GPU-környezet és az alkalmazásablak adatai;
- `logs`: indítási és telepítési naplók.

Frissítéshez zárd be az Aurist, majd futtasd az új telepítőt. A programfájlok
cserélődnek, a külön tárolt könyvek és modellek megmaradnak. Eltávolításkor
is megmarad a felhasználói adatmappa. A forrásból futó webes példány és az
asztali kiadás külön könyvtárat használ; az Auris ZIP-mentésével viheted át
a könyveket és beállításokat.

Az Auris ablakának bezárása leállítja a helyi szervert és a háttérfeladatokat.
A megszakadt tartós feladatokat a következő indítás után a Feladatok oldalon
folytathatod. Egy adatmappához egyszerre egy asztali példány indulhat.

## Hibaelhárítás

- A kiadás nincs digitálisan aláírva. A Windows SmartScreen jelezhet ismeretlen
  kiadót; a telepítőt a projekt GitHub Release oldaláról töltsd le. A kiadás
  `SHA256SUMS.txt` fájlja tartalmazza a telepítő ellenőrzőösszegét.
- Ha a modellletöltés megszakad, ellenőrizd az internetet és a szabad
  tárhelyet, majd indítsd újra a beállítást. A kész fájlokat újra felhasználjuk.
- Ha az ablak nem indul, futtasd újra a telepítőt internetkapcsolattal a
  hiányzó WebView2 beállításához. Ellenőrizd a `logs/desktop.log` és
  `logs/launcher.log` fájlokat.
- Ha a klónozó motor nem jelenik meg vagy nem tölt be, ellenőrizd, hogy a
  `vendor/XTTSv2-Streaming-ONNX/xtts_onnx` könyvtár megvan-e, és telepítve
  vannak-e a `librosa` és `pypinyin` csomagok. A Beállítások motorválasztója
  jelzi, melyik motor futtatható ezen a gépen.
- OCR-hez a Tesseract és a magyar nyelvi adat, speciális könyvformátumokhoz
  a Calibre továbbra is opcionális, külön telepítendő eszköz. Az FFmpeg és
  ffprobe az asztali csomag része.

## Fejlesztői build és ellenőrzés

A build a projekt tesztelt virtuális környezetéből csak a szükséges
csomagokat másolja át; könyveket, beállításokat, hangmintákat és
TTS-modellsúlyokat nem csomagol. A meglévő magyar NLP-modell
belekerülhet a függőségekkel együtt. A csomagolt CPython önálló, izolált
futtatókörnyezet (ARM64 célra a beágyazott Python-csomagnak illeszkednie
kell a gazdaarchitektúrához). A WebView-alapú ablakhoz pywebview 6.2.0
kerül bele. A build letölti a hivatalos Python-csomagot, a Microsoft
WebView2 bootstrappert és az FFmpeg Windows-buildjét. A teljes
függőséglista a release `runtime-manifest.json` mellékletében van.

```powershell
reader\.venv\Scripts\python.exe scripts\windows\build.py
# Forrásmódosítás után, a már elkészített runtime újrahasználatával:
reader\.venv\Scripts\python.exe scripts\windows\build.py --app-only
```

Az Inno Setup 7 buildeszköz szükség esetén a `build/windows/inno` mappába
települ. Kimenet: `dist/windows/Auris-Setup-<verzió>-x64.exe`.

A csomagolt háttérszerver külön tesztadatokkal is elindítható:

```powershell
build\windows\package\Auris.exe --headless --data-dir C:\Temp\auris-desktop-qa --port 17903
```

A normál webes indítás (`reader/run.bat`, illetve `reader/app.py`) továbbra
is használható. Egyedi telepítésben az `AURIS_DATA_DIR` környezeti változóval
adható meg a külön adattárolás helye. A kiadási ellenőrzés a csomagolt
futtatókörnyezetet rendszer-Python és külső FFmpeg nélküli PATH mellett,
első modellletöltéssel, valódi felolvasással, újraindítással, frissítési
adatmegőrzéssel és helyi Playwright-képernyőképekkel vizsgálja.
