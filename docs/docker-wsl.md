# Docker és WSL a Windows ARM64 gépen

Ez a dokumentum két dolgot tisztáz:

1. **melyik út az elsődleges** az Auris futtatására Windowson (a natív venv),
   és mikor érdemes a Dockert választani;
2. **mi a WSL2 és a Docker Engine állapota** ezen a gépen, és mit kell tenni,
   ha a Docker Engine nem indul el.

A gép: **Windows 11 ARM64, Snapdragon X Elite, 12 mag, 31,6 GB RAM**.
Minden alábbi mérés 2026-10-21-én, ezen a gépen készült.

---

## 1. Rövid válasz: igen, a Dockerben működik az Auris

A Docker Desktop 4.96.0 ezen a gépen **működik**, natív ARM64-en, és az Auris
konténerben is fut: a `supertonic_sherpa` motor 1,6 s alatt betölt, a könyv
importálása és a mondatgenerálás HTTP 200-as választ ad.

A korábbi „nem indul a Docker Engine” hiba nem a Dockerben lévő Auris hibája
volt, hanem a Docker Desktop WSL2 háttér-motor indulási problémája. A javítás
a [`scripts/windows/docker-desktop-repair.sh`](../scripts/windows/docker-desktop-repair.sh)
scriptben van; a 4. fejezet leírja a kézi lépéseket is.

## 2. Docker Engine vs. Docker Desktop — mi különbözteti meg őket?

Ez a kérdés rendszeresen felmerül, ezért röviden:

| | **Docker Engine** | **Docker Desktop** |
|---|---|---|
| Mi ez | A konténer-motornak magja (`dockerd`): konténereket indít, képeket kezel, hálózatot és tárhelyet épít | A teljes grafikus alkalmazás, amiben az Engine fut |
| Hogyan indul | Service-ként vagy a `dockerd` parancsal | A Start menüből / a tálcáról |
| Kell hozzá GUI | Nem | Igen, de a `docker` CLI ugyanúgy használható |
| Mi biztosítja a hálózatot és a tárhelyet | A hoszt operációs rendszere (Windowson: a WSL2 kernel + a virtuális lemez) | Ugyanaz, plusz a grafikus kezelés |
| Tipikus cél | Szerver, CI, felhő | Fejlesztői gép, Windowson kényelmes GUI + a Linux konténerek |

**Röviden: a Docker Engine az a technológia, amely a konténereket futtatja; a
Docker Desktop az a csomag, amely ezt a Windowson kényelmesen elérhetővé
teszi.** A `docker` parancs mindkettőhöz ugyanúgy kapcsolódik, így a
`docker build` / `docker compose up` parancsok nem függnek tőle, hogy a
gépen Desktop van-e vagy sem.

Ezen a gépen tehát:

```
docker CLI (29.9.1, windows/arm64)
        │  npipe:////./pipe/dockerDesktopLinuxEngine
        ▼
Docker Desktop 4.96.0  ──►  Docker Engine 29.9.1  (linux/arm64)
        │                              │
        │                              └─ containerd v2.4.2, runc 1.5.2
        ▼
WSL2 kernel 6.18.33.2-microsoft-standard-WSL2  (a `docker-desktop` distro)
        │
        ▼
overlayfs a `docker-desktop-data` virtuális lemezen
```

A `docker context ls` ezt mutatja:

```
NAME              DESCRIPTION                     DOCKER ENDPOINT
default           Current DOCKER_HOST config      npipe:////./pipe/docker_engine
desktop-linux *   Docker Desktop                 npipe:////./pipe/dockerDesktopLinuxEngine
```

A `*` a használt kontextus. A `default` a Windows natív named pipe-jére
mutat — az a hiba esetén használandó, ha a Docker Desktop nincs elindítva.

## 3. A WSL2 állapota ezen a gépen

```
$ wsl --version
WSL-verzió:  2.7.13.0
Kernelverzió: 6.18.33.2-2
WSLg-verzió:  1.0.73.2
Windows-verzió: 10.0.26300.9807

$ wsl -l -v
  NAME                   STATE           VERSION
* appop-ubuntu           Running         2
  docker-desktop         Running         2
  docker-desktop-data    Running         2
```

| Distro | Szerep | Megjegyzés |
|---|---|---|
| `appop-ubuntu` | A fejlesztői WSL környezet | Alapértelmezett distro; itt fut a homelab tooling |
| `docker-desktop` | A Docker Engine futtatási környezete | Docker Desktop menedzseli, **ne módosítsd kézzel** |
| `docker-desktop-data` | A képek és a konténerek tárhelye | A felhasználói image-ek itt vannak |

### A `.wslconfig` és a kernel

```ini
[wsl2]
memory=8GB
swap=2GB
# 2026-10-21: az egyedi (appop-lab) kernel kikapcsolva, mert a Docker Desktop
# WSL2 engine csak a Microsoft által forgalmazott kernelrel indul el.
# Visszaallitas: #kernel=C:\Users\<user>\Dev\labs\appop-lab\wsl-kernel\Image
```

**Ez a lényeg:** a Docker Desktop WSL2 engine a Microsoft saját
kerneljével (`6.18.33.2-microsoft-standard-WSL2`) indul el; egy saját
(`appop-lab`) WSL kernelrel a backend nem állt fel. Ha a `.wslconfig`-ban
saját `kernel=` sor van, a Docker Engine el sem indul — ezt kell először
kikapcsolni.

### Hogyan működik a Docker Desktop 4.96 a WSL2-ben

Az engine payload **nem** a `docker-desktop` distro fáklsystemén belül van,
hanem egy ISO-képből van csatolva:

```
/entries/docker-desktop.iso/<hash>  ->  /tmp/docker-desktop-root   (overlayfs, ro)
```

Ezért a `/opt/docker-desktop` könyvtár **nem létezik** — a korábbi
diagnosztikai scriptek ezt a hiányt hibának tekintették, és hamis riasztást
adtak. A 4. fejezet scriptje már nem ezt vizsgálja.

## 4. Ha a Docker Engine nem indul el — a javítási sorrend

A legtöbb ilyen hiba a következő sorrendben oldható meg, minden lépés
előtt/után a `docker version` kimenetét ellenőrizve.

### 0. lépés — Diagnózis (2 másodperc)

```bash
docker version                  # a kliens és a szerver is elérhető?
docker desktop status           # a Desktop fut?
wsl -l -v                      # mindhárom distro fut?
```

Ha a kliens működik, de a szerver nem, a `docker version` üres
`Server:` blokkal tér vissza. **Ez a jellemző tünet.**

### 1. lépés — Szelényes újraindítás

```bash
docker desktop stop
wsl --shutdown          # leállítja a WSL2 virtuális gépet is
docker desktop start
```

A `wsl --shutdown` a lényeg: amíg a WSL2 gép fut, a Docker Desktop nem tud
újraindítani a saját motorját.

### 2. lépés — A `.wslconfig` ellenőrzése

```bash
grep -n "kernel" "$USERPROFILE/.wslconfig"
```

Ha van `kernel=` sor (saját kernel), **töröld ki**, majd ismételd az
1. lépést. A Microsoft kernelre kell visszaállni.

### 3. lépés — A `docker-desktop` distro újraépítése

Csak akkor kell, ha a distro sérült (a payload hiányzik, a
`docker-desktop-data` viszont megmarad, mert az a felhasználói image-eket
tartalmazza):

```bash
docker desktop stop
wsl --shutdown
wsl --unregister docker-desktop
docker desktop start
```

> **Figyelem:** a `docker-desktop-data` unregisterelése **törli az összes
> lokális Docker képet és a konténereket**. Soha ne ezt unregistereld, ha
> a képekre szükség van.

### 4. lépés — Kézi újraindítás, ha a processzek beragadnak

```bash
taskkill /IM "Docker Desktop.exe" /F /T
taskkill /IM "com.docker.backend.exe" /F /T
taskkill /IM "com.docker.build.exe" /F /T
wsl --shutdown
docker desktop start
```

### Mindez egyben

A fenti lépések össze vannak fűzve a
[`scripts/windows/docker-desktop-repair.sh`](../scripts/windows/docker-desktop-repair.sh)
scriptbe, alparancsokkal:

```bash
# állapot kiírása (diagnózis)
bash scripts/windows/docker-desktop-repair.sh status

# 1. lépés: szelényes újraindítás
bash scripts/windows/docker-desktop-repair.sh restart

# 2. lépés: saját WSL kernel kikapcsolása, majd újraindítás
bash scripts/windows/docker-desktop-repair.sh kernel-reset

# 3. lépés: a docker-desktop distro újraépítése (a képek megmaradnak)
bash scripts/windows/docker-desktop-repair.sh rebuild-distro

# 4. lépés: beragadt processzek leállítása, majd újraindítás
bash scripts/windows/docker-desktop-repair.sh force-restart
```

A script minden futáskor naplót ír a `%TEMP%\auris-docker-repair.log` fájlba,
és a `docker version` kimenetével igazolja az eredményt.

## 5. Az Auris futtatása Dockerben

### A compose fájl

A [`docker-compose.yml`](../docker-compose.yml) CPU-képet épít
(`torch` nélkül, minden motor ONNX Runtime-on fut), és a következő
kötnyezeteket csatolja:

| Hosztbeli út | Konténerbeli út | Tartalom |
|---|---|---|
| `./docker-data/data` | `/app/reader/data` | Könyvtár, SQLite, beállítások |
| `./docker-data/models` | `/app/reader/models` | Letöltött beszédmodellek |
| `./docker-data/exports` | `/app/reader/exports` | Exportált hanganyagok |
| `./docker-data/audio_cache` | `/app/reader/audio_cache` | Hangcache |
| `./docker-data/uploads` | `/app/reader/uploads` | Feltöltött források |
| `./model_backup` | `/app/model_backup` | Modell-mentések |
| `huggingface` (volume) | `/data/huggingface` | Hugging Face gyorsítótár |

### Indítás

```bash
docker compose up -d --build
# majd: http://127.0.0.1:7860
```

A port átirányítás a `docker-compose.yml`-ben `127.0.0.1:7860:7860`, vagyis
csak a helyi gépről érhető el. Ha 7860-as porton már fut valami, a compose
fájl `ports:` sorát módosítani kell (pl. `127.0.0.1:7861:7860`).

### A modellek bekötése

A modellek **nem** a képen belül vannak, hanem a `docker-data/models` mappából
jönnek bind mounttal — így a hoszton letöltött modellek azonnal használhatók
a konténerben, és a konténer újraépítésekor nem kell újra letölteni őket.

A `supertonic_sherpa` motor a `docker-data/models/supertonic-sherpa-int8`
mappából olvassa a saját modelljét (~139 MB), a Hugging Face gyorsítótár pedig
a `huggingface` volume-ban van. Ellenőrzés:

```bash
docker exec <konténer> ls -1 /app/reader/models
docker exec <konténer> .venv/bin/python -c "import onnxruntime; print(onnxruntime.__version__)"
```

A betöltést a napló igazolja:

```
core.local_engines: supertonic_sherpa ready (CPU · sherpa-onnx int8 · betöltés 1.6s)
```

## 6. A natív venv vagy a Docker?

| | Natív venv (`reader\setup.bat` + `reader\run.bat`) | Docker (`docker compose up -d --build`) |
|---|---|---|
| Teljesítmény | **Gyorsabb** — nincs konténer-réteg, az ONNX motor közvetlenül a CPU-n fut | Lassabb indulás, de futásidőben közel azonos |
| Modell-újratöltés | A `reader/models/` mappából, azonnal | A bind mounton át ugyanonnan |
| Tárhely | A `reader/` mappában, a projekttel együtt | A `docker-data/` mappában |
| Elérhetőség | Csak ezen a gépen | Bármely ARM64/x64 gépen, ahol Docker fut |
| Hibakeresés | Egyszerű — a `run.bat` a `reader/.venv` Pythonját indítja | A konténer logját kell olvasni |
| Előny a Snapdragondon | **A natív út az ajánlott.** | Az átviteli/portabilitási igény esetén |

**Ajánlás:** a Snapdragondon a **natív venv** az elsődleges út, a Docker a
hordozható alternatíva. A Dockeres út akkor indokolt, ha a gép nem ARM64,
vagy a modelllemezen kívül akarod futtatni az Aurist.

## 7. Tisztítás

A build cache és az árva képek rendszeresen nőnek:

```bash
docker system df          # aktuális méretek
docker builder prune      # csak a build cache (14,5 GB volt a méréskor)
docker image prune        # csak a nem használt képek
```

A `docker system prune --volumes` **törli a nem használt volume-okat** is —
ebben a `huggingface` volume és a modell-gyorsítótár is érintett lehet, ezért
csak tudatosan futtasd.
