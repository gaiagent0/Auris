"""dubstudio — magyar szinkronkészítő alkalmazás filmekhez.

Az alkalmazás alapelve: **előbb a forduló, aztán a hang.**
A korábbi lánc feliratsoronként, rövid akusztikus ablakon klaszterezett, ezért
ugyanaz a színész mondatról mondatra más magyar hangot kaphatott. Az új lánc a
teljes énekbeszéd-sávon pyannote 3.0 szegmentálóval és CAM++ embeddinggel keresi
a beszélőfordulókat, a feliratsort ahhoz rendeli, majd nem és hanghasonlóság
alapján validálja.

Ez az alkalmazás ezért:
  1. minden feliratsort a **beszélőfordulóhoz** rendel (pyannote + CAM++);
  2. a fordulókat nem szerint ellenőrzi, a kevert neműt szétvágja, és csak
     azonos nemű, indokolt esetben vonja össze;
  3. a sorokat fordulónként, **összefüggő megnyilatkozásokba** csoportosítja,
     így egy mondaton belül elvileg sem szólalhat meg két hang;
  4. a klaszterezést **évad-szinten** végzi, és a klasztert egy
     **szereplő-rekordhoz** köti, aminek a hangja **zárolt**;
  5. minden hanghoz **fix generálási paramétert** rendel (a hang karaktere
     soronként nem változhat), az időzítést pedig utólag, atempo-val igazítja;
  6. minden lépést **mér** (QA-kapu), és a hibát nem engedi ki a kapun.

Modulok:
  config      – projekt és epizód útvonalak
  subs        – SRT beolvasás, szövegtisztítás, magyar szöveg-előkészítés
  audio       – WAV I/O, resample, atempo, fix hangszín-eltolás
  media       – ffmpeg/ffprobe, mux
  embeddings  – CAM++ beszélő-embedding
  gender      – tanított nem-osztályozó (nem F0!)
  diarization – pyannote + CAM++ forduló-alapú beszélőazonosítás
  utterances  – sorok → beszélőfordulók → megnyilatkozások
  clustering  – évad-szintű klaszterezés + diarizációs validálás
  voices      – 13 magyar hang + XTTS klónhang, fix paraméterekkel
  registry    – szereplő-rekord és hangzár (cast.json)
  plan        – időzítési terv
  synth       – TTS szintézis fix paraméterekkel
  mix         – keverés és mux
  qa          – mérési kapuk
  report      – HTML riport
  speaker_audit – forrás-hang próba (diarizáció + héber szóillesztés)
  identity    – stabil, epizódokon átívelő C### karakterazonosság (galéria)
  review      – emberi elfogadás/javítás állapota (review.json)
  voice_decision – tipizált döntési modell a hangkiosztáshoz
  words       – szó-idő ellenőrzés (szerkezet, energia, tempó, támasz)
  pipeline    – a teljes lánc
  cli         – parancssor
"""

__version__ = "1.0.0"
