# Hasonló repók — kutatás és az átvételi döntés (2026-11-01)

A feladat: a **szinkron modul Auris-ba építése** előtt nézzük meg, mi van erre
a nyílt forráskódban, és mi érdemel beépítést. Ez a jegyzőkönyv a talált
repókat, az átvett elveket és az elutasított elemeket rögzíti, hogy később
visszakövethető legyen, miért úgy döntöttünk.

> Alapelv a repóban: **„Ami nincs mérve, az nincs kész.”** Ezért a pipeline
> magját (diarizáció → klaszterezés → hangzár → szintézis → keverés → QA) az
> építés közben nem bántottuk: kutatás-vezérrelt változtatás csak új mérés
> után jöhet (M-számozott mérések az Aurisban:
> `reader/docs/meresek.md`).

---

## 1. A talált repók röviden

Forrás-URL-ek (ellenőrizve: 2026-11-01):

| Repó | URL | Mit csinál jól |
|---|---|---|
| pyVideoTrans | <https://github.com/jianchang512/pyvideotrans> | teljes lánc, kézi szerepkiosztás, „SpeedRate” illesztő-motor (a felirat végét a következő felirat kezdetéig nyújtja) |
| Dub Studio (timoncool) | <https://github.com/timoncool/dub-studio> | **évad-szintű casting-profil**: karakter→hang párt az egész sorozatra ment |
| youtube-auto-dub | <https://github.com/mazzasaverio/youtube-auto-dub> | mondatalapú újrasegmentálás + hossz-tudatos fordítás; méréseik: stretch-igény 6/6→1/4 sor, túl-tömörített szegmensek 86%→56% |
| Edge-TTS-Subtitle-Dubbing | <https://github.com/fr0stb1rd/Edge-TTS-Subtitle-Dubbing> | tiszta SRT→dub: time-slot filling, minta-pontos konkatenáció, futásidejű QA-riport (overlaps / late-starts / accuracy) |
| mazinger | <https://github.com/bakrianoo/mazinger> | **speech map**: VAD a vokál-stemen, így a sor a tényleges beszéddel szinkronizál, nem a felirat-ablakkal |
| VideoLingo | <https://github.com/Huanshere/VideoLingo> | Netflix-szabványú felirat-szegmentálás, fordítás-ellenőrző hurok. A README szerint **nincs** beszélőnkénti hang |
| Subdub | <https://github.com/lukaszliniewicz/Subdub> | stage-ekre bontott CLI; `--speed_up 115` (legfeljebb 115%-os gyorsítás) kapu |
| Linly-Dubbing | <https://github.com/Kedreamix/Linly-Dubbing> | vokál-szeparáció + lip-sync; elavult dependency-stack — csak referencia |
| OmniPlay | <https://github.com/akarsh911/OmniPlay> | beszélőnkénti referencia-minták; similarity-küszöb 0,85 |

Amit **egyik** repó sem tud, csak mi: **feliratsor-szintű ASR-szóidő
építfél-től függetlenül**, tipizált (nem-nyelvi) hangkiosztó döntési modell
CAM++ hanglenyomaton, **fő-szereplő kizárólagos hanggal**, és
240 mp-es **forrás-audit próba emberi jóváhagyással** — a felülvizsgáló-felület
teljesen az Auris részére készült.

---

## 2. Amit átvettünk (azonnal, ebben az építésben)

1. **WebUI-ból szerkeszthető beállítás** — a pyVideoTrans „explicit érték
   minden küszöbre” elve: a projekt-JSON `tuning`, `gate`, `llm`, `asr`,
   `models` szakaszai az Auris `/szinkron` Beállítások szekciójából
   olvashatók és írhatók. Minden mentés **biztonsági mentéssel** történik
   (`projects/_backup/`), és írás előtt érvényesítés: a tuning/gate
   számfolt-medáljait és a models útvonaltípusát nehéz-e — hibás bemásolást
   nem enged a fájlba kerülni.
2. **Futtatás-felügyelő job-runner** — egyszerre egy futás (a TTS és az ffmpeg
   nem párhuzamosítható értelmesen), a napló fájlba megy, a UI **3 mp-ként
   lekérdezi** az állapotot és a napló utolsó ~120 sorát mutatja, leállítás
   gombbal (terminate → 10 mp vár → kill). Ez a mazinger „stage-enként
   újraindítható” elvének gyors megvalósítása: QA-bukás nem jelent teljes
   újrafuttatást, a napló marad.
3. **Hangkatalógus modell betöltése nélkül** (a CLI `voices` parancsának
   HTTP-változata): a UI-listához nem tölt be ~1 GB modellt, csak a
   modell-mappák létezését nézi (`model_present` jelöléssel a kártyán).

---

## 3. Amit később veszünk át (sorban, mert mérni kell)

A kutatás legkonkrétabb tanulságai (a számok a fenti repók saját mérései,
nem a mieink — az átvétel előtt nézünk M-számú mérést):

| Átveendő | Honnan | Hol lenne a helyünk | Kapu/referencia |
|---|---|---|---|
| sorrend: mondatalapú rétegzés → hossz-tudatos fordítás → **utolsó lépésként** time-stretch szigorú plafonnal | youtube-auto-dub + Subdub + Edge-TTS-Subtitle-Dubbing | `plan`/`synth`/`fit`-kapu | plafon ~1,10–1,15 (Subdub 115%); fölötte **újrafordítás**, nem nyújtás; Edge-TTS mérés: 99,97%-os célhossz-pontosság, 0,00 s drift |
| **speech map** — a sor ott kezdődik, ahol a tényleges beszélő, ott szünetel, ahol az eredeti | mazinger | `plan` + `mix` (ducking) | így a nevetés, levegővétel és a zene/effekt a helyén marad |
| futó-riport minta: összesített sorok száma / cache-elt / failed / overlaps / late-starts | Edge-TTS-Subtitle-Dubbing | `report.py` | jól olvasható, futás közben bővülő lista |
| **LUFS / peak normalizálás a keverésnél** | Dub Studio (−14 LUFS, −1 dBTP), OmniPlay (bg 0,3 / tts 0,9) | `mix.py` mastering | a forrásfilm-világ,-szabvány; az Auris audiobook oldala −16 LUFS-t mért — a szinkronnál −14/−1 lesz a cél |
| casting-profil még könnyebb visszahívása | Dub Studio | `registry.py` / cast-JSON | a miénk már évad-szinttens köti a hangot; a „profil” tovább vihető sorozatváltásra |

Licensz-bariát: a fentiek közül a pyVideoTrans GPL-3.0 — **annak forráskódját
nem másoljuk** (a megoldás-elv, nem a kód megy át); MIT/Apache-2.0
repokód esetén is csak kis, megcímzettuar képletekkel jön be.

---

## 4. Amit tudatosan NEM vettünk át

- **Fordítás-újraceszerrés** (VideoLingo háromlépéses fordítási hurok): a
  szinkron-szöveg a dubtitle feliratból jön, nem gépi fordítás; a
  `textcheck.py` már csinál LLM-es szövegellenőrzést szintézis előtt.
- **Lip-sync** (Linly-Dubbing): ADR-sorcikat kellene színész-szinkronizálni;
  a mi felületünk felirat-vezérelt, és a forrás sorozat, nem videó-remezés.
- **Szigorúan automatikus besorolás** (nincs emberi kapu): a kutatás
  egybehangzó tapasztalata (pyVideoTrans popup-kiosztás, mazinger
  chunk-editor), hogy a teljes automatizálás nem életszerű; nálunk a
  beszélő-azonosság **emberi döntéssel** záródik (`review.json`), ez marad
  a jellemző — a WebUI ezt tette gyorsabbá, nem vette el.
- **A 0,85-ös koszinusz-küszöb blind átvétele**: OmniPlay 0,85 /
  youtube-auto-dub 0,888 hang-egyezésre; a mi `link_cos`-unk 0,62 —
  alacsonyabb, mert mögötte évad-szintű Viterbi-finomítás és emberi
  jóváhagyás van. Nálunk a mérés (M-szám) döntsön, nem a repók száma.

---

## 5. Ellentmondások, amikre ráéreztünk

- **stretch-kapuérték**: a kutatás konszenzusa ±10–15%; a mi `fit` kapunk
  ≤2% (≤0,6 s) — szigorúbb, mert nálunk a szintézis fix paraméteres, a
  sebességet csak utólag, `atempo`-val igazítjuk. Ha a speech map és a
  hossz-tudatos szöveg jön, ezen a kapun szétvigyük a kapuszámokat.
- **audio-vs-video kompenzáció** (pyVideoTrans: az eltérés fele videó-PTS):
  filmre jó, sorozatra nem — a video PTS nem évad-szintű konstrukció.
- **felirat-ablak vs. beszéd-terület**: a pyVideoTrans az ablakot nyújtja a
  következő sorig, a mazinger a beszédet követi — a kettő a jelenet-ütközés
  szempontjából másképp viselkedik; döntés előtt mindkettőt kellene mérni.

---

## 6. Források (mindegyiket megnyitva, 2026-11-01)

- <https://github.com/jianchang512/pyvideotrans> (és an.pyvideotrans.com: subtitle-sound-alignment, multi-role-dubbing)
- <https://github.com/timoncool/dub-studio>
- <https://github.com/mazzasaverio/youtube-auto-dub>
- <https://github.com/fr0stb1rd/Edge-TTS-Subtitle-Dubbing>
- <https://github.com/bakrianoo/mazinger>
- <https://github.com/Huanshere/VideoLingo>
- <https://github.com/lukaszliniewicz/Subdub>
- <https://github.com/Kedreamix/Linly-Dubbing>
- <https://github.com/akarsh911/OmniPlay>
- <https://github.com/YYuX-1145/Srt-AI-Voice-Assistant>
- <https://github.com/R3gm/SoniTranslate>
- <https://github.com/dngrs-dev/dublaro> (2026-11-07-i kutatás; lásd `github-ai-video-tts-2026-11-07.md`)
- <https://github.com/wassgha/rescript> (2026-11-07-i kutatás; UX-referencia, PolyForm Noncommercial)
- <https://github.com/OpenCut-app/OpenCut> (2026-11-07-i kutatás; MIT)
- <https://videodubbing.com/blog/post/best-open-source-video-dubbing-tools-2026/>
- <https://github.com/topics/video-dubbing>
