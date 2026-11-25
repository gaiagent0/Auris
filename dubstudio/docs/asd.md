# Képi beszélő-ellenőrzés (ASD) — Light-ASD, kézi ground truthtal mérve

Miért van erre szükség: a hang-alapú azonosítás nem tud különbséget tenni két
ember között, ha a képen **egyikük** beszél és a másik csak hallgat, de a
hangkarakterisztika hasonló. A kép segít — de csak akkor, ha a kérdésre
(„melyik arc szája jár a hallható beszéd alatt”) **mérhető** válasz van.

## Miért nem a korreláció (a korábbi megoldás)

A `dubstudio/asd.py`-ban maradt `corr_sig` a száj-sáv mozgását korrelálja a
hang-burkológéppel. A mérés (2026-11, S02E04):

* 2 s-os ablakon (±3 képkocka eltolás-kereséssel) a **véletlen** párok
  legjobb korrelációja is eléri r = 0,80-at; 60 mintán 0,41-et.
* A 15 leghosszabb gyanús soron a legjobb r **0,05–0,37** volt, mind a
  véletlen szintjén → **0** megerősítés, 0 téves állítás.

Vagyis a jel a zajszinten van: a korreláció **mérés** maradt (r, z, szájaktív
idő a riportban), de **nem dönt** — a `score_candidates` függvényt kivettük.
Ha valakinek mégis a legegyszerűbb kép kell, a `work/visual_audit.py` adja:
„van-e ajakmozgás a sor idejében”, arc-azonosítás nélkül.

## Mi dönt helyette

**Light-ASD** (Liao et al., *A Light Weight Model for Active Speaker
Detection*, CVPR 2023, MIT, <https://github.com/Junhua-Liao/Light-ASD>):
1,02 M paraméter, 0,6 GFLOP, tisztán CPU-n fut. A port a
`dubstudio/lightasd.py` (architektúra + MFCC + arckivágás), a képi rész a
`work/visual_asd.py`, a döntés az `asd.decide_light`.

Bemenet a tanítással egyezően:

* **25 fps**, szürke, 112×112-es **arckivágás** (YuNet követés, 960×540-es
  dekódolás, négyzetes kivágás 1,3× ráhagyással);
* **MFCC 13** együttható 100 fps-en (25 ms ablak, 10 ms lépés) → képkockánként
  pontosan 4 MFCC-sor.

Három finomítás a tanításon túl, mert **sor**-szinten dolgozunk (mindhárom
mérésből született, nem találgatásból):

1. **Beszéd-súly**: a soron belüli csend lehúzza az átlagot (egy 11 s-os
   sorban lehet 3 s beszéd). A „beszél-e ez az arc **a hallható beszéd
   alatt**” kérdésre a válasz a leghangosabb MFCC-sor −25 dB-jénél aktív
   képkockák átlaga (`lightasd.speech_weight`, `score_line`).
2. **Jelenlét-maszk**: csak azokon a képkockákon pontozunk, ahol az arc
   **tényleg látszik** (`score_line(present=…)`). Enélkül a követő lyukait
   kitöltő elavult kivágás is beleszámít, és egy 1,7 s-os közeli vágás
   beszélőjét a 7 s-os sor többi része lehúzza.
3. **Képkocka-simítás** ±4 képkocka mozgóátlaggal a beszéd-intervallumokhoz.

### Amit a kézi címkézés talált a mérőláncban (és javítottunk)

| hiba | tünet | javítás |
| --- | --- | --- |
| YuNet fix 0,6-os küszöb | sötét jelenetben a **valódi** 355 px-es arc csak 0,28–0,64 pontot kap → „nincs arc a képen” | kontraszt-nyújtás (2–98 percentilis) + `SCORE_MIN = 0,35`; a döntést nem a detektor pontszáma hozza |
| követő csak távolságot nézett | vágás után identitást váltott: egy „arc” során két különböző ember | jelenetvágás-detektálás (8×-ra ritkított képkocka-eltérés > 10) + 16×16-os kinézet-ellenőrzés |
| pályaszűrő `len/4` | egy 1,5 s-os közeli vágás beszélője kiesett, a **hallgató** maradt | enyhe szűrő (≥ 5 képkocka), a döntésnél a jelenlét számít |

## Döntési szabály (`asd.decide_light`)

| feltétel | kimenet |
| --- | --- |
| legjobb p ≥ 0,45 **és** ≥ 0,15-tel vezet | `arc #k beszél (p …)` |
| legjobb p ≤ 0,35 | `nincs beszélő arc a képen` → képen kívüli beszélő |
| 0,35 < p < 0,45, vagy kicsi az előny | `nem eldönthető` |
| nincs arc / 48 px-nél kisebb arc | `nincs arc` / `nem eldönthető (túl kicsi)` |

A bizonytalant **nem** kényszerítjük döntésre: a szinkronláncban ez
bizonyíték, nem kapu. A küszöb 0,65-ről 0,45-re került a lenti mérés alapján
(a hallgató arcok 0,00–0,15-öt kapnak, a látható beszélők 0,41–0,94-et — a
0,65 a valódi beszélőket tolta „nem eldönthető”-be).

## Kézi ground truth — a 15 hosszú gyanús sor (S02E04)

A címke a `work/biohackers/E04/visual/labels/` kontaktlapjai alapján készült:
soronként 12 teljes képkocka időrendben (befoglalt arcokkal) + arconként 9
nagyított arckivágás. A bizonytalan sorokon külön nagyított szájcsík
(`dubstudio/tools/zoom.py`). A címkék:
`work/biohackers/E04/visual/labels/truth.json`.

* `arc` — a látható arcok egyikének mozog a szája (7 sor)
* `kivul` — a beszélő arca nem látszik (2 sor: éjszakai autó, hangposta)
* `nemtudom` — a képkockákból nem dönthető el (6 sor: kicsi arc, lehajtott
  fej, montázs, tömegjelenet)

| t (s) | igaz | AVA p → kimenet | TalkSet p → kimenet |
| --- | --- | --- | --- |
| 447,17 | arc | 0,13 → képen kívüli ✗ | 0,21 → képen kívül ✗ |
| 1032,42 | arc | **0,94 → arc #0** ✓ | **0,98 → arc #0** ✓ |
| 1050,54 | arc | **0,74 → arc #0** ✓ | **0,89 → arc #0** ✓ |
| 1365,00 | arc | 0,41 → nem eldönthető ✗ | **0,72 → arc #1** ✓ |
| 1398,00 | arc | **0,82 → arc #0** ✓ | **0,68 → arc #0** ✓ |
| 1445,96 | arc | **0,89 → arc #0** ✓ | 0,26 → képen kívüli ✗ |
| 1026,67 | arc | **0,91 → arc #0** ✓ | 0,35 → képen kívüli ✗ |
| 1498,92 | kivul | 0,00 → nem eldönthető (csak 45 px arc) ✗ | ugyanaz ✗ |
| 1764,17 | kivul | **0,00 → képen kívüli** ✓ | **0,05 → képen kívüli** ✓ |
| 965,54 | nemtudom | 0,84 / +0,11 → nem eldönthető | 0,99 / +0,05 → nem eldönthető |
| 1077,38 | nemtudom | 0,96 → arc #1 | 1,00 → arc #1 |
| 180,21 | nemtudom | 0,74 → arc #6 | 0,61 → arc #6 |
| 684,88 | nemtudom | 0,61 / +0,12 → nem eldönthető | 0,56 → arc #1 |
| 942,96 | nemtudom | 0,62 → arc #0 | 0,38 → nem eldönthető |
| 1708,12 | nemtudom | 0,90 → arc #0 | 0,30 → képen kívüli |

**Eredmény a 9 eldönthető soron:**

| pontozó | egyezés | megjegyzés |
| --- | --- | --- |
| korreláció (régi) | – | 0 döntés, nem is pontoz |
| Light-ASD + TalkSet | **5/9 (56 %)** | 3 valódi beszélőt „képen kívüli”-nek mond |
| Light-ASD + AVA | **6/9 (67 %)** | egyetlen kirívó tévedés: a sötét, fekvő arc (447 s) |

A két súly a 15 sorból **9-en mond ellent** egymásnak; a ground truth az
**AVA** mellett dönt (0,68–0,94 a valódi beszélőkre, ahol a TalkSet 0,26–0,35).
Ezért marad az alapértelmezés az AVA súly
(`ASD_WEIGHT=light-asd-talkset.model` env-változóval váltható).

Ahol a küszöb áll: a mért sáv **0,15 … 0,41** üres (hallgató/statiszta fölött,
a legkisebb valódi beszélő alatt) — ide esik a 0,45-ös `SPEAK_P` és a 0,35-ös
`OFFSCREEN_P`. A 1365 s-os sor (p 0,41) **4 századdal** marad a küszöb alatt:
ez nem modellhiba, hanem a 6,8 s-os összevont sor átlagolása (benne 1,7 s
közeli vágás a beszélőről). A helyes javítás nem a küszöb leszorítása, hanem
a soron belüli szakaszra bontás — ez a következő mérés feladata.

## Bizonyíték a riportban (nem kapu)

A `report.json` és a `report.html` `asd` szakasza soronként tartalmazza:
ablak, sor-uid, hány arc volt a képen, melyik beszél, `p`, arcméret,
kimenet, beszéd-intervallumok. A `report.write` a
`work/<EP>/visual/asd.json`-ból dolgozik, a `visual_asd.py` futtatása pedig
rögtön bele is írja a `report.json`-ba. **Semmilyen kaput nem állít** és nem
ír felül hang-alapú döntést: a besorolás a hangból jön, a kép csak azt
mutatja, hogy a beszélő arca látszik-e.

## Futtatás

A súlyok nincsenek a repóban (a `work/` gitignore-olt):

```bash
git clone --depth 1 https://github.com/Junhua-Liao/Light-ASD /tmp/lightasd
cp /tmp/lightasd/weight/pretrain_AVA_CVPR.model   work/models/light-asd-ava.model
cp /tmp/lightasd/weight/finetuning_TalkSet.model  work/models/light-asd-talkset.model
```

```bash
cd dubstudio
python work/visual_asd.py E04 --suspicious 15        # a riport gyanús sorai
python work/visual_asd.py E04 341.12 718.67 --net light
python work/visual_asd.py E04 341.12 --net corr      # csak a régi mérés
python -m dubstudio.tools.asd_bench                 # a két súly a ground truthon
python -m dubstudio.tools.label_sheets 3            # kontaktlap egy sorról
python -m dubstudio.tools.zoom 1032.42 0 420 198 69 1037.62   # nagyított szájcsík
```

Sebesség: ~1 videómásodperc/másodperc CPU-n (15 sor ≈ 4 perc). Teljes
epizódra (~43 perc) tehát ~40 perc — ezért fut csak kérésre, nem a
szinkronlánc részeként.

Kimenet: `work/biohackers/E04/visual/asd.json` (soronként minden arc
`prob`-bal, jelenléttel, beszéd-intervallummal, mérettel) és
`visual/bench_light-asd-*.json` (a ground truthos mérés).

## Korlátok

* **48 px alatti arc**: a 112×112-re nagyítás mosott, a háló nem lát
  ajakmozgást → „nem eldönthető”. Ez a távoli képkockák (totál) esete.
* **Profil / elfordult arc**: a tanítóadatban ritka; ilyenkor a p
  használhatatlan (a 447 s-os, sötét, fekvő arc 0,13-at kapott, pedig beszél).
* **Képen kívüli beszélő**: helyesen „nincs beszélő arc” — a hang-alapú
  azonosítás marad érvényben.
* **Összevont (montázs) sor**: ha egy sor alatt vágás és másik beszélő jön,
  a sor-szintű átlag elveszti az információt; ilyenkor a szakaszra bontás
  (következő lépés) kell.
* A háló **nem tudja**, ki kicsoda: arc-számot ad, a név-hozzárendelés a
  klaszterezés + cast dolga.
