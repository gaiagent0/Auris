"""Magyar érthetőségi mérés (WER/CER) a beszédmotorhoz.

A mérés valódi kört jár be: magyar mondatot szintetizálunk, majd visszaírjuk
ASR-rel, és a felolvasott szöveget összevetjük az eredetivel. Ugyanazt a
`core.qa` értékelést használja, mint a fejezetellenőrző oldal, tehát a számok
egyeznek azzal, amit a felhasználó a felületen lát.

Ez NEM új mérőműszert jelent, hanem ugyanannak a mérésnek a reprodukálható,
verziózott korpuszát. A `run_bench` a szintetizálást és a visszaírást
függvényben kapja, így méréslogika modellek betöltése nélkül is tesztelhető.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable

from core.qa import score_transcript


@dataclass(frozen=True)
class Case:
    """Egy korpuszmondat."""

    id: str
    text: str
    note: str


# magyar mondatok; a cél a kiejtés szöveges megítélése, nem az irodalmi érték
CORPUS: tuple[Case, ...] = (
    Case("simple", "A tengerparton reggel még köd lebegett a fák között.", "egyszerű mondat"),
    Case("soft", "Az öreg kőház udvarán csendesen gólyák bújtak a nádasban.", "cs, ty, hosszú á"),
    Case("digraph", "Gyönyörű tájakon barangoltunk hosszú nyári délutánokon.", "gy, ty, hosszú ű/á"),
    Case("csucs", "A hegycsúcsról lenézve láttuk a völgyben futó patakot.", "csúcs, szótag záró sz"),
    Case("numbers", "Százhuszonhárom forintot kellett kifizetnie a vendéglőben.", "szám szavazva"),
    Case("year", "A házban 1872-ben született, és tizenkilenc éves korában költözött el.", "évszám és életkor"),
    Case("shakespeare", "Shakespeare drámáinak magyar fordítását nagyon szerette olvasni.", "kiejtési szótár: idegen név"),
    Case("bethlen", "Bethlen Gábor fejedelem idején éltek Erdélyben.", "kiejtési szótár: néma h"),
    Case("moricz", "Móricz Zsigmond regényeit a könyvtárban lehetett kölcsönözni.", "kiejtési szótár: cz"),
    Case("abbrev", "A kérdésre nincs további válasz, mondta ki nyugodtan.", "rövidítés feloldása nélkül"),
    Case("cselek", "A miniszter úr útmutatása szerint a fejlesztést 2030-ig kell befejezni.", "kettős mássalhangzó"),
    Case("double-name", "Kőrösfői-Móricz Erzsébet megírta a magyar felsőoklat történetét.", "kettős nevű író"),
    Case("medals", "A díjkategóriák közül a legjobb az aranyé volt, az ezüst és a bronz követte.", "összetett szavak"),
    Case("motto", "Szabadság, egyenlőség, testvériség — e három szó sok mindent elmond.", "hosszú magánhangzók"),
    Case("summer", "A nyár elején megjött a forróság, és a folyó lassan apadni kezdett.", "hosszú á, ő"),
    Case("zsuzsi", "Zsuzsanna néni zsályát ültette a kert végében, ahogy régen mindig.", "zs, ty"),
    Case("knight", "Az ifjú lovag csillogó páncélban lépett a terembe.", "cs, páncél"),
    Case("hermit", "A hegyek között megbúvó kunyhóban egy öreg ember élt magányosan.", "kettős mássalhangzó"),
    Case("temple", "Föléptette a szentélyt, majd csengettyént megszólalt a nagyteremben.", "ty, lt"),
    Case("bigyear", "Háromezerhatyszázhuszonkét esztendő telt el a város alapítása óta.", "hosszú szám"),
    Case("dragon", "A hétfejű sárkány réges-régi meséje minden magyar gyereket elkísért.", "hosszú ű"),
    Case("valley", "Fátyolos völgyön jártunk, ahol a fák lombja összehajolt a hangos szélben.", "hosszú ö, ly"),
    Case("grandfather", "A nagyatyám azt mondta, hogy soha ne bízzunk senkiben, akit nem vérrokon ismertünk.", "hosszú mondat"),
    Case("lighthouse", "A tengerparti világítótorony vörös fénye távolról is felismerhető volt.", "rövid magánhangzók"),
)


def corpus_ids() -> list[str]:
    return [case.id for case in CORPUS]


def get_case(case_id: str) -> Case:
    for case in CORPUS:
        if case.id == case_id:
            return case
    raise KeyError(f"Ismeretlen korpuszmondat: {case_id}")


def run_bench(
    cases: list[Case],
    synthesize: Callable[[Case], str],
    transcribe: Callable[[str], str],
    language: str = "hu",
    progress: Callable[[Case, dict], None] | None = None,
) -> dict:
    """Lefuttatja a mérést a megadott szintetizáló és visszaíró függvénnyel.

    `synthesize` a mondatból WAV-fájlt készít és visszaadja az útvonalát,
    `transcribe` a fájlról szöveget ad vissza. A két függvény külön
    paraméter, hogy a mérés tesztelhető legyen valódi modellek nélkül.
    """
    rows: list[dict] = []
    errors = 0
    total_errors = 0
    total_words = 0
    started = time.perf_counter()

    for case in cases:
        row = {"id": case.id, "note": case.note, "text": case.text}
        try:
            audio_path = synthesize(case)
            row["audio_path"] = audio_path
            row["heard"] = transcribe(audio_path)
        except Exception as exc:  # egy mondat hibája ne állítsa le a mérést
            row["error"] = f"{type(exc).__name__}: {exc}"
            errors += 1
            rows.append(row)
            if progress:
                progress(case, row)
            continue
        score = score_transcript(case.text, row["heard"], language)
        row.update(
            wer=score["wer"],
            cer=score["cer"],
            missing_words=score["missing_words"],
        )
        # Az összesített WER-t a teljes szövegre számítjuk, nem mondatonkénti
        # átlaggal: az utóbbi a rövid mondatok túlzott súlyozását hozná.
        total_errors += score["wer"] * len(score["expected"].split())
        total_words += len(score["expected"].split())
        rows.append(row)
        if progress:
            progress(case, row)

    scored = [r for r in rows if "wer" in r]
    elapsed = time.perf_counter() - started
    return {
        "language": language,
        "cases": len(rows),
        "scored": len(scored),
        "errors": errors,
        "wer": round(total_errors / total_words, 4) if total_words else None,
        "cer": round(sum(r["cer"] for r in scored) / len(scored), 4) if scored else None,
        "worst": sorted(
            (r for r in scored if r["wer"] > 0),
            key=lambda r: -r["wer"],
        )[:5],
        "elapsed_sec": round(elapsed, 2),
        "results": rows,
    }


def format_report(report: dict) -> str:
    """EMBER által olvasható összegzés a mérési eredményből."""
    lines = [
        "Magyar érthetőségi mérés",
        "=" * 52,
        f"mondat: {report['cases']}  |  értékelve: {report['scored']}  "
        f"|  hiba: {report['errors']}  |  {report['elapsed_sec']} s",
    ]
    if report["wer"] is None:
        lines.append("Nincs értékelhető eredmény.")
        return "\n".join(lines)
    lines.append(f"WER: {report['wer'] * 100:.1f}%   átlag CER: {report['cer'] * 100:.1f}%")
    lines.append("")
    lines.append(f"{'mondat':<14}{'WER':>8}  {'elvétett szavak'}")
    lines.append("-" * 52)
    for row in sorted(
        report["results"], key=lambda r: (-r["wer"], r["id"]) if "wer" in r else (0, r["id"])
    ):
        if "wer" not in row:
            lines.append(f"{row['id']:<14}{'—':>8}  HIBA: {row.get('error', '')[:40]}")
            continue
        if row["wer"] == 0:
            continue
        missed = ", ".join(row["missing_words"][:4])
        lines.append(f"{row['id']:<14}{row['wer'] * 100:7.1f}%  {missed}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Magyar WER-mérés a Supertonic 3-ra.")
    parser.add_argument("--case", action="append", help="csak adott korpuszmondat")
    parser.add_argument("--out", help="eredmény JSON fájlba")
    args = parser.parse_args(argv)

    cases = [get_case(cid) for cid in args.case] if args.case else list(CORPUS)

    from core.local_engines import SupertonicEngine
    from core.qa import Transcriber

    engine = SupertonicEngine()
    engine.load_sync()
    transcriber = Transcriber.for_language("hu")

    def synthesize(case: Case) -> str:
        result = engine.generate(text=case.text, language="hu")
        return result["audio_path"]

    def transcribe(path: str) -> str:
        return transcriber.transcribe(path, "hu").get("text", "")

    def on_progress(case: Case, row: dict) -> None:
        state = "HIBA" if "error" in row else f"WER {row['wer'] * 100:.1f}%"
        print(f"  {case.id:<14} {state}", flush=True)

    print(f"Magyar érthetőségi mérés — {len(cases)} mondat")
    report = run_bench(cases, synthesize, transcribe, progress=on_progress)
    print()
    print(format_report(report))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print(f"\nRészletek: {args.out}")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())