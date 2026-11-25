"""A 15 kézzel címkézett sor mérése: melyik Light-ASD súly jobb, hol a küszöb?

Ugyanazokon a sorokon, ugyanazon a képcsőn lefuttatja az ASD-t a kért
súlyokkal, és összeveti a kézi címkével (``labels/truth.json``):

  arc       — a látható arcok egyike beszél
  kivul     — a beszélő arca nem látszik (a helyes kimenet: „nincs beszélő arc”)
  nemtudom  — a képkockákból nem eldönthető (nem pontozzuk)

Használat:
  python -m dubstudio.tools.asd_bench                       # mindkét súly
  python -m dubstudio.tools.asd_bench light-asd-ava.model
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dubstudio import asd, audio, lightasd  # noqa: E402
from dubstudio.asd_visual import MODEL, Yunet, analyze  # noqa: E402
from dubstudio.config import load_project  # noqa: E402

EP = "E04"
WEIGHTS = ("light-asd-ava.model", "light-asd-talkset.model")


def verdict_class(r: dict) -> str:
    """A kimenet három osztálya: arc / kivul / nemtudom."""
    v = r["light"]["verdict"]
    if r["light"]["speaking"] is not None:
        return "arc"
    if "képen kívüli" in v or v == "nincs arc a képen":
        return "kivul"
    return "nemtudom"


def run(weight: str, rows: list[dict], truth: dict) -> list[dict]:
    p = load_project("projects/biohackers.json")
    ep = p.ep_dir(EP)
    src = sorted(Path(p.data_dir).glob(f"*{EP}*.mkv"))[0]
    net = Yunet(MODEL)
    model = lightasd.load(Path("work/models") / weight)
    x, sr = audio.load_mono(p.vocals(EP))
    x16 = audio.to_16k(x, sr)
    out = []
    for r in rows:
        a = r["t"]
        b = r["end"]
        res = analyze(net, model, src, x16, max(0.0, a - 0.15),
                      (b - a) + 0.3, "light")
        best = max((c["prob"] for c in res["light"]["cands"]), default=0.0)
        out.append({"t": a, "truth": truth[a]["label"], "cls": verdict_class(res),
                    "best_p": round(best, 3), "faces": res["faces"],
                    "cands": res["light"]["cands"], "verdict": res["light"]["verdict"]})
    (ep / "visual" / f"bench_{weight.replace('.model', '')}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def report(name: str, rows: list[dict]) -> dict:
    print(f"\n=== {name} ===")
    ok = bad = und = 0
    arc_p, kiv_p = [], []
    for r in rows:
        mark = "OK " if r["cls"] == r["truth"] else "HIBA"
        if r["truth"] == "nemtudom":
            mark = "—  "
            und += r["cls"] == "nemtudom"
        elif r["cls"] == r["truth"]:
            ok += 1
        else:
            bad += 1
        if r["truth"] == "arc":
            arc_p.append(r["best_p"])
        elif r["truth"] == "kivul":
            kiv_p.append(r["best_p"])
        print(f"  {r['t']:8.2f}s  igaz={r['truth']:8s} gép={r['cls']:8s} "
              f"p={r['best_p']:.2f} arc={r['faces']}  {mark}  {r['verdict'][:60]}")
    n = ok + bad
    print(f"  egyezés: {ok}/{n} ({100 * ok / max(n, 1):.0f}%) · "
          f"bizonytalanra hagyva: {und}/{sum(1 for r in rows if r['truth'] == 'nemtudom')}")
    if arc_p:
        print(f"  „arc” sorok p-je: {['%.2f' % v for v in sorted(arc_p)]}")
    if kiv_p:
        print(f"  „kivul” sorok p-je: {['%.2f' % v for v in sorted(kiv_p)]}")
    # küszöb: a legkisebb p, amitől minden „arc” sor beszélőt ad, és a legnagyobb
    # olyan p, ami alatt minden „kivul” sor helyesen „nincs beszélő arc” marad.
    if arc_p and kiv_p:
        need = min(arc_p)
        offscreen = max(kiv_p) + 0.01
        gap = need - offscreen
        print(f"  a küszöb-sáv: beszél p ≥ {need:.2f} · képen kívüli p ≤ "
              f"{max(kiv_p):.2f} → {'szétválik' if gap > 0 else 'ÜTKÖZIK'} "
              f"(rés {gap:+.2f})")
    return {"ok": ok, "bad": bad, "undecided_kept": und}


def main(argv: list[str]) -> int:
    p = load_project("projects/biohackers.json")
    labels = json.loads((p.ep_dir(EP) / "visual" / "labels" / "truth.json")
                        .read_text(encoding="utf-8"))
    truth = {r["t"]: r for r in labels["rows"]}
    rows = json.loads((p.ep_dir(EP) / "visual" / "asd.json").read_text(encoding="utf-8"))
    rows = [r for r in rows if r["t"] in truth]
    names = list(argv) or list(WEIGHTS)
    summary = {}
    for w in names:
        summary[w] = report(w, run(w, rows, truth))
    print("\nÖSSZEGZÉS")
    for w, s in summary.items():
        n = s["ok"] + s["bad"]
        print(f"  {w:32s} {s['ok']}/{n} egyezés, {s['undecided_kept']} bizonytalan helyesen hagyva")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
