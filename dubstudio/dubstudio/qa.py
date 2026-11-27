"""Mérési kapuk (QA) — a hiba nem mehet ki a kapun.

Minden kapu egy **szám**, ami a hallható hibát méri, és egy küszöb. A kapuk
kudarca nem figyelmeztetés: a folyamat hibás kilépéssel áll meg (``--allow-fail``
esetén csak jelzi), és a riportban pirosan jelenik meg.

Kapuk:
  voice_switch      – akusztikusan azonos szomszédos megszólalások más hangon
                      szólalnak meg (a „színész több hangon beszél” hiba)
  one_voice_per_utt – egy mondaton belül egy hang szól (szerkezeti garancia)
  gender_conflict   – a magyar hang neme ellentétes a szereplő nemével
  coverage          – minden megszólaláshoz tartozik szereplő és hang
  scene_conflict    – egy jelenetben két szereplő ugyanazt a hangot használja
  scene_minor       – egy jelenetben az azonos nemű beszéd nagy részét egy
                      szereplő viszi, mégis egy másik azonos nemű szereplő
                      csak rövid betétet mond (a diarizáció szétszabdalta
                      ugyanazt a színészt több címkére)
  fit               – a szintetizált sorok hossza illeszkedik az ablakhoz
  text_errors       – a szövegminőség-ellenőrző pont megoldatlan hibái
                      (a hanggenerálás ELŐTT, szabályok + helyi LLM)
"""
from __future__ import annotations

import numpy as np

from . import clustering, plan, subs
from .config import Project
from .plan import Entry
from .registry import Registry
from .utterances import Utterance

DEFAULT_GATES = {
    "voice_switch_max": 0,          # db
    "voice_switch_ratio_max": 0.0,  # a mért párok aránya
    "voice_overlap_max": 0,         # egy hangon kiadott, 2+ beszélős megnyilatkozás
    "one_voice_per_utt_max": 0,
    "mixed_cues_max": 0,            # két beszélő egy megnyilatkozásban
    "gender_conflict_max": 0,
    "coverage_min": 1.0,
    "scene_conflict_max": 0,
    "scene_minor_ratio_max": 0.01,  # jelenet-töredék beszéd / teljes beszéd
    "main_shared_max": 0,           # főszereplő, akinek a hangját más is használ
    "fit_over_ratio_max": 0.02,
    "fit_max_over_s": 0.60,
    "text_errors_max": 0,           # a textcheck után megmaradt bizonyított hiba
}


def evaluate(project: Project, tag: str, utts: list[Utterance],
             entries: list[Entry], tts_dir, clusters: dict,
             registry: Registry, synth_stats: dict | None = None,
             text_stats: dict | None = None,
             context_utts: list[Utterance] | None = None) -> dict:
    """Az epizód mérése és a kapuk kiértékelése."""
    t = {**DEFAULT_GATES, **(project.gate or {})}
    t.update({k: v for k, v in (project.tuning or {}).items()
              if k in DEFAULT_GATES})
    metrics: dict = {}

    switches = clustering.violations(utts, project.tuning,
                                     context=context_utts)
    metrics["voice_switch"] = len(switches)
    metrics["voice_switch_examples"] = switches[:20]
    from .embeddings import cosine

    by_ep: dict[str, list[Utterance]] = {}
    for u in utts:
        by_ep.setdefault(u.ep, []).append(u)
    long_pairs = 0
    for seq in by_ep.values():
        seq.sort(key=lambda u: u.start)
        for u, v in zip(seq, seq[1:]):
            if u.emb is None or v.emb is None:
                continue
            if cosine(u.emb, v.emb) >= 0.55:
                long_pairs += 1
    metrics["voice_switch_pairs"] = long_pairs
    metrics["voice_switch_ratio"] = round(len(switches) / max(long_pairs, 1), 3)

    bad_utt = plan.voice_consistency(entries)
    metrics["one_voice_per_utt"] = len(bad_utt)
    metrics["one_voice_per_utt_examples"] = bad_utt[:10]

    # Kétbeszélős feliratsor egy hangon: a felirat „- ” jelölése két fordulót
    # jelöl, tehát itt két különböző színész szövege szólal meg ugyanazon a
    # hangon (gyakran ellentétes nemen). Ez a „nem vált a hang” hiba
    # szerkezeti mutatója; a prepare-nek fordulónként kell szétvágnia.
    # Kétbeszélős megnyilatkozás: vagy a szövegben maradt fordulójel („- A? -
    # B.”), vagy a csoport egy későbbi sora **fordulókezdő** (a szöveg-tisztítás
    # a lógó jelet már kiveszi, ezért a szerkezetet is mérjük).
    mixed = [u for u in utts
             if subs.split_dialogue_parts(u.text)
             or any(subs.starts_new_turn(c.text) for c in u.cues[1:])]
    metrics["mixed_cues"] = len(mixed)

    # Egy hangon két ember: a diarizáció szerint a megnyilatkozás idejében
    # **két külön beszélő** szól, mégis egy magyar hang viszi ki. Ez a
    # leggyakoribb hallható hiba („férfi szöveget női hang mond”), és eddig
    # egyetlen kapu sem látta. A darabszám a félig átfedő sorokat nem
    # tartalmazza: csak a ≥ 0,6 s-os, önálló fordulókat.
    overlap = [u for u in utts
               if int(getattr(u, "multi_diar", 0) or 0) >= 2
               and not subs.is_screen_text(u.text)]
    metrics["voice_overlap"] = len(overlap)
    metrics["voice_overlap_examples"] = [
        {"uid": u.uid, "start": round(u.start, 2), "voice": u.voice,
         "speakers": int(getattr(u, "multi_diar", 0)), "text": u.text[:80]}
        for u in sorted(overlap, key=lambda u: u.start)[:12]]
    metrics["mixed_cues_examples"] = [
        {"uid": u.uid, "char": u.char, "voice": u.voice, "text": u.text[:90]}
        for u in mixed[:10]]

    # Szereplő-szintű nem-ellenőrzés. Két külön dolgot mérünk, mert nem
    # ugyanaz a hiba:
    #   gender_conflict        – a magyar hang neme ellentétes a szereplő
    #                            rögzített nemével: ez casting-hiba, kapu.
    #   gender_source_mismatch – a hang neme ellentétes a **forrásbeszélő**
    #                            erős mért nemével. Lehet tudatos rendezői
    #                            döntés (a rögzített nem az irányadó), ezért
    #                            csak jelzés, nem kapu.
    # A soronkénti p_female a rövid, kiabálós fordulókon zajos: csak a
    # legalább 3 s erős bizonyíték számít.
    from .voices import BY_ID

    evidence: dict[str, dict[str, float]] = {}
    voices: dict[str, str] = {}
    for u in utts:
        if not u.char:
            continue
        if u.voice:
            voices.setdefault(u.char, u.voice)
        if u.p_female is None:
            continue
        d = evidence.setdefault(u.char, {"F": 0.0, "M": 0.0})
        if u.p_female >= 0.85:
            d["F"] += u.dur
        elif u.p_female <= 0.15:
            d["M"] += u.dur

    declared = {c.char_id: c.gender for c in registry.characters}
    conflicts, source_mismatch = [], []
    for char_id, voice in voices.items():
        v = BY_ID.get(voice)
        if not v:
            continue
        want = declared.get(char_id)
        if want in ("F", "M") and v.gender != want:
            conflicts.append({"char": char_id, "voice": voice,
                              "gender": v.gender, "want": want})
        d = evidence.get(char_id) or {}
        if max(d.get("F", 0.0), d.get("M", 0.0)) < 3.0:
            continue
        source = "F" if d["F"] >= d["M"] else "M"
        if source != v.gender:
            source_mismatch.append({"char": char_id, "voice": voice,
                                    "gender": v.gender, "want": source,
                                    "strong_F": round(d["F"], 1),
                                    "strong_M": round(d["M"], 1)})
    metrics["gender_conflict"] = len(conflicts)
    metrics["gender_conflict_examples"] = conflicts[:20]
    metrics["gender_source_mismatch"] = len(source_mismatch)
    metrics["gender_source_examples"] = source_mismatch[:20]

    # Jelenet-szintű szétszabdalás: az azonos nemű beszéd nagy részét egy
    # szereplő viszi, mégis egy másik azonos nemű szereplő csak rövid betétet
    # mond benne. Ez a diarizáció túlszegmentálásának mérőszáma (ugyanaz a
    # színész több címkén): a szinkronban ez hallható hangváltás.
    minor: list[dict] = []
    total: dict[str, float] = {}
    for u in utts:
        if u.char:
            total[u.char] = total.get(u.char, 0.0) + u.dur
    for scene in clustering.scenes(utts):
        per_gender: dict[str, dict[str, float]] = {"F": {}, "M": {}}
        for u in scene:
            g = declared.get(u.char) if u.char else None
            if g in per_gender:
                per_gender[g][u.char] = per_gender[g].get(u.char, 0.0) + u.dur
        for g, vals in per_gender.items():
            if len(vals) < 2:
                continue
            lead = max(vals, key=vals.get)
            # csak valódi töredék számít: a rövid válaszreakció önmagában nem
            # hiba, ha a szereplő máshol önállóan beszél
            if total.get(lead, 0.0) < clustering.SCENE_MAJOR_MIN_S:
                continue
            for cid, secs in vals.items():
                if cid == lead or secs > clustering.SCENE_MINOR_S:
                    continue
                if total.get(cid, 0.0) > clustering.SCENE_FRAGMENT_MAX_S:
                    continue
                minor.append({"char": cid, "lead": lead,
                              "start": round(scene[0].start, 1),
                              "seconds": round(secs, 1)})
    metrics["scene_minor"] = len(minor)
    minor_s = sum(m["seconds"] for m in minor)
    total_s = sum(u.dur for u in utts)
    metrics["scene_minor_s"] = round(minor_s, 1)
    metrics["scene_minor_ratio"] = round(minor_s / max(total_s, 1e-6), 4)
    metrics["scene_minor_examples"] = minor[:12]

    # a tisztán képernyőszöveg sorok nem beszélő szereplők: ne rontsák a lefedettséget
    # a zene/dalszöveg sorok nem dialógusok: nincs hangjuk, és a lefedettséget
    # sem rontják (korábban beszélő nélküli „szereplőként” jelentek meg)
    voiced = [u for u in utts if (u.emb is not None or u.dur >= 0.5)
              and not u.music
              and not (subs.is_screen_text(u.text) and not u.char)]
    covered = [u for u in voiced if u.voice and u.char]
    metrics["utterances"] = len(utts)
    metrics["coverage"] = round(len(covered) / max(len(voiced), 1), 4)

    scene = registry.conflicts(utts)
    metrics["scene_conflict"] = len(scene)
    metrics["scene_conflict_examples"] = scene[:20]
    # szigorúbb nézet: ugyanaz a hang akkor is, ha a pitch eltér. A kapu a
    # pitch-csúszást engedi, de a riportban látszódjon a valódi azonosság.
    same_voice = registry.conflicts(utts, strict=True)
    metrics["scene_same_voice"] = len(same_voice)
    metrics["scene_same_voice_examples"] = same_voice[:20]

    # főszereplő = a legtöbbet beszélők (a hanggyakoriság rangsora);
    # nekik kizárólagos hang jár — a hangjukat más szereplő nem használhatja
    mains = [c for c in registry.characters if c.role == "fő"]
    shared = [c for c in mains if not c.exclusive]
    metrics["mains"] = len(mains)
    metrics["main_shared"] = len(shared)
    metrics["main_shared_examples"] = [
        {"char_id": c.char_id, "voice": c.voice, "duration": round(c.duration, 1),
         "share": round(c.share, 3),
         "others": [o.char_id for o in registry.characters
                    if o.voice == c.voice and o is not c]}
        for c in sorted(shared, key=lambda c: -c.duration)[:20]]

    if synth_stats:
        n = max(synth_stats.get("done", 0) + synth_stats.get("cached", 0), 1)
        metrics["fit_over"] = int(synth_stats.get("over", 0))
        metrics["fit_over_ratio"] = round(synth_stats.get("over", 0) / n, 4)
        metrics["fit_max_over_s"] = round(float(synth_stats.get("max_over", 0.0)), 2)

    metrics["clusters"] = len(clusters)
    metrics["clusters_no_gender"] = sum(1 for c in clusters.values() if not c.gender)
    metrics["clusters_multi_episode"] = sum(1 for c in clusters.values()
                                            if len(set(c.episodes)) > 1)
    # kézi szereplő-döntéssel megszólaló sorok: a felülvizsgálat hatása
    metrics["manual_lines"] = sum(1 for e in entries if e.manual)

    # Töredék-szereplők: 5 s alatti beszédidő. Nem kapu (egy rövid statiszta
    # is lehet valódi), de a szám mutatja, mennyire szabdálta szét a
    # diarizáció ugyanazt a színészt több hangra.
    frags = [c for c in registry.characters
             if 0 < c.duration < 5.0 and c.role != "fő"]
    metrics["fragments"] = len(frags)
    metrics["fragment_examples"] = [
        {"char_id": c.char_id, "voice": c.voice, "seconds": round(c.duration, 1)}
        for c in sorted(frags, key=lambda c: c.duration)[:12]]

    checks = [
        ("voice_switch", metrics["voice_switch"] <= t["voice_switch_max"],
         f"{metrics['voice_switch']} db (max {t['voice_switch_max']})"),
        ("voice_switch_ratio", metrics["voice_switch_ratio"] <= t["voice_switch_ratio_max"],
         f"{metrics['voice_switch_ratio']:.3f} (max {t['voice_switch_ratio_max']})"),
        ("one_voice_per_utt", metrics["one_voice_per_utt"] <= t["one_voice_per_utt_max"],
         f"{metrics['one_voice_per_utt']} db (max {t['one_voice_per_utt_max']})"),
        ("mixed_cues", metrics["mixed_cues"] <= t["mixed_cues_max"],
         f"{metrics['mixed_cues']} db (max {t['mixed_cues_max']})"),
        ("voice_overlap", metrics["voice_overlap"] <= t["voice_overlap_max"],
         f"{metrics['voice_overlap']} db (max {t['voice_overlap_max']})"),
        ("gender_conflict", metrics["gender_conflict"] <= t["gender_conflict_max"],
         f"{metrics['gender_conflict']} db (max {t['gender_conflict_max']})"),
        ("coverage", metrics["coverage"] >= t["coverage_min"],
         f"{metrics['coverage']:.3f} (min {t['coverage_min']})"),
        ("scene_conflict", metrics["scene_conflict"] <= t["scene_conflict_max"],
         f"{metrics['scene_conflict']} db (max {t['scene_conflict_max']})"),
        ("scene_minor", metrics["scene_minor_ratio"] <= t["scene_minor_ratio_max"],
         f"{metrics['scene_minor']} db / {metrics['scene_minor_s']} s "
         f"({metrics['scene_minor_ratio']:.1%}, max {t['scene_minor_ratio_max']:.0%})"),
        ("main_shared", metrics["main_shared"] <= t["main_shared_max"],
         f"{metrics['main_shared']} db / {metrics['mains']} főszereplő "
         f"(max {t['main_shared_max']})"),
    ]
    # nem mért kapuk: a mix-only futás nem szintetizál, ezért az illesztés
    # (fit) mérőszámai nem keletkeznek. A hiányzó "over"/"max_over" miatt a
    # kapu 0 lógó sorral **hamisan átmenne** — ezért "nem mért" jelölést kap,
    # és a riportban is így látszik (nem zöld pipa).
    unmeasured: set[str] = set()
    if synth_stats:
        if synth_stats.get("mix_only"):
            unmeasured.add("fit")
            checks.append(("fit", True,
                           "nem mért — csak keverés (nincs új szintézis); "
                           "az utolsó teljes futás mérése érvényes"))
        else:
            checks.append(("fit", metrics.get("fit_over_ratio", 0) <= t["fit_over_ratio_max"]
                           and metrics.get("fit_max_over_s", 0) <= t["fit_max_over_s"],
                           f"{metrics.get('fit_over', 0)} sor lóg túl, "
                           f"max {metrics.get('fit_max_over_s', 0):+.2f}s "
                           f"(max {t['fit_over_ratio_max']:.0%}, {t['fit_max_over_s']}s)"))

    if text_stats:
        # az LLM-kötelezőség ("required") esetén a nem futott LLM is bukás:
        # a minőség-ellenőrzés nem maradhat csendben elmaradva
        forced = bool(text_stats.get("required")) and \
            text_stats.get("llm_status") not in ("ok", "disabled")
        metrics["text_errors"] = int(text_stats.get("remaining", 0))
        metrics["textcheck"] = text_stats
        det = (f"{metrics['text_errors']} megoldatlan · "
               f"{text_stats.get('rule_fixed', 0)} szabály- + "
               f"{text_stats.get('override_fixed', 0)} felülírás + "
               f"{text_stats.get('override_reviewed', 0)} lezárt + "
               f"{text_stats.get('llm_fixed', 0)} LLM-javítás · "
               f"{text_stats.get('checked', 0)} sor ellenőrizve · "
               f"LLM: {text_stats.get('llm_status')}")
        if forced:
            det += " (szükséges, nem futott)"
        checks.append(("text_errors",
                       metrics["text_errors"] <= t["text_errors_max"]
                       and not forced,
                       det))

    metrics["mix_only"] = bool((synth_stats or {}).get("mix_only"))
    metrics["unmeasured"] = sorted(unmeasured)
    metrics["gates"] = [{"name": n, "ok": bool(ok), "detail": d,
                         "measured": n not in unmeasured}
                        for n, ok, d in checks]
    metrics["passed"] = all(ok for _, ok, _ in checks)
    return metrics


def print_report(metrics: dict) -> None:
    print("\n── QA kapuk ───────────────────────────────────────────")
    if metrics.get("mix_only"):
        print("  (csak keverés: nincs új szintézis — a nem mért kapuk jelölve)")
    for g in metrics["gates"]:
        mark = ("OK " if g["ok"] else "HIBA") if g.get("measured", True) \
            else "NEM MÉRT"
        print(f"  {mark}  {g['name']:20s} {g['detail']}")
    print(f"  összesítés: {'MINDEN KAPU ÁTMENT' if metrics['passed'] else 'VAN BUKÓ KAPU'}")
    if metrics.get("voice_switch_examples"):
        print("\n  hangváltás példák (ugyanaz a hang, más magyar hang):")
        for e in metrics["voice_switch_examples"][:8]:
            print(f"    cos={e['cos']:.2f} {e['a_voice']}->{e['b_voice']}  "
                  f"{e['a_text'][:32]!r} | {e['b_text'][:32]!r}")
    if metrics.get("gender_conflict_examples"):
        print("\n  nem-ellentmondás példák (hang vs. szereplő neme):")
        for e in metrics["gender_conflict_examples"][:6]:
            print(f"    {e['char']} {e['voice']} — hang: {e['gender']}, "
                  f"szereplő: {e['want']}")
    if metrics.get("gender_source_examples"):
        print("\n  forrásnem-eltérés (jelzés, nem kapu):")
        for e in metrics["gender_source_examples"][:6]:
            print(f"    {e['char']} {e['voice']} — hang: {e['gender']}, "
                  f"forrás: {e['want']} "
                  f"(erős F {e['strong_F']:.1f} s / M {e['strong_M']:.1f} s)")
    if metrics.get("main_shared_examples"):
        print("\n  osztott hangú főszereplők:")
        for e in metrics["main_shared_examples"][:8]:
            print(f"    {e['char_id']} {e['voice']} {e['duration']:.0f}s "
                  f"({e['share']:.1%}) — osztozik: {', '.join(e['others'])}")
