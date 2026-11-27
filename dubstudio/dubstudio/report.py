"""HTML riport: mi történt, mit mértünk, hol maradt hiba."""
from __future__ import annotations

import html
import json
from pathlib import Path

from .config import Project
from .plan import Entry
from .registry import Registry
from .utterances import Utterance

CSS = """
:root{--bg:#0f1115;--card:#171a21;--fg:#e7eaf0;--dim:#9aa3b2;--ok:#3ddc84;
--bad:#ff5d5d;--warn:#ffc857;--acc:#6ea8ff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.5 ui-sans-serif,system-ui,'Segoe UI',sans-serif}
header{padding:22px 28px;border-bottom:1px solid #232733}
h1{margin:0;font-size:20px} h2{font-size:15px;margin:26px 0 10px;color:var(--acc)}
.sub{color:var(--dim);font-size:13px;margin-top:4px}
main{padding:20px 28px 60px;max-width:1180px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}
.card{background:var(--card);border:1px solid #232733;border-radius:10px;padding:14px}
.card .k{color:var(--dim);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.card .v{font-size:22px;font-weight:600;margin-top:6px}
table{border-collapse:collapse;width:100%;margin-top:8px;font-size:13px}
th,td{border-bottom:1px solid #232733;padding:6px 8px;text-align:left;vertical-align:top}
th{color:var(--dim);font-weight:500}
.ok{color:var(--ok)} .bad{color:var(--bad)} .warn{color:var(--warn)}
.dim{color:var(--dim)}
code{background:#11141b;padding:1px 5px;border-radius:4px;font-size:12px}
.gate{display:flex;gap:10px;align-items:baseline;padding:7px 0;border-bottom:1px solid #1d2029}
.gate .n{width:190px;color:var(--dim)}
details{background:var(--card);border:1px solid #232733;border-radius:10px;padding:10px 14px;margin-top:10px}
summary{cursor:pointer}
"""


def _fmt(x, nd=2):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}".replace(".", ",")
    return str(x)


def asd_evidence(project: Project, tag: str) -> dict | None:
    """A képi beszélő-ellenőrzés (``visual/asd.json``) — bizonyíték, nem kapu.

    Soronként a Light-ASD kimenete: hány arc volt a képen, melyik beszél
    (vagy „nincs beszélő arc” / „nem eldönthető”), mekkora valószínűséggel és
    mettől meddig. Ha nincs fájl, ``None`` — a riport ilyenkor kihagyja.
    """
    path = project.ep_dir(tag) / "visual" / "asd.json"
    if not path.is_file():
        return None
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(rows, list) or not rows:
        return None
    out = []
    n_speak = n_off = n_und = 0
    for r in rows:
        light = r.get("light") or {}
        cands = [c for c in (light.get("cands") or [])
                 if isinstance(c, dict)]
        best = max(cands, key=lambda c: float(c.get("prob", 0.0)),
                   default=None)
        verdict = str(light.get("verdict") or "")
        if light.get("speaking") is not None:
            n_speak += 1
        elif (verdict.startswith("nincs beszélő arc")
              or verdict == "nincs arc a képen" or "képen kívüli" in verdict):
            n_off += 1
        else:
            n_und += 1
        out.append({
            "t": r.get("t"), "end": r.get("end"), "uid": r.get("uid"),
            "faces": r.get("faces"), "speaking": light.get("speaking"),
            "verdict": verdict,
            "best_prob": None if best is None
            else round(float(best.get("prob", 0.0)), 3),
            "best_size": None if best is None else best.get("size"),
            "spans": None if best is None else best.get("spans"),
        })
    return {"rows": out, "n": len(out), "on_screen": n_speak,
            "offscreen": n_off, "undecided": n_und,
            "source": str(path.relative_to(project.ep_dir(tag)))}


def asd_html(ev: dict | None) -> str:
    """A riport képi beszélő-ellenőrzés szakasza (bizonyíték, nem kapu)."""
    if not ev or not ev.get("rows"):
        return ""
    rows = "".join(
        f"<tr><td>{_fmt(r.get('t'))}–{_fmt(r.get('end'))}</td>"
        f"<td><code>{html.escape(str(r.get('uid') or '—'))}</code></td>"
        f"<td>{r.get('faces')}</td>"
        f"<td>{'arc #%d' % r['speaking'] if r.get('speaking') is not None else '—'}</td>"
        f"<td>{_fmt(r.get('best_prob'))}</td>"
        f"<td>{r.get('best_size') or '—'} px</td>"
        f"<td>{html.escape(str(r.get('verdict') or ''))}</td>"
        f"<td class=\"dim\">{html.escape(str(r.get('spans') or ''))}</td></tr>"
        for r in ev["rows"][:200])
    return (
        "<h2>Képi beszélő-ellenőrzés (Light-ASD) — bizonyíték, nem kapu</h2>"
        f'<div class="sub">{ev.get("n", 0)} sor · '
        f'<b class="ok">{ev.get("on_screen", 0)}</b> „arc beszél” · '
        f'<b>{ev.get("offscreen", 0)}</b> képen kívüli · '
        f'<b class="warn">{ev.get("undecided", 0)}</b> nem eldönthető · '
        f'súly <code>{html.escape(str(ev.get("weight") or "—"))}</code> · '
        f'<code>{html.escape(str(ev.get("source") or ""))}</code></div>'
        "<div class=\"sub\">A kép nem dönt a hang helyett: a besorolás "
        "hang-alapú marad, a kimenet csak azt mondja meg, hogy a beszélő arca "
        "látszik-e a képen, illetve hogy a száj mozgása melyik arcot támasztja "
        "alá. „Nem eldönthető” esetén szándékosan nincs állítás.</div>"
        "<table><tr><th>ablak</th><th>sor</th><th>arc</th><th>beszél</th>"
        "<th>p</th><th>arcméret</th><th>döntés</th><th>beszéd-intervallumok</th>"
        "</tr>" + rows + "</table>")


def write(project: Project, tag: str, metrics: dict, utts: list[Utterance],
          entries: list[Entry], clusters: dict, registry: Registry,
          out_path: Path | None = None) -> Path:
    out = out_path or (project.ep_dir(tag) / "report.html")
    m = metrics
    gates = m.get("gates", [])
    rows = []
    cards = [
        ("megnyilatkozás", m.get("utterances", 0)),
        ("szereplő", m.get("clusters", 0)),
        ("hangváltás hiba", m.get("voice_switch", 0)),
        ("nem-ellentmondás", m.get("gender_conflict", 0)),
        ("lefedettség", f"{_fmt(m.get('coverage', 0), 3)}"),
        ("túl hosszú sor", "nem mért" if m.get("mix_only")
         else m.get("fit_over", 0)),
        ("szöveghiba (TTS előtt)",
         (m.get("textcheck") or {}).get("remaining", "—")),
    ]
    for k, v in cards:
        rows.append(f'<div class="card"><div class="k">{html.escape(k)}</div>'
                    f'<div class="v">{v}</div></div>')

    gate_html = []
    unmeasured = set(m.get("unmeasured") or [])
    for g in gates:
        # csak keverés (mix-only) futásnál néhány kapu nem mérhető: ezt jelölni
        # kell, nem zöldnek mutatni (a hiányzó mérés nem „átment”)
        if g.get("measured", True) is False or g["name"] in unmeasured:
            cls, mark = "warn", "NEM MÉRT"
        else:
            cls = "ok" if g["ok"] else "bad"
            mark = "ÁTMENT" if g["ok"] else "BUKOTT"
        gate_html.append(
            f'<div class="gate"><span class="n">{html.escape(g["name"])}</span>'
            f'<span class="{cls}">{mark}</span>'
            f'<span class="sub">{html.escape(g["detail"])}</span></div>')

    chars = []
    by_char: dict[str, dict] = {}
    for u in utts:
        if not u.char:
            continue
        d = by_char.setdefault(u.char, {"voice": u.voice, "dur": 0.0, "n": 0})
        d["dur"] += u.dur
        d["n"] += 1
    reg_by_id = {c.char_id: c for c in registry.characters}
    for cid in sorted(by_char, key=lambda c: -by_char[c]["dur"]):
        d = by_char[cid]
        rc = reg_by_id.get(cid)
        role = rc.role if rc else "—"
        tier = rc.tier if rc else "—"
        excl = ("kizárólagos" if rc.exclusive else "megosztott") if rc else "—"
        rank = rc.rank if rc else 0
        share = f"{_fmt(100 * rc.share, 1)} %" if rc else "—"
        season_dur = rc.duration if rc else d["dur"]
        chars.append(f"<tr><td>{rank}</td><td><code>{html.escape(cid)}</code></td>"
                     f"<td><code>{html.escape(str(d['voice']))}</code></td>"
                     f"<td>{_fmt(d['dur'], 1)} s</td>"
                     f"<td>{_fmt(season_dur, 1)} s</td><td>{d['n']}</td>"
                     f"<td>{html.escape(role)}</td><td>{html.escape(tier)}</td>"
                     f"<td>{share}</td><td>{html.escape(excl)}</td></tr>")

    usage = registry.usage or registry.analyze()
    ep_rows = "".join(
        f"<tr><td>{html.escape(str(e['ep']))}</td><td>{e['n_chars']}</td>"
        f"<td>{e['n_f']} / {e['n_m']}</td><td>{e['n_main']}</td>"
        f"<td>{_fmt(e['seconds'], 1)} s</td></tr>"
        for e in usage.get("episodes", []))
    urows = []
    for v in usage.get("voices", []):
        who = " · ".join(
            f"<code>{html.escape(c['char_id'])}</code> "
            f"{_fmt(c['duration'], 0)} s ({html.escape(c['role'])}, "
            f"pitch {c['pitch']:+.1f})"
            for c in v["chars"])
        cls = "ok" if v["exclusive"] else "warn"
        state = "kizárólagos" if v["exclusive"] else "megosztott"
        urows.append(
            f"<tr><td><code>{html.escape(v['voice'])}</code></td>"
            f"<td>{v['n_chars']}</td><td>{_fmt(v['seconds'], 0)} s</td>"
            f"<td>{_fmt(100 * v['share'], 1)} %</td>"
            f'<td class="{cls}">{state}</td>'
            f"<td>{who}</td></tr>")

    sw = m.get("voice_switch_examples") or []
    sw_html = "".join(
        f"<tr><td>{_fmt(e['cos'])}</td>"
        f"<td><code>{html.escape(str(e['a_voice']))}</code> → "
        f"<code>{html.escape(str(e['b_voice']))}</code></td>"
        f"<td>{html.escape(e['a_text'])}</td><td>{html.escape(e['b_text'])}</td></tr>"
        for e in sw[:40])

    gc = m.get("gender_conflict_examples") or []
    gc_html = "".join(
        f"<tr><td><code>{html.escape(str(e.get('char') or ''))}</code></td>"
        f"<td><code>{html.escape(str(e.get('voice') or ''))}</code></td>"
        f"<td>{html.escape(str(e.get('gender') or ''))} (hang) ≠ "
        f"{html.escape(str(e.get('want') or ''))} (szereplő)</td></tr>"
        for e in gc[:40])

    gs = m.get("gender_source_examples") or []
    gs_html = "".join(
        f"<tr><td><code>{html.escape(str(e.get('char') or ''))}</code></td>"
        f"<td><code>{html.escape(str(e.get('voice') or ''))}</code></td>"
        f"<td>{html.escape(str(e.get('gender') or ''))} (hang) / "
        f"{html.escape(str(e.get('want') or ''))} (forrás)</td>"
        f"<td>F {_fmt(e.get('strong_F'), 1)} s / M {_fmt(e.get('strong_M'), 1)} s</td></tr>"
        for e in gs[:40])

    sc = m.get("scene_conflict_examples") or []
    sc_html = "".join(
        f"<tr><td>{html.escape(str(e['ep']))}</td>"
        f"<td>{html.escape(str(e['char_a']))} / {html.escape(str(e['char_b']))}</td>"
        f"<td><code>{html.escape(str(e['voice']))}</code></td>"
        f"<td>{e['co_occurrence']}</td></tr>" for e in sc[:40])

    plan_rows = "".join(
        f"<tr><td>{e.i}</td><td>{_fmt(e.start)}–{_fmt(e.end)}</td>"
        f"<td><code>{html.escape(str(e.char))}</code></td>"
        f"<td><code>{html.escape(str(e.voice))}</code></td>"
        f"<td>{html.escape(e.hu[:90])}</td></tr>"
        for e in entries[:400])

    tc = m.get("textcheck") or {}
    tc_html = ""
    if tc:
        st_rows = "".join(
            f"<tr><td>{html.escape(str(r.get('src') or ''))}</td>"
            f"<td>{r.get('i')}</td><td>{_fmt(r.get('start'))}</td>"
            f"<td>{html.escape(str(r.get('before') or '')[:90])}</td>"
            f"<td>{html.escape(str(r.get('after') or '')[:90])}</td>"
            f"<td>{html.escape(str(r.get('errors') or '')[:90])}</td></tr>"
            for r in tc.get("rows", [])[:200])
        st = str(tc.get("llm_status") or "")
        st_cls = "ok" if st in ("ok", "disabled") else "bad"
        tc_html = (
            "<h2>Szövegminőség-ellenőrző pont — a hanggenerálás ELŐTT</h2>"
            f'<div class="sub">LLM: <span class="{st_cls}">'
            f"{html.escape(st)}</span> · "
            f"<code>{html.escape(str(tc.get('llm_base') or ''))}</code> / "
            f"<code>{html.escape(str(tc.get('llm_model') or ''))}</code> · "
            f"{tc.get('checked', 0)} sor ellenőrizve · "
            f"{tc.get('rule_fixed', 0)} szabály- + "
            f"{tc.get('override_fixed', 0)} felülírás + "
            f"{tc.get('override_reviewed', 0)} lezárt + "
            f"{tc.get('llm_fixed', 0)} LLM-javítás · "
            f"<b>{tc.get('remaining', 0)}</b> megoldatlan · "
            f"{tc.get('flagged', 0)} jelzés · {tc.get('seconds', 0)} s</div>"
            + (("<table><tr><th>forrás</th><th>#</th><th>kezdés</th>"
                "<th>előtte</th><th>utána</th><th>ok / hiba</th></tr>"
                + st_rows + "</table>") if st_rows
               else '<div class="sub">Nincs találat.</div>'))

    asr = m.get("asr") or {}
    asr_html = ""
    if asr:
        def _kind_cell(r: dict) -> str:
            k = str(r.get("kind") or "")
            if k == "real":
                cls = "bad" if float(r.get("cer", 0)) >= 0.5 else "warn"
                return f'<td class="{cls}">valódi</td>'
            if k == "loop":
                return '<td class="dim">hurok</td>'
            if k == "empty":
                return '<td class="dim">üres</td>'
            return '<td class="dim">—</td>'

        arows = "".join(
            f"<tr><td>{r.get('i')}</td><td>{_fmt(r.get('start'))}</td>"
            + _kind_cell(r) +
            f"<td class=\"{'bad' if float(r.get('cer', 0)) >= 0.5 else 'warn'}\">"
            f"{_fmt(r.get('cer'), 2)}</td>"
            f"<td>{html.escape(str(r.get('expected') or ''))}</td>"
            f"<td>{html.escape(str(r.get('heard') or ''))}</td>"
            f"<td>{html.escape(', '.join(str(w) for w in
                                        (r.get('missing') or [])))}</td></tr>"
            for r in asr.get("rows", [])[:300])
        a_cls = "ok" if float(asr.get("cer", 1)) < 0.15 else (
            "warn" if float(asr.get("cer", 1)) < 0.30 else "bad")
        n_real = int(asr.get("real", 0))
        n_loop = int(asr.get("loops", 0))
        n_empty = int(asr.get("empties", 0))
        confirm_txt = (f"{asr.get('resolved')} feloldva a Whisper-HU-val · "
                       if asr.get("resolved") else "")
        asr_html = (
            "<h2>ASR visszaellenőrzés — a kész MKV magyar hangja</h2>"
            f'<div class="sub">modell <code>'
            f"{html.escape(str(asr.get('model') or ''))}</code> · "
            f'össz-CER <b class="{a_cls}">{_fmt(asr.get("cer"), 3)}</b> · '
            f"WER <b>{_fmt(asr.get('wer'), 3)}</b> · "
            f"{asr.get('bad', 0)}/{asr.get('checked', 0)} sor gyanús "
            f"(CER ≥ {_fmt(asr.get('bad_cer', 0), 2)}) · "
            f"{confirm_txt}"
            f"<b>{n_real} valódi</b> · {n_loop} hurok · {n_empty} üres · "
            f"{asr.get('seconds', 0)} s · "
            f"{html.escape(str(asr.get('mkv') or ''))}</div>"
            "<div class=\"sub\">típusok: <b>valódi</b> = hallható eltérés a "
            "szándékolt szöveghez képest · <b>hurok</b> = whisper-ismétlés a "
            "zene alatt (felismerési hiba, nem hallható szöveg) · "
            "<b>üres</b> = nem hallott szót az ablak (kiment TTS vagy teljes "
            "elfedés). A lista a valódi találatokkal kezdődik.</div>"
            + (("<table><tr><th>#</th><th>kezdés</th><th>típus</th><th>CER</th>"
                "<th>szándékolt</th><th>hallott</th><th>hiányzó szavak</th>"
                "</tr>" + arows + "</table>") if arows
               else '<div class="sub">Nincs gyanús sor.</div>'))

    mix = m.get("mix") or {}
    # A képi beszélő-ellenőrzés bizonyítéka: ha van ``visual/asd.json``, bekerül
    # a riportba (és a report.json-ba) — kaput nem állít, döntést nem hoz.
    asd = m.get("asd") or asd_evidence(project, tag)
    if asd:
        m["asd"] = asd
    doc = f"""<!doctype html><html lang="hu"><meta charset="utf-8">
<title>dubstudio riport — {html.escape(tag)}</title><style>{CSS}</style>
<header><h1>dubstudio riport — {html.escape(tag)}</h1>
<div class="sub">{html.escape(project.name)} · S{project.season:02d} ·
{'MINDEN KAPU ÁTMENT' if m.get('passed') else 'VAN BUKÓ KAPU'} ·
{m.get('seconds', 0)} s feldolgozás · {html.escape(str(m.get('mkv') or ''))}
{' · <b class="warn">CSAK KEVERÉS (mix-only, nincs új szintézis)</b>' if m.get('mix_only') else ''}</div></header>
<main>
<div class="cards">{''.join(rows)}</div>
<h2>Kapuk</h2>
{'<div class="sub warn">Ez a futás csak újrakevert (mix-only): a TTS nem futott, ezért a hanghossz-illesztés (fit) kapu NEM MÉRT — az utolsó teljes szintézis mérése érvényes.</div>' if m.get('mix_only') else ''}
{''.join(gate_html)}
{tc_html}
<h2>Keverés</h2>
<table><tr><th>M&amp;E szint</th><th>M&amp;E normalizálás</th><th>dialógus szint</th>
<th>erősítés</th><th>mély ducking</th><th>mastering</th><th>eredmény</th>
<th>hossz</th></tr>
<tr><td>{_fmt(mix.get('me_lufs'), 1)} LUFS</td>
<td>{_fmt(mix.get('me_gain_db'), 1)} dB</td>
<td>{_fmt(mix.get('dialog_db'), 1)} dB</td>
<td>{_fmt(mix.get('gain_db'), 1)} dB</td>
<td>{_fmt(mix.get('duck_max_db'), 1)} dB ({mix.get('duck_frames', 0)} keret)</td>
<td>{_fmt(mix.get('makeup_db'), 1)} dB{' + limiter' if mix.get('limited') else ''}</td>
<td>{_fmt(mix.get('lufs'), 1)} LUFS (cél {_fmt(mix.get('lufs_target'), 1)}) /
{_fmt(mix.get('peak_db'), 1)} dBFS (plafon {_fmt(mix.get('peak_target_db'), 1)})</td>
<td>{_fmt(mix.get('duration'), 1)} s</td></tr></table>
{asr_html}
{asd_html(asd)}

<h2>Szereplők ({len(by_char)}) — a színészi hang gyakorisága</h2>
<table><tr><th>rang</th><th>szereplő</th><th>hang</th><th>epizód idő</th>
<th>évad idő</th><th>sor</th><th>szerep</th><th>réteg</th><th>arány</th><th>hang</th></tr>
{''.join(chars)}</table>
<div class="sub">főszereplő = az évad beszédidejének aránya ≥ {_fmt(100 * usage.get('main_share_min', 0), 0)} % "
— neki kizárólagos hang jár ({usage.get('n_main', 0)} fő; ebből "
{usage.get('n_main_f', 0)} nő / {usage.get('n_main_m', 0)} férfi); a nem fő "
(mellék/epizód/statiszta, oszthatnak hangot) száma: {usage.get('n_chars', 0) - usage.get('n_main', 0)} "
({usage.get('n_f', 0)} nő / {usage.get('n_m', 0)} férfi összesen). A hangkészlet "
{usage.get('n_pool', 0)} hang ({usage.get('n_pool_f', 0)} nő / {usage.get('n_pool_m', 0)} férfi), "
ebből használt: {usage.get('n_voices', 0)}.<br>Rétegek: fő {usage.get('tiers', {}).get('fő', 0)},
  mellék {usage.get('tiers', {}).get('mellék', 0)}, epizód {usage.get('tiers', {}).get('epizód', 0)},
  statiszta {usage.get('tiers', {}).get('statiszta', 0)}.</div>

<h2>Epizódonkénti beszédidő</h2>
<table><tr><th>epizód</th><th>szereplők</th><th>nő / férfi</th><th>főszereplő</th>
<th>beszédidő</th></tr>
{ep_rows}</table>

<h2>Hanggyakoriság — melyik hangot ki használja</h2>
<table><tr><th>hang</th><th>szereplő</th><th>beszédidő</th><th>arány</th>
<th>állapot</th><th>kik</th></tr>
{''.join(urows)}</table>

<h2>Hangváltás-hibák (ugyanaz a hang, más magyar hang)</h2>
{'<table><tr><th>cos</th><th>hang</th><th>előtte</th><th>utána</th></tr>' + sw_html + '</table>' if sw_html else '<div class="sub">Nincs ilyen hiba.</div>'}

<h2>Nem-ellentmondások</h2>
<div class="sub">A magyar hang neme ellentétes a szereplő rögzített nemével — ez casting-hiba.</div>
{'<table><tr><th>szereplő</th><th>hang</th><th>nem</th></tr>' + gc_html + '</table>' if gc_html else '<div class="sub">Nincs ilyen hiba.</div>'}
<div class="sub">A hang neme ellentétes a forrásbeszélő mért nemével — tudatos döntés lehet, ezért csak jelzés: <b>{m.get('gender_source_mismatch', 0)}</b></div>
{'<table><tr><th>szereplő</th><th>hang</th><th>nem</th><th>erős bizonyíték</th></tr>' + gs_html + '</table>' if gs_html else ''}

<h2>Egy jelenetben azonos hang</h2>
<div class="sub">azonos hang + azonos pitch: <b>{m.get('scene_conflict', 0)}</b> ·
azonos hang, de eltérő pitch: <b>{m.get('scene_same_voice', 0)}</b></div>
{'<table><tr><th>epizód</th><th>szereplők</th><th>hang</th><th>együtt-szólás</th></tr>' + sc_html + '</table>' if sc_html else '<div class="sub">Nincs azonos hang + pitch ütközés.</div>'}

<h2>Terv (első 400 sor)</h2>
<details open><table><tr><th>#</th><th>ablak</th><th>szereplő</th><th>hang</th><th>szöveg</th></tr>
{plan_rows}</table></details>
<div class="sub">A riport a méréseket tartalmazza; a <code>metrics</code> JSON ugyanitt:
<code>{html.escape(out.with_suffix('.json').name)}</code></div>
</main></html>"""
    out.write_text(doc, encoding="utf-8")
    out.with_suffix(".json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    return out
