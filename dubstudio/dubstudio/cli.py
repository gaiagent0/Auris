"""Parancssor: ``python -m dubstudio <parancs> [opciók]``.

A projektet a ``--project`` (projektfájl), a ``DUBSTUDIO_PROJECT`` környezeti
változó, vagy — ha a ``projects/`` mappában pontosan egy JSON van — maga a
parancs választja ki; beégetett projekt nincs a repóban.

Példák:
  python -m dubstudio voices                      # a hangok (13 + klónhang) és elérhetőségük
  python -m dubstudio prepare E05                 # feliratok + hang-identitás
  python -m dubstudio cluster E01 E02 ... E11     # évad-szintű szereplők + hangzár
  python -m dubstudio dub E05                     # szinkron elkészítése
  python -m dubstudio run E05 E06 --season        # prepare + cluster + dub
  python -m dubstudio narrate eloadas.mp4 --ref minta.wav   # egyszereplős narráció
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_project, resolve_project
from .pipeline import Pipeline
from .registry import Registry, recall_threshold, share_threshold
from .voices import CATALOG, VoiceBank


def _pipeline(args) -> Pipeline:
    project = load_project(args.project)
    return Pipeline(project, verbose=not getattr(args, "quiet", False))


def _tags(args) -> list[str]:
    return [t.upper() for t in args.episodes]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dubstudio", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default=None,
                    help="a projekt JSON (alapértelmezés: a projects/ egyetlen "
                         "projektje, illetve a DUBSTUDIO_PROJECT)")
    ap.add_argument("-q", "--quiet", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("voices", help="a hangkészlet és elérhetősége")

    sp = sub.add_parser("usage", help="hanggyakoriság: ki milyen gyakran beszél, "
                                      "kinek melyik hang (fő/mellék, kizárólagos)")

    sp = sub.add_parser("prepare", help="feliratok + hang-alapú identitás")
    sp.add_argument("episodes", nargs="+")

    sp = sub.add_parser("intake", help="nyers epizód előkészítése: magyar "
                                        "felirat + leválasztott vokál/M&E")
    sp.add_argument("episodes", nargs="+")
    sp.add_argument("--no-translate", dest="translate", action="store_false",
                    default=True, help="ne fordítson feliratot")
    sp.add_argument("--no-separate", dest="separate", action="store_false",
                    default=True, help="ne válassza szét a hangsávot")
    sp.add_argument("--force", action="store_true",
                    help="a meglévő feliratot/sávokat is készítse újra")
    sp.add_argument("--then-run", action="store_true",
                    help="az intake után indítsa a szinkron láncot is")
    sp.add_argument("--allow-fail", action="store_true",
                    help="a QA-bukás ellenére is fejezze be a szinkront")

    sp = sub.add_parser("youtube", help="YouTube URL letöltése, majd felirat + szinkron")
    sp.add_argument("episodes", nargs="+")
    sp.add_argument("--url", required=True, help="nyilvános YouTube URL")
    sp.add_argument("--source-lang", default=None,
                    help="a videó beszélt nyelve (alap: a projekt source_langja)")
    sp.add_argument("--then-run", action="store_true",
                    help="a letöltés után intake + szinkron is fusson")
    sp.add_argument("--allow-fail", action="store_true",
                    help="a QA-bukás ellenére is fejezze be a szinkront")

    sp = sub.add_parser("cluster", help="évad-szintű klaszterezés + hangzár")
    sp.add_argument("episodes", nargs="+")

    sp = sub.add_parser("recast", help="évadszintű újrakiosztás: főszereplő "
                                       "kizárólagos hangot kap, a mellékszereplők "
                                       "a maradékon osztoznak")
    sp.add_argument("episodes", nargs="*",
                    help="a bevonandó epizódok (alap: a kasztban szereplők)")

    sp = sub.add_parser("dub", help="szinkron elkészítése")
    sp.add_argument("episodes", nargs="+")
    sp.add_argument("--limit", type=int, default=0, help="csak az első N sor (próba)")
    sp.add_argument("--no-mux", action="store_true")
    sp.add_argument("--out", default=None, help="a kimeneti MKV neve")
    sp.add_argument("--allow-fail", action="store_true",
                    help="bukó QA kapu esetén is fejezze be")
    sp.add_argument("--mix-only", action="store_true",
                    help="csak újrakeverés a meglévő sor-WAV-okkal (TTS nélkül)")

    sp = sub.add_parser("run", help="prepare + cluster + dub")
    sp.add_argument("episodes", nargs="+")
    sp.add_argument("--season", action="store_true",
                    help="az összes előkészített epizódot bevonja a klaszterezésbe")
    sp.add_argument("--limit", type=int, default=0)
    sp.add_argument("--no-mux", action="store_true")
    sp.add_argument("--allow-fail", action="store_true")

    sp = sub.add_parser("narrate", help="egyszereplős videó narráció — külön lánc, "
                                        "szereplőkezelés és utómunka nélkül")
    sp.add_argument("source", help="videófájl vagy nyilvános YouTube URL")
    sp.add_argument("--out", default=None, help="a kimeneti MKV")
    sp.add_argument("--work", default=None,
                    help="munkamappa (alap: work/narration/<név>)")
    sp.add_argument("--voice", default=None,
                    help="hang azonosítója (pl. SUP_M1); --ref nélkül kész hang")
    sp.add_argument("--ref", default=None,
                    help="klónozandó hangminta WAV — ezzel XTTS klónhang szól")
    sp.add_argument("--lang", default="hu", help="a beszéd nyelve (alap: hu)")
    sp.add_argument("--srt", default=None, help="kézzel adott felirat (SRT)")
    sp.add_argument("--limit", type=int, default=0,
                    help="csak az első N blokk (próba)")
    sp.add_argument("--block-chars", type=int, default=320,
                    help="egy folyamatos blokk karakterkorlátja (alap: 320)")
    sp.add_argument("--block-max", type=float, default=26.0,
                    help="egy blokk maximális hossza másodpercben (alap: 26)")
    sp.add_argument("--gap", type=float, default=1.0,
                    help="ennél nagyobb szünetnél új blokk kezdődik")
    sp.add_argument("--models", type=Path, default=None,
                    help="modellek JSON-ja (opcionális, pl. xtts útvonal)")
    sp.add_argument("--replace-audio", dest="keep_original", action="store_false",
                    default=True, help="az eredeti hangsávot ne tartsa meg")
    sp.add_argument("--no-mux", action="store_true")
    sp.add_argument("--force", action="store_true",
                    help="a gyorsítótárat (felirat, blokkok) is újra készítse")

    sp = sub.add_parser("asr", help="a kész MKV magyar hangjának felismerése "
                                    "és WER/CER-összevetése a szándékolt "
                                    "szöveggel (időpontos hibalista)")
    sp.add_argument("episodes", nargs="+")
    sp.add_argument("--mkv", default=None,
                    help="az ellenőrizendő MKV (alap: a szokásos kimenet)")

    sp = sub.add_parser("words-audit", help="egy audit-riport szóidőinek ellenőrzése "
                                             "(szerkezet, energia, visszaolvasás)")
    sp.add_argument("episode")
    sp.add_argument("--start", type=float, default=None)
    sp.add_argument("--duration", type=float, default=None)
    sp.add_argument("--audit", type=Path, default=None, help="a JSON riport (alap: a kurzushoz tartozó)")

    sp = sub.add_parser("speaker-audit", help="kísérleti héber forrás-hang audit rövid mintán")
    sp.add_argument("episode")
    sp.add_argument("--start", type=float, default=30.0)
    sp.add_argument("--duration", type=float, default=45.0)
    sp.add_argument("--audio-stream", type=int, default=None)
    sp.add_argument("--segmentation-model", type=Path)
    sp.add_argument("--embedding-model", type=Path)
    sp.add_argument("--align-model", type=Path, default=None,
                    help="héber ONNX illesztőmodell csomag mappája")
    sp.add_argument("--no-align", action="store_true")
    sp.add_argument("--no-diarize", action="store_true")
    sp.add_argument("--no-campp", action="store_true")
    sp.add_argument("--no-identity", action="store_true",
                    help="ne kösse a fordulókat stabil C### karakterazonossághoz")
    sp.add_argument("--language", default="en", choices=("en", "he"),
                    help="a fő hang nyelve (en: angol hangsáv + ASR szóidő; héber csak felirat)")

    sp = sub.add_parser("voice-timbres",
                        help="a TTS-hangok beágyazása (hangszín-bizonyíték a "
                             "hangkiosztó döntési modellhez)")
    sp.add_argument("--out", default=None)

    sp = sub.add_parser("speaker-review", help="diarizált beszélő emberi döntése")
    sp.add_argument("episode")
    sp.add_argument("speaker", help="pl. SPEAKER_04")
    sp.add_argument("--decision", required=True,
                    choices=["accepted", "corrected", "rejected"])
    sp.add_argument("--char", default=None, help="C### azonosító")
    sp.add_argument("--gender", default=None, choices=["F", "M"])
    sp.add_argument("--voice", default=None, help="pl. SUP_M4")
    sp.add_argument("--note", default="")

    args = ap.parse_args(argv)
    if args.cmd == "narrate":
        import json as _json

        from . import narration

        models = None
        if args.models:
            try:
                models = _json.loads(Path(args.models).read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                print(f"[narráció] a modell-JSON nem olvasható: {exc}", file=sys.stderr)
                return 2
            models = models.get("models", models)
        opts = narration.Options(
            language=args.lang, block_chars=args.block_chars,
            block_max_s=args.block_max, gap_split=args.gap, limit=args.limit,
            keep_original=args.keep_original)
        try:
            narration.run(args.source, out=args.out, work=args.work,
                          voice=args.voice, ref=args.ref, srt=args.srt,
                          opts=opts, models=models, force=args.force,
                          no_mux=args.no_mux, verbose=not args.quiet)
        except Exception as exc:          # noqa: BLE001 — a hiba a felhasználóé
            print(f"[narráció] HIBA: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        return 0
    try:
        args.project = str(resolve_project(args.project))
    except FileNotFoundError as exc:
        print(f"[dubstudio] {exc}", file=sys.stderr)
        return 2

    if args.cmd == "voices":
        p = load_project(args.project)
        bank = VoiceBank(p, verbose=True)
        print(f"\n{'hang':>9} {'motor':>12} {'nem':>4}  fix paraméterek")
        for v in CATALOG:
            ok = "OK " if v.vid in bank.available else "HIÁNY"
            print(f"{v.vid:>9} {v.engine:>12} {v.gender:>4}  {ok} {v.params}")
        print(f"\nelérhető: {len(bank.available)}/{len(CATALOG)}")
        return 0

    if args.cmd == "usage":
        p = load_project(args.project)
        reg = Registry(p.cast_file, share_threshold(p.tuning), recall_threshold(p.tuning),
                     embed_model=p.model("embedding").name)
        if not reg.characters:
            print(f"nincs szereplő-rekord: {p.cast_file} (futtasd: cluster …)")
            return 1
        print(reg.table_usage())
        return 0

    if args.cmd == "recast":
        import json as _json

        p = load_project(args.project)
        reg = Registry(p.cast_file, share_threshold(p.tuning), recall_threshold(p.tuning),
                     embed_model=p.model("embedding").name)
        if not reg.characters:
            print(f"nincs szereplő-rekord: {p.cast_file} (futtasd: cluster …)")
            return 1
        tags = ([t.upper() for t in args.episodes]
                or sorted({ep for c in reg.characters for ep in c.episodes}))
        pipe = Pipeline(p, verbose=False)
        utts = []
        for t in tags:
            if (p.work / t / "utterances.json").exists():
                utts.extend(pipe.load_utterances([t]))
        if not utts:
            print("nincs előkészített epizód a kaszthoz", file=sys.stderr)
            return 1
        bank = VoiceBank(p, verbose=not args.quiet)
        reg.recast(utts, bank, verbose=True)
        reg.save()
        # a hangvisszaírás minden epizód-JSON-ba (a dub ezt olvassa)
        by_uid = {u.uid: u for u in utts}
        for t in tags:
            path = p.work / t / "utterances.json"
            if not path.exists():
                continue
            data = _json.loads(path.read_text(encoding="utf-8"))
            for d in data:
                u = by_uid.get(d.get("uid"))
                if u is not None:
                    d["char"], d["voice"] = u.char, u.voice
            path.write_text(_json.dumps(data, ensure_ascii=False, indent=1),
                            encoding="utf-8")
        print(reg.table_usage())
        return 0

    if args.cmd == "words-audit":
        import json as _json

        from .words import report as _report
        from .words import verify_audit as _verify_audit

        project = load_project(args.project)
        tag = args.episode.upper()
        audit_path = args.audit
        if audit_path is None:
            out_dir = project.ep_dir(tag) / "source_audit"
            if args.start is not None and args.duration is not None:
                audit_path = out_dir / f"audit_{args.start:.0f}_{args.duration:.0f}.json"
                if not audit_path.is_file():
                    found = sorted(out_dir.glob(
                        f"audit_*_{args.start:.0f}_{args.duration:.0f}.json"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
                    if found:
                        audit_path = found[0]
            else:
                found = sorted(out_dir.glob("audit_*.json"),
                               key=lambda p: p.stat().st_mtime, reverse=True)
                if not found:
                    print(f"nincs audit-riport itt: {out_dir}", file=sys.stderr)
                    return 1
                audit_path = found[0]
        if not audit_path.is_file():
            print(f"nincs ilyen audit-riport: {audit_path}", file=sys.stderr)
            return 1
        audit = _json.loads(audit_path.read_text(encoding="utf-8"))
        print(f"[words] riport: {audit_path}")
        try:
            result = _verify_audit(project, tag, audit)
        except Exception as exc:  # noqa: BLE001 — a hiba oka a felhasználónak kell
            print(f"[words] HIBA: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        print(_report(result))
        if result.get("path"):
            print(f"[words] eredmény: {result['path']}")
        return 0 if result.get("verdict") != "fail" else 3

    if args.cmd == "speaker-audit":
        from .speaker_audit import run as run_speaker_audit
        project = load_project(args.project)
        try:
            result = run_speaker_audit(
                project, args.episode, args.start, args.duration,
                audio_stream=args.audio_stream,
                segmentation_model=args.segmentation_model,
                embedding_model=args.embedding_model, align_model=args.align_model,
                align=not args.no_align, diarize=not args.no_diarize,
                campp=not args.no_campp, identity_link=not args.no_identity,
                language=args.language)
        except Exception as exc:
            print(f"[source-audit] HIBA: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        print(f"[source-audit] riport: {result}")
        return 0

    if args.cmd == "voice-timbres":
        from .voice_decision import measure_voice_timbres
        project = load_project(args.project)
        payload = measure_voice_timbres(project, args.out)
        print(f"[hangszín] {len(payload['voices'])} hang beágyazva")
        return 0

    if args.cmd == "speaker-review":
        from . import review as review_mod
        project = load_project(args.project)
        review_mod.set_speaker(project, args.episode, args.speaker, args.decision,
                               char_id=args.char, gender=args.gender,
                               voice=args.voice, note=args.note)
        summary = review_mod.summary(review_mod.load(project, args.episode))
        print(f"[review] {args.speaker}: {args.decision} "
              f"({summary['accepted']} elfogadva, {summary['corrected']} javítva, "
              f"{summary['rejected']} elutasítva)")
        return 0

    pipe = _pipeline(args)
    tags = _tags(args)

    if args.cmd == "prepare":
        for t in tags:
            pipe.prepare(t)
        return 0

    if args.cmd == "intake":
        from .intake import ensure_ready

        for tag in tags:
            print(f"[intake] {tag}", flush=True)
            try:
                ensure_ready(pipe.p, tag, translate=args.translate,
                             separate=args.separate, force=args.force)
            except Exception as exc:      # noqa: BLE001 — a napló a felhasználóé
                print(f"[intake] HIBA ({tag}): {type(exc).__name__}: {exc}",
                      file=sys.stderr, flush=True)
                return 2
            if args.then_run:
                season = sorted(p.name[:3] for p in pipe.p.work.glob("*/utterances.json"))
                pipe.run(tag, season_tags=season or None,
                         allow_fail=args.allow_fail)
        return 0

    if args.cmd == "youtube":
        from .intake import ensure_ready
        from .youtube import download

        for tag in tags:
            print(f"[youtube] {tag}", flush=True)
            try:
                result = download(pipe.p, tag, args.url,
                                  source_lang=args.source_lang or pipe.p.source_lang)
            except Exception as exc:      # noqa: BLE001 — a napló a felhasználóé
                print(f"[youtube] HIBA ({tag}): {type(exc).__name__}: {exc}",
                      file=sys.stderr, flush=True)
                return 2
            print(f"[youtube] kész: {result.get('title') or result.get('media')}",
                  flush=True)
            if args.then_run:
                try:
                    ensure_ready(pipe.p, tag)
                except Exception as exc:  # noqa: BLE001
                    print(f"[youtube] intake HIBA ({tag}): "
                          f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                    return 2
                season = sorted(p.name[:3] for p in pipe.p.work.glob("*/utterances.json"))
                pipe.run(tag, season_tags=season or None,
                         allow_fail=args.allow_fail)
        return 0

    if args.cmd == "cluster":
        pipe.cluster_season(tags)
        return 0

    if args.cmd == "dub":
        for t in tags:
            pipe.dub(t, limit=args.limit, no_mux=args.no_mux,
                     out_name=args.out, allow_fail=args.allow_fail,
                     mix_only=getattr(args, "mix_only", False))
        return 0

    if args.cmd == "run":
        season = None
        if args.season:
            season = sorted(p.name[:3] for p in pipe.p.work.glob("*/utterances.json"))
        for i, t in enumerate(tags):
            pipe.run(t, season_tags=season, limit=args.limit, no_mux=args.no_mux,
                     allow_fail=args.allow_fail)
        return 0

    if args.cmd == "asr":
        for t in tags:
            pipe.check_asr(t, mkv=args.mkv)
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
