"""A teljes lánc: előkészítés -> klaszterezés -> terv -> szintézis -> keverés.

Lépések (mindegyik külön is futtatható a CLI-ból):

  prepare  – feliratok + hang-alapú beszélő-embedding + nem + megnyilatkozások
  cluster  – évad-szintű klaszterezés és szereplő-rekord (hangzár)
  dub      – terv, szintézis, keverés, mux, QA-riport

Az ``prepare`` eredménye epizódonként egy JSON (``work/<ep>/utterances.json``),
benne az embeddingekkel — így a klaszterezés a teljes évadon **egyszer**
futhat, nem epizódonként újra.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from . import (asrcheck, audio, clustering, diarization, linereview, mix, media,
               plan, qa, report, subs, textcheck, utterances, word_assignment)
from .config import Project
from .embeddings import SpeakerEmbedder
from .gender import GenderClassifier
from .registry import Registry, recall_threshold, share_threshold
from .synth import Synthesizer
from .utterances import Utterance
from .voices import VoiceBank


def _cache_key(tag: str, cues: list[subs.Cue], model: str = "") -> str:
    """Gyorsítótár-kulcs; a hanglenyomat-modell is része, különben egy
    modellcsere után a régi (rossz) vektorok maradnának a lemezen."""
    name = Path(str(model)).name if model else ""
    return f"v2:{tag}:{len(cues)}:{round(sum(c.start for c in cues), 1)}:{name}"


class Pipeline:
    def __init__(self, project: Project, verbose: bool = True):
        self.p = project
        self.verbose = verbose
        self._embedder: SpeakerEmbedder | None = None
        self._gender: GenderClassifier | None = None
        self._bank: VoiceBank | None = None

    # ── lusta erőforrások ────────────────────────────────────────────────
    @property
    def embedder(self) -> SpeakerEmbedder:
        if self._embedder is None:
            self._embedder = SpeakerEmbedder(self.p.model("embedding"))
        return self._embedder

    @property
    def gender(self) -> GenderClassifier:
        if self._gender is None:
            mid = self.p.models.get("gender_model")
            self._gender = (GenderClassifier(str(mid)) if mid
                            else GenderClassifier())
        return self._gender

    @property
    def bank(self) -> VoiceBank:
        if self._bank is None:
            self._bank = VoiceBank(self.p, verbose=self.verbose)
        return self._bank

    # ── 1. előkészítés ───────────────────────────────────────────────────
    def _cue_embeddings(self, tag: str, cues, x, sr) -> list:
        path = self.p.ep_dir(tag) / "cue_emb.npz"
        key = _cache_key(tag, cues, self.p.model("embedding"))
        if path.exists():
            z = np.load(path, allow_pickle=True)
            if str(z["key"]) == key:
                return [None if v is None else np.asarray(v, dtype=np.float32)
                        for v in z["vecs"]]
        spans = [(c.start, c.end) for c in cues]
        got = self.embedder.embed_windows(x, sr, spans)
        np.savez(path, key=key,
                 vecs=np.array([None if v is None else v for v in got], dtype=object))
        if self.verbose:
            print(f"[{tag}] {sum(1 for v in got if v is not None)}/{len(cues)} "
                  f"sor hanglenyomata kész")
        return got

    def _cue_genders(self, tag: str, cues, x, sr) -> list:
        path = self.p.ep_dir(tag) / "cue_gender.npz"
        key = _cache_key(tag, cues)
        if path.exists():
            z = np.load(path, allow_pickle=True)
            if str(z["key"]) == key:
                return [None if v < 0 else float(v) for v in z["p"]]
        spans = [(c.start, c.end) for c in cues]
        got = self.gender.measure_windows(x, sr, spans)
        np.savez(path, key=key,
                 p=np.array([-1.0 if v is None else v for v in got], dtype=np.float32))
        if self.verbose:
            print(f"[{tag}] {sum(1 for v in got if v is not None)}/{len(cues)} "
                  f"sor nem-mérése kész")
        return got

    def prepare(self, tag: str) -> list[Utterance]:
        """Egy epizód feldolgozása a klaszterezés előtt."""
        t0 = time.time()
        hun_path = self.p.subtitle(tag, "hun")
        if not hun_path.exists():
            raise FileNotFoundError(f"nincs magyar felirat: {hun_path}")
        cues = [c for c in subs.parse_srt(hun_path)
                if subs.one_line(c.text)
                and not any(k in c.text.lower() for k in subs.SKIP_PATTERNS)]
        # A zene/dalszöveg/háttérhang sorok **megmaradnak** a sorlistában (ez
        # tartja stabilan a megnyilatkozás-azonosítókat és a kézi
        # felülvizsgálat kulcsait), de `music=True` jelöléssel: nem kapnak
        # szereplőt, hangot, beszédidőt és nem mennek a TTS-be.
        music = [c for c in cues if subs.is_music(c.text)]
        for c in music:
            c.music = True
        eng_path = self.p.subtitle(tag, "eng")
        eng = subs.parse_srt(eng_path) if eng_path.exists() else []

        vocals = self.p.vocals(tag)
        x, sr = audio.load_mono(vocals)
        # A teljes sávot **egyszer** mintavételezzük 16 kHz-re: a kétbeszélős
        # sorok fordulónkénti kivágása több száz rövid ablakot kér, és a
        # hívásonkénti újramintavételezés percig tartott.
        x16 = audio.to_16k(x, sr)
        if self.verbose:
            print(f"[{tag}] {len(cues)} feliratsor, énekbeszéd-sáv {len(x) / sr / 60:.1f} perc")
            if music:
                print(f"[{tag}] {len(music)} zenei/háttérhang-sor kihagyva "
                      f"(nem dialógus): "
                      + ", ".join(f"#{c.index}" for c in music))

        embs = self._cue_embeddings(tag, cues, x, sr)
        pgen = self._cue_genders(tag, cues, x, sr)

        def embed_span(a: float, b: float):
            core = audio.speech_core(x16, audio.SR16, a, b)
            return None if core is None else self.embedder.embed(core)

        def gender_span(a: float, b: float):
            """Kétbeszélős sorból kivágott forduló saját nem-mérése."""
            return self.gender.measure_windows(x16, audio.SR16, [(a, b)])[0]

        cfg = diarization.settings(self.p, tag)
        turns: list[dict] = []
        if cfg.get("enabled"):
            segments = diarization.run(self.p, tag, verbose=self.verbose)
            turns = [{"start": s.start, "end": s.end,
                      "speaker": str(s.speaker)} for s in segments]
            labels = diarization.assign(cues, segments,
                                        float(cfg.get("attach_gap", 2.0)))
            utts = utterances.build_diarized(
                tag, cues, labels, embs, pgen, embed_span=embed_span,
                tuning=self.p.tuning, en_cues=eng,
                gender_span=gender_span,
                label_span=lambda a, b: diarization.label_span(
                    segments, a, b, float(cfg.get("attach_gap", 2.0))),
                # A kétbeszélős sorok diarizáció-alapú szétvágásához és a
                # „egy hangon két ember” kapuhoz kellenek a nyers fordulók.
                runs_span=lambda a, b: [
                    (s.speaker, s.start, s.end) for s in segments
                    if min(b, s.end) - max(a, s.start) > 0.0])
            # A diarizáció alatt nincs beszéd (képernyőszöveg, zene): a
            # szomszéd fordulóhoz csatoljuk, hogy ne maradjon hang nélkül.
            seq = sorted(utts, key=lambda u: u.start)
            for i, u in enumerate(seq):
                if (u.cluster is not None or u.music
                        or subs.is_screen_text(u.text)):
                    continue
                prev = next((v for v in reversed(seq[:i]) if v.cluster is not None), None)
                nxt = next((v for v in seq[i + 1:] if v.cluster is not None), None)
                if prev is not None and (nxt is None
                                         or u.start - prev.end <= nxt.start - u.end):
                    u.cluster = prev.cluster
                elif nxt is not None:
                    u.cluster = nxt.cluster
            self.diarization_info = diarization.validity(segments)
        else:
            utts = utterances.build(tag, cues, embs, pgen, embed_span=embed_span,
                                    tuning=self.p.tuning, en_cues=eng,
                                    gender_span=gender_span)
            self.diarization_info = {}
        self.save_utterances(tag, utts)
        # ── szó-szintű vágásjelöltek (itt, mert a fordulók már a memóriában
        # vannak): a `dub` innen olvassa, nem futtat diarizációt csak ezért.
        labeled = word_assignment.segments_by_label(utts, turns)
        word_assignment.save_marks(self.p, tag, labeled)
        if self.verbose:
            print(f"[{tag}] szóhatáron vágható feliratsor: {len(labeled)} "
                  f"(word_marks.json)")
            n_switch = sum(1 for u in utts if len(u.cues) > 1)
            print(f"[{tag}] {len(utts)} megnyilatkozás "
                  f"({n_switch} több feliratsorból, {len(cues)} sorból) "
                  f"— {time.time() - t0:.0f}s")
        return utts

    def save_utterances(self, tag: str, utts: list[Utterance]) -> Path:
        path = self.p.ep_dir(tag) / "utterances.json"
        path.write_text(json.dumps([u.to_dict() for u in utts],
                                   ensure_ascii=False, indent=1), encoding="utf-8")
        return path

    def load_utterances(self, tags: list[str]) -> list[Utterance]:
        out: list[Utterance] = []
        for tag in tags:
            path = self.p.ep_dir(tag) / "utterances.json"
            if not path.exists():
                raise FileNotFoundError(f"nincs előkészített epizód: {path} "
                                        f"(futtasd: prepare {tag})")
            for d in json.loads(path.read_text(encoding="utf-8")):
                out.append(Utterance.from_dict(d))
        return out

    # ── 2. klaszterezés + szereplők ──────────────────────────────────────
    def cluster_season(self, tags: list[str], verbose: bool | None = None
                       ) -> tuple[dict, Registry]:
        v = self.verbose if verbose is None else verbose
        utts = self.load_utterances(tags)
        # névvédelem: a sorok uid-je stabil, a klaszter-azonosító nem. A
        # klaszterezés előtti besorolást elmentjük, hogy a nevek és a kézi
        # markok a megfelelő új szereplőre kerülhessenek át (linereview.carry).
        chars_before: dict[str, dict[str, str]] = {}
        for tag in tags:
            path = self.p.ep_dir(tag) / "utterances.json"
            if not path.exists():
                continue
            chars_before[tag] = {
                str(d["uid"]): str(d["char"])
                for d in json.loads(path.read_text(encoding="utf-8"))
                if d.get("char")}
        if v:
            print(f"[klaszterezés] {len(utts)} megnyilatkozás "
                  f"{len(tags)} epizódból")
        if (self.p.diarization or {}).get("enabled") and any(
                u.cluster is not None for u in utts):
            # Keresztvalidált PLDA a cohort-ból (beszélő-címkék = klaszterek):
            # minden vizsgált klaszterpár olyan modellt kap, amely egyik
            # klasztert sem látta tanításkor (nincs önigazolás).
            plda_model = clustering.plda_crossfit(utts)
            if v and plda_model is not None:
                print(f"[PLDA] {plda_model.describe()}")
            clusters = clustering.from_diarization(
                utts, self.p.tuning, verbose=v, plda_model=plda_model)
        else:
            clusters = clustering.cluster(utts, self.p.tuning, verbose=v)
        reg = Registry(self.p.cast_file, share_threshold(self.p.tuning),
                                    recall_threshold(self.p.tuning),
                                    embed_model=self.p.model("embedding").name)
        reg.assign(clusters, utts, self.bank, verbose=v)
        # **Évadszintű újrakiosztás:** a kaszt többi epizódját is bevonjuk a
        # jelenetgráfba, hogy a főszereplő hangja ne maradjon megosztott, és a
        # mellékszereplők se kapjanak epizódonként más hangot.
        season_utts = list(utts)
        extra = sorted({ep for c in reg.characters for ep in c.episodes}
                       - {u.ep for u in utts})
        for ep in extra:
            if (self.p.work / ep / "utterances.json").exists():
                season_utts.extend(self.load_utterances([ep]))
        reg.recast(season_utts, self.bank, verbose=v)
        reg.save()
        if v:
            print(reg.table())
            print(f"kiírva: {self.p.cast_file}")
        # a szereplő/hang visszaírása az epizód-JSON-okba
        for tag in tags:
            mine = [u for u in utts if u.ep == tag]
            cluster_by_uid = {u.uid: u.cluster for u in mine}
            voices = {u.uid: (u.char, u.voice) for u in mine}
            path = self.p.ep_dir(tag) / "utterances.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            for d in data:
                d["cluster"] = cluster_by_uid.get(d["uid"])
                ch, vo = voices.get(d["uid"], (None, None))
                d["char"], d["voice"] = ch, vo
            path.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                            encoding="utf-8")
            # a nevek/markok átvitele az új azonosítókra (emberi döntés védelme)
            after = {str(d["uid"]): str(d["char"]) for d in data if d.get("char")}
            review = linereview.load(self.p, tag)
            remap = linereview.carry(
                review, chars_before.get(tag, {}), after,
                genders={c.char_id: c.gender for c in reg.characters})
            if remap:
                linereview.save(self.p, tag, review)
                if v:
                    print(f"[{tag}] névvédelem: {len(remap)} szereplő neve/markja "
                          f"átkerült az új azonosítóra")
        self._last_clusters = clusters
        return clusters, reg

    # ── 3. szinkron elkészítése ──────────────────────────────────────────
    def _pitch_map(self) -> dict[str, float]:
        """Szereplő -> fix hangszín-eltolás a hangzárból (a tervhez)."""
        try:
            reg = Registry(self.p.cast_file, share_threshold(self.p.tuning),
                                    recall_threshold(self.p.tuning),
                                    embed_model=self.p.model("embedding").name)
        except (OSError, ValueError):
            return {}
        return {c.char_id: float(c.pitch or 0.0) for c in reg.characters}

    def _voice_map(self) -> dict[str, str]:
        """Szereplő -> hang a hangzárból.

        A klaszterezés a megnyilatkozásokba is beírja a hangot, de a hangzár
        későbbi (emberi) módosítása csak itt érvényesül: enélkül a kész
        szinkron a régi hanggal szólna, hiába javítottuk a castot.
        """
        try:
            reg = Registry(self.p.cast_file, share_threshold(self.p.tuning),
                                    recall_threshold(self.p.tuning),
                                    embed_model=self.p.model("embedding").name)
        except (OSError, ValueError):
            return {}
        return {c.char_id: c.voice for c in reg.characters if c.voice}

    def _clusters_from_cast(self, utts: list[Utterance],
                            registry: Registry | None = None) -> dict:
        """Klaszter-információ a mentett szereplő-rekordokból (dub lépéshez).

        A QA/riport nem csak a hosszt nézi: a klaszter nemét és epizódjait is
        a szereplő-rekordból töltjük ki, különben minden klaszter "bizonytalan"
        és "egyepizódos" lenne a riportban.
        """
        durs: dict[int, float] = {}
        char_of: dict[int, str] = {}
        eps: dict[int, set] = {}
        for u in utts:
            if u.cluster is None:
                continue
            durs[u.cluster] = durs.get(u.cluster, 0.0) + u.dur
            if u.char:
                char_of.setdefault(u.cluster, u.char)
            eps.setdefault(u.cluster, set()).add(u.ep)
        gender_of = ({c.char_id: c.gender for c in registry.characters}
                     if registry else {})
        out = {}
        for cid, dur in durs.items():
            cl = clustering.Cluster(cid=cid, dur=dur,
                                    episodes=sorted(eps.get(cid, [])))
            cl.gender = gender_of.get(char_of.get(cid, ""))
            out[cid] = cl
        return out

    def dub(self, tag: str, limit: int = 0, no_mux: bool = False,
            out_name: str | None = None, allow_fail: bool = False) -> dict:
        t0 = time.time()
        self._asd_metrics = None
        utts = self.load_utterances([tag])
        if not any(u.cluster is not None for u in utts):
            raise RuntimeError(f"a(z) {tag} epizód nincs besorolva — futtasd: "
                               f"cluster {tag}")
        # a hangzár az irányadó: a megnyilatkozásba írt hang elavulhat
        voices = self._voice_map()
        # kézi hang a teljes epizódra (a felület „Magyar hang” választója):
        # egyszemélyes YouTube-videóhoz, ahol nincs értelme szereplőnként dönteni
        tn = self.p.tuning or {}
        override = str(tn.get("voice_override") or "").strip()
        if override:
            from .voices import BY_ID
            if override in BY_ID:
                voices = {str(u.char): override for u in utts if u.char}
                if self.verbose:
                    print(f"[hang] kézi hang az egész epizódra: {override}")
            elif self.verbose:
                print(f"[hang] ismeretlen kézi hang ({override}) — "
                      f"marad az automatikus kiosztás")
        for u in utts:
            if u.char and voices.get(str(u.char)):
                u.voice = voices[str(u.char)]
        # a felülvizsgáló kézi szereplő-döntései (review/lines.json): a teljes
        # sorra szóló mark felülírja a hangot, a szótartományos mark kettévágja
        labels = linereview.labels(self.p, tag)
        pitch_of = self._pitch_map()
        for cid, person in (linereview.load(self.p, tag).get("people") or {}).items():
            person = person or {}
            if person.get("voice"):
                voices.setdefault(str(cid), str(person["voice"]))
            try:
                pitch_of.setdefault(str(cid), float(person.get("pitch") or 0.0))
            except (TypeError, ValueError):
                pass
        vocals = self.p.vocals(tag)
        x, sr = audio.load_mono(vocals)
        # szó-szintű hozzárendelés: a `prepare`-ben mentett szóhatár-jelöltek
        # vágják a sort (a kézi markot nem írják felül)
        word_marks = word_assignment.marks_for_project(self.p, tag, utts)
        if self.verbose:
            if word_marks:
                print(f"[szó-hozzárendelés] {len(word_marks)} feliratsor "
                      f"vágandó szóhatáron")
            elif not word_assignment.marks_path(self.p, tag).is_file():
                print(f"[szó-hozzárendelés] nincs szóhatár-jelölt ({tag}) — "
                      f"futtasd: prepare {tag}")
        entries = plan.build(utts, x, sr, verbose=self.verbose,
                             pitch_of=pitch_of, voices=voices,
                             offsets=linereview.offsets(self.p, tag),
                             labels=labels, auto_marks=word_marks)

        # ellenőrző pont a hanggenerálás ELŐTT: magyar szövegminőség
        # (szabályok + helyi LLM) — a javított szöveg megy a TTS-be
        tc_stats = textcheck.run(entries, self.p, verbose=self.verbose)

        tts_dir = self.p.ep_dir(tag) / "tts"
        synth = Synthesizer(self.p, self.bank)
        stats = synth.run(entries, tts_dir, limit=limit, verbose=self.verbose)

        me = self.p.me_bed(tag)
        dub_wav = self.p.ep_dir(tag) / "dub.wav"
        mix_info = mix.build_track(
            entries, tts_dir, me, dub_wav, verbose=self.verbose,
            duck_db=float(tn.get("duck_db", mix.DUCK_DB)),
            over_bed_db=float(tn.get("dialogue_over_bed_db", mix.DIALOG_OVER_BED_DB)),
            bed_db=float(tn.get("bed_db", 0.0) or 0.0),
            speech_duck_db=float(tn.get("bed_speech_duck_db", 0.0) or 0.0),
            orig_vocals=self.p.vocals(tag))

        mkv = media.mux(self.p, tag, dub_wav,
                        out_name=out_name or media.safe_name(tag, self.p),
                        skip=no_mux)

        # médiaszerver: a kész szinkron visszamásolása az eredeti fájl helyére
        deliver_info = None
        if mkv is not None and (self.p.media_server or {}).get("deliver"):
            from . import deliver
            try:
                deliver_info = deliver.copy_back(self.p, tag, mkv)
                if self.verbose:
                    print(f"[médiaszerver] visszamásolva: {deliver_info['target']}")
            except FileNotFoundError as exc:
                if self.verbose:
                    print(f"[médiaszerver] nincs visszamásolás: {exc}")
            except OSError as exc:
                print(f"[médiaszerver] a visszamásolás nem sikerült: {exc}",
                      file=sys.stderr)

        reg = Registry(self.p.cast_file, share_threshold(self.p.tuning),
                                    recall_threshold(self.p.tuning),
                                    embed_model=self.p.model("embedding").name)
        clusters = self._clusters_from_cast(utts, reg)
        # képi ASD evidence alkalmazása (ha van visual/asd.json)
        try:
            from . import asd_integration
            asd_rows = asd_integration.load_asd_results(self.p.ep_dir(tag))
            if asd_rows:
                n = asd_integration.apply_asd_evidence(utts, asd_rows)
                if self.verbose:
                    print(f"[ASD] {n} megnyilatkozás kapott képi evidence-t")
                self._asd_metrics = asd_integration.asd_metrics(utts)
        except Exception as exc:
            if self.verbose:
                print(f"[ASD] evidence alkalmazás sikertelen: {exc}")
        # szereplő-azonosság: a többi epizód ugyanazon klaszterének mintái is
        context = []
        for ep in [f"E{i:02d}" for i in range(1, 12)]:
            if ep == tag:
                continue
            if (self.p.ep_dir(ep) / "utterances.json").exists():
                context.extend(self.load_utterances([ep]))
        metrics = qa.evaluate(self.p, tag, utts, entries, tts_dir, clusters,
                              reg, synth_stats=stats, text_stats=tc_stats,
                              context_utts=context)
        if getattr(self, "_asd_metrics", None):
            metrics["asd"] = self._asd_metrics
        metrics["mix"] = mix_info
        metrics["synth"] = {k: v for k, v in stats.items() if k != "fits"}
        metrics["episode"] = tag
        metrics["seconds"] = round(time.time() - t0, 1)
        metrics["dub_wav"] = str(dub_wav)
        metrics["mkv"] = None if mkv is None else str(mkv)
        metrics["delivered"] = deliver_info
        rp = report.write(self.p, tag, metrics, utts, entries, clusters, reg)
        if self.verbose:
            qa.print_report(metrics)
            print(f"[riport] {rp}")
            print(f"[kész] {tag} — {metrics['seconds']:.0f}s")
        if not metrics["passed"] and not allow_fail:
            raise SystemExit(f"QA kapu nem teljesült ({tag}) — lásd a riportot: {rp}")
        return metrics

    # ── a kész MKV vissza-ellenőrzése (ASR) ──────────────────────────────
    def check_asr(self, tag: str, mkv: str | None = None) -> dict:
        """A kész MKV magyar hangjának felismerése és WER/CER-összevetése.

        A szándékolt feliratszöveget (a terv sorait) vetjük össze azzal, amit
        a magyar Whisper-finetune *ténylegesen hall* a kész fájl magyar
        sávján — időpontos hibalistával. Az eredmény a meglévő riportba kerül.
        """
        utts = self.load_utterances([tag])
        if not any(u.cluster is not None for u in utts):
            raise RuntimeError(f"a(z) {tag} epizód nincs besorolva — futtasd: "
                               f"cluster {tag}")
        x, sr = audio.load_mono(self.p.vocals(tag))
        voices = self._voice_map()
        pitch_of = self._pitch_map()
        for cid, person in (linereview.load(self.p, tag).get("people") or {}).items():
            person = person or {}
            if person.get("voice"):
                voices.setdefault(str(cid), str(person["voice"]))
            try:
                pitch_of.setdefault(str(cid), float(person.get("pitch") or 0.0))
            except (TypeError, ValueError):
                pass
        entries = plan.build(utts, x, sr, verbose=self.verbose,
                             pitch_of=pitch_of, voices=voices,
                             offsets=linereview.offsets(self.p, tag),
                             labels=linereview.labels(self.p, tag),
                             auto_marks=word_assignment.marks_for_project(
                                 self.p, tag, utts))
        mkv_path = Path(mkv) if mkv else self.p.out / media.safe_name(tag, self.p)
        if not mkv_path.exists():
            raise FileNotFoundError(f"nincs kész MKV: {mkv_path} "
                                    f"(futtasd: dub {tag})")
        stats = asrcheck.check(self.p, tag, entries, mkv_path,
                               verbose=self.verbose)
        rp_json = self.p.ep_dir(tag) / "report.json"
        metrics = (json.loads(rp_json.read_text(encoding="utf-8"))
                   if rp_json.exists() else {"episode": tag})
        metrics["asr"] = stats
        metrics = asrcheck.apply_gate(metrics, stats, self.p)
        reg = Registry(self.p.cast_file, share_threshold(self.p.tuning),
                                    recall_threshold(self.p.tuning),
                                    embed_model=self.p.model("embedding").name)
        clusters = self._clusters_from_cast(utts, reg)
        rp = report.write(self.p, tag, metrics, utts, entries, clusters, reg)
        if self.verbose:
            print(f"[riport] {rp}")
        return metrics

    # ── magas szintű parancsok ───────────────────────────────────────────
    def run(self, tag: str, season_tags: list[str] | None = None,
            limit: int = 0, no_mux: bool = False, allow_fail: bool = False) -> dict:
        self.prepare(tag)
        others = [t for t in (season_tags or []) if t != tag
                  and (self.p.ep_dir(t) / "utterances.json").exists()]
        self.cluster_season([tag] + others)
        return self.dub(tag, limit=limit, no_mux=no_mux, allow_fail=allow_fail)
