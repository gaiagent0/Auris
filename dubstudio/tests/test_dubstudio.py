"""Egységtesztek a dubstudio magjához (hálózat és GPU nélkül futnak).

Futtatás:
  ../reader/.venv/Scripts/python.exe -m unittest discover -s tests -v
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dubstudio import (asrcheck, asd_integration, clustering, identity, linereview, mix,  # noqa: E402
                       plan, plda, registry, review, subs, speaker_audit, synth,
                       textcheck, utterances, voice_decision, voices, words,
                       word_assignment)


def _cue(i, a, b, text):
    return subs.Cue(index=i, start=a, end=b, text=text)


def _v(seed: int, dim: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=dim).astype(np.float32)
    return x / np.linalg.norm(x)


class TestVocative(unittest.TestCase):
    """Szöveg-alapú megszólítás: a rövid jelenet-töredék nem új szereplő."""

    @staticmethod
    def _utt(uid, start, dur, text, cluster, p_female):
        u = utterances.Utterance(uid=uid, ep="E01", start=start,
                                 end=start + dur, text=text)
        u.cluster, u.p_female = cluster, p_female
        return u

    def test_vocative_forms(self):
        self.assertEqual(subs.vocative("Nem tudod a teljes történetet, Mark."),
                         "Mark")
        self.assertEqual(subs.vocative("Hé, Claire, ne mondd ezt!"), "Claire")
        self.assertEqual(subs.vocative("Claire, lányom."), "Claire")
        self.assertEqual(subs.vocative("Garner nővér?"), "Garner")
        self.assertEqual(subs.vocative("Kicsim."), "Kicsim")
        self.assertEqual(subs.vocative("Mark?"), "Mark")

    def test_weak_forms_need_a_known_name(self):
        known = (subs.vocative_names("Hé, Claire, ne mondd ezt!")
                 | subs.vocative_names("Átveszem a helyed, Claire."))
        self.assertEqual(known, {"Claire"})
        self.assertEqual(subs.vocative("Claire, lányom.", known), "Claire")
        self.assertEqual(subs.vocative("Mark?", {"Mark"}), "Mark")
        # mondatkezdő szó / egyszavas felkiáltás nem bizonyított név
        self.assertIsNone(subs.vocative("Emlékszel, amikor segített nekünk.",
                                        known))
        self.assertIsNone(subs.vocative("Öhm, 5-kor indul a gépem, oké?", known))
        self.assertIsNone(subs.vocative("Gyere!", known))

    def test_non_vocatives(self):
        for text in ("Anna elvesztése volt a legnehezebb dolog.",
                     "Vége.",
                     "Nos, épp birtokháborítást követ el.",
                     "Ó, Istenem, micsoda tragédia.",
                     "Rendben, rendben, kérem, siessenek!"):
            self.assertIsNone(subs.vocative(text), text)

    def test_short_fragment_with_vocative_joins_scene_lead(self):
        lead = [self._utt(f"E01:{i + 1:04d}", i * 10.0, 2.0, "Beszélek.",
                          1, 0.99) for i in range(4)]
        frag = self._utt("E01:0005", 33.0, 2.0, "Kicsim.", 2, 0.99)
        pairs = clustering.vocative_fragments(
            lead + [frag], {1: "F", 2: "F"}, {1: 120.0, 2: 2.0})
        self.assertEqual([(u.uid, lead_cid) for u, lead_cid in pairs],
                         [("E01:0005", 1)])

    def test_fragment_without_vocative_stays(self):
        lead = [self._utt(f"E01:{i + 1:04d}", i * 10.0, 2.0, "Beszélek.",
                          1, 0.99) for i in range(4)]
        frag = self._utt("E01:0005", 33.0, 2.0, "Igen.", 2, 0.99)
        self.assertEqual(clustering.vocative_fragments(
            lead + [frag], {1: "F", 2: "F"}, {1: 120.0, 2: 2.0}), [])

    def test_unfinished_sentence_continues_the_same_speaker(self):
        """Félbemaradt mondat kisbetűs folytatása nem új beszélő."""
        self.assertTrue(subs.continues("és meghalt, amikor még mindketten",
                                       "nagyon fiatalok voltatok."))
        self.assertFalse(subs.continues("Rendben, oké.", "Hová mész?"))
        a = [self._utt(f"E01:{i + 1:04d}", i * 3.0, 2.5,
                       "Beszélek valamit,", 1, 0.99) for i in range(3)]
        b = self._utt("E01:0004", 9.0, 1.2, "és folytatom.", 2, 0.99)
        self.assertEqual(len(clustering.from_diarization(a + [b], verbose=False)), 1)
        self.assertEqual(b.cluster, 0)

    def test_only_short_fragments_of_a_same_gender_major_lead_move(self):
        # a befogadó nem nagy szereplő -> nincs összevonás
        lead = [self._utt(f"E01:{i + 1:04d}", i * 10.0, 2.0, "Beszélek.",
                          1, 0.99) for i in range(4)]
        frag = self._utt("E01:0005", 33.0, 2.0, "Kicsim.", 2, 0.99)
        self.assertEqual(clustering.vocative_fragments(
            lead + [frag], {1: "F", 2: "F"}, {1: 20.0, 2: 2.0}), [])
        # ellentétes nemű töredék nem kerül a női fő beszélőhöz
        man = self._utt("E01:0005", 33.0, 2.0, "Kicsim.", 2, 0.01)
        self.assertEqual(clustering.vocative_fragments(
            lead + [man], {1: "F", 2: "M"}, {1: 120.0, 2: 2.0}), [])


class TestGrouping(unittest.TestCase):
    """A sorok akusztikus csoportosítása — a fő hibaforrás megszüntetése."""

    def test_sentence_broken_across_cues_is_one_utterance(self):
        cues = [_cue(1, 0.0, 1.0, "Minden rendben,"),
                _cue(2, 1.05, 2.0, "hazamegyünk.")]
        embs = [_v(1), _v(2)]      # szándékosan *különböző* embedding
        g = utterances.group(cues, embs)
        self.assertEqual(g, [[0, 1]], "a folytatódó mondat nem válhat szét")

    def test_short_gap_same_voice_merges(self):
        cues = [_cue(1, 0.0, 1.0, "Gyere ide."), _cue(2, 1.2, 2.0, "Siess!")]
        embs = [_v(3), _v(3)]
        self.assertEqual(utterances.group(cues, embs), [[0, 1]])

    def test_different_voice_with_pause_splits(self):
        cues = [_cue(1, 0.0, 1.0, "Hol van Doron?"), _cue(2, 1.5, 2.4, "Nem tudom.")]
        embs = [_v(4), _v(40)]
        self.assertEqual(utterances.group(cues, embs), [[0], [1]])

    def test_long_gap_breaks_even_with_same_voice(self):
        cues = [_cue(1, 0.0, 1.0, "Vége."), _cue(2, 5.0, 6.0, "Kezdés.")]
        embs = [_v(5), _v(5)]
        self.assertEqual(utterances.group(cues, embs), [[0], [1]])

    def test_embedding_centroid_of_member_cues(self):
        cues = [_cue(1, 0.0, 1.0, "Egy, kettő,"), _cue(2, 1.1, 2.0, "három.")]
        e = _v(6)
        utts = utterances.build("E01", cues, [e, e])
        self.assertEqual(len(utts), 1)
        self.assertAlmostEqual(utterances.cosine(utts[0].emb, e), 1.0, places=5)


class TestDialogueSplit(unittest.TestCase):
    """A felirat szöveges fordulójelölése vágja szét a kétbeszélős sort.

    A régi lánc ezt akusztikus hasonlóságtól tette függővé; rövid, zenével
    kevert ablakokon a kapu elutasította a valódi fordulóváltást, és a két
    szereplő (gyakran ellentétes nemű) sora egy hangon szólt.
    """

    def test_dash_turns_split_by_text(self):
        cue = _cue(1, 0.0, 2.0, "- Mi folyik itt? - Minden rendben.")
        parts = utterances.split_turns(cue)
        self.assertIsNotNone(parts)
        self.assertEqual([p.text for p in parts],
                         ["Mi folyik itt?", "Minden rendben."])
        self.assertAlmostEqual(parts[0].end, parts[1].start, places=3)
        self.assertGreater(parts[1].start, 0.5)

    def test_short_two_turn_cue_splits(self):
        # E06: 0,84 s-os sor, két beszélő — a régi 0,90 s-os vágás miatt
        # egy hangon maradt.
        cue = _cue(123, 676.48, 677.32, "- Igen? - Igen.")
        parts = utterances.split_turns(cue)
        self.assertIsNotNone(parts)
        self.assertEqual([p.text for p in parts], ["Igen?", "Igen."])

    def test_thought_dash_is_not_a_turn(self):
        cue = _cue(1, 0.0, 2.0, "Hát - nem tudom, mit mondjak.")
        self.assertIsNone(utterances.split_turns(cue))

    def test_no_split_for_single_speaker_text(self):
        cue = _cue(1, 0.0, 2.0, "Csak egy ember beszél itt.")
        self.assertIsNone(utterances.split_turns(cue))

    def test_all_bracket_turns_split(self):
        cue = _cue(1, 0.0, 3.0, "[Mi a neved?] [- Juszef.] [- Ki van veled?]")
        parts = utterances.split_turns(cue)
        self.assertEqual([p.text for p in parts],
                         ["Mi a neved?", "Juszef.", "Ki van veled?"])

    def test_single_bracket_block_is_not_split(self):
        cue = _cue(1, 0.0, 2.0, "[Tökéletes az akcentusa.]")
        self.assertIsNone(utterances.split_turns(cue))
        self.assertFalse(subs.has_bracket_dialogue("[Tökéletes az akcentusa.]"))
        self.assertFalse(subs.has_bracket_dialogue("[Beszélünk] [ARAB]"),
                         "a nyelvjelölő nem beszélő-rész")
        self.assertTrue(subs.has_bracket_dialogue("[Hol vagyunk?] [Egy fogolytáborban.]"))

    def test_split_turns_get_own_gender_measurement(self):
        cue = _cue(1, 0.0, 2.0, "- Mi folyik itt? - Minden rendben.")
        e = _v(5)
        seen = []

        def gender_span(a, b):
            seen.append((a, b))
            return 0.9 if a < 1.0 else 0.1

        utts = utterances.build("E01", [cue], [e], [0.5],
                                embed_span=lambda a, b: e,
                                gender_span=gender_span)
        self.assertEqual(len(utts), 2)
        self.assertAlmostEqual(utts[0].p_female, 0.9, places=3)
        self.assertAlmostEqual(utts[1].p_female, 0.1, places=3)
        self.assertEqual(len(seen), 2, "fordulónként külön nem-mérés kell")


class TestContinuation(unittest.TestCase):
    """Egy mondat nem szakadhat két hangra — sem „…”, sem levegővétel miatt."""

    def test_ellipsis_does_not_end_sentence(self):
        # 02:44 (E05): „A zsidók meg…” + „visszamehetnek oda, ahonnan jöttek.”
        cues = [_cue(1, 164.0, 165.4, "A zsidók meg…"),
                _cue(2, 166.5, 169.4, "visszamehetnek oda, ahonnan jöttek.")]
        self.assertEqual(utterances.group(cues, [None, None]), [[0, 1]])

    def test_sentence_survives_1_4s_pause(self):
        # 02:26 (E05): 1,36 mp szünet — a szöveg-folytonosság dönt, nem a gap
        cues = [_cue(1, 140.0, 144.64, "hogy amit Európában mondanak nektek…"),
                _cue(2, 146.0, 148.2, "az nem a valóság.")]
        embs = [_v(21), _v(22)]          # szándékosan eltérő hanglenyomat
        self.assertEqual(utterances.group(cues, embs), [[0, 1]])

    def test_finished_sentence_still_splits(self):
        cues = [_cue(1, 0.0, 1.0, "Vége van."),
                _cue(2, 1.5, 2.5, "Szóval mi lesz?")]
        self.assertEqual(utterances.group(cues, [_v(23), _v(24)]), [[0], [1]])


class TestVoiceLock(unittest.TestCase):
    """A hangzár: egy megnyilatkozás egy hangon szól."""

    def test_plan_keeps_voice_per_utterance(self):
        cues = [_cue(1, 0.0, 1.0, "Egy, kettő,"), _cue(2, 1.1, 2.0, "három.")]
        e = _v(11)
        utts = utterances.build("E01", cues, [e, e])
        utts[0].char, utts[0].voice = "C001", "SUP_F1"
        entries = plan.build(utts, verbose=False)
        self.assertEqual(len(entries), 2)
        self.assertEqual({x.voice for x in entries}, {"SUP_F1"})
        self.assertEqual(plan.voice_consistency(entries), [])

    def test_back_to_back_cues_do_not_overlap(self):
        # gyors párbeszéd: a régi sor farka nem lóghat az új sor alá
        cues = [_cue(1, 0.0, 0.2, "Anya!"), _cue(2, 0.2, 0.6, "Mi az?")]
        e = _v(13)
        utts = utterances.build("E01", cues, [e, e])
        utts[0].char, utts[0].voice = "C001", "SUP_F1"
        entries = plan.build(utts, verbose=False)
        a, b = entries
        self.assertGreaterEqual(b.start - a.end, plan.GAP - 0.001)

    def test_mismatch_is_detected(self):
        cues = [_cue(1, 0.0, 1.0, "Egy,"), _cue(2, 1.1, 2.0, "kettő.")]
        e = _v(12)
        utts = utterances.build("E01", cues, [e, e])
        utts[0].char, utts[0].voice = "C001", "SUP_F1"
        entries = plan.build(utts, verbose=False)
        entries[1].voice = "SUP_M1"          # szándékos hiba
        self.assertEqual(len(plan.voice_consistency(entries)), 1)


class TestManualLines(unittest.TestCase):
    """A felülvizsgáló kézi döntése érvényesül a tervben."""

    @staticmethod
    def _utt() -> utterances.Utterance:
        u = utterances.Utterance(uid="E01:0001", ep="E01", start=10.0, end=12.0)
        u.cues = [subs.Cue(index=1, start=10.0, end=12.0,
                           text="Hová mész? Menj el.")]
        u.char, u.voice = "C001", "SUP_F1"
        return u

    def test_whole_line_mark_overrides_voice(self):
        entries = plan.build(
            [self._utt()], verbose=False,
            labels={("E01:0001", 1): [{"kind": "char", "char": "C002",
                                        "words": None}]},
            voices={"C001": "SUP_F1", "C002": "SUP_M2"},
            pitch_of={"C002": 1.5})
        self.assertEqual(len(entries), 1)
        self.assertEqual((entries[0].char, entries[0].voice), ("C002", "SUP_M2"))
        self.assertEqual(entries[0].pitch, 1.5)
        self.assertTrue(entries[0].manual)

    def test_word_range_splits_the_line(self):
        entries = plan.build(
            [self._utt()], verbose=False,
            labels={("E01:0001", 1): [{"kind": "char", "char": "C002",
                                        "words": [2, 3]}]},
            voices={"C001": "SUP_F1", "C002": "SUP_M2"})
        self.assertEqual([e.char for e in entries], ["C001", "C002"])
        self.assertEqual([e.hu for e in entries], ["Hová mész?", "Menj el."])
        # az idő a szavak arányában oszlik (2-2 szó -> félidő)
        self.assertAlmostEqual(entries[0].cue_start, 10.0, places=3)
        self.assertAlmostEqual(entries[1].cue_start, 11.0, places=3)
        self.assertAlmostEqual(entries[1].cue_end, 12.0, places=3)
        # a kézi vágás szándékos: nem „egy hangon két ember” hiba
        self.assertEqual(plan.voice_consistency(entries), [])

    def test_without_labels_the_line_stays_one(self):
        entries = plan.build([self._utt()], verbose=False)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].voice, "SUP_F1")
        self.assertFalse(entries[0].manual)


class TestSilenceBorrow(unittest.TestCase):
    """A sor ablaka a következő sor eléig bővülhet, hogy ne kelljen összenyomni."""

    @staticmethod
    def _entry(start: float, dur: float) -> plan.Entry:
        return plan.Entry(i=1, uid="E01:0001", ep="E01", hu="szöveg",
                          start=start, end=start + dur)

    def test_room_is_capped_by_the_next_line_and_by_borrow_max(self):
        a = self._entry(0.0, 2.0)
        far = self._entry(5.0, 1.0)
        self.assertAlmostEqual(synth._room(a, {id(a): far.start}), 2.6, places=3)
        near = self._entry(2.1, 1.0)
        self.assertAlmostEqual(synth._room(a, {id(a): near.start}), 2.08, places=3)

    def test_stored_fit_is_read_from_cache_meta(self):
        self.assertAlmostEqual(synth._stored_fit("abc|2.345"), 2.345, places=3)
        self.assertIsNone(synth._stored_fit("abc"))


class TestClustering(unittest.TestCase):
    """Klaszterezés: két hang szétválik, ugyanaz a hang összeáll."""

    def _mk(self, specs):
        utts = []
        for n, (seed, ep, start, dur) in enumerate(specs):
            u = utterances.Utterance(uid=f"{ep}:{n:04d}", ep=ep,
                                     start=float(start), end=float(start + dur),
                                     text="x")
            u.emb = _v(seed)
            utts.append(u)
        return utts

    def test_two_speakers_separate(self):
        utts = self._mk([(100, "E01", 0, 5), (200, "E01", 6, 5),
                         (100, "E01", 12, 5), (200, "E01", 18, 5)])
        clusters = clustering.cluster(utts, verbose=False)
        self.assertEqual(utts[0].cluster, utts[2].cluster)
        self.assertEqual(utts[1].cluster, utts[3].cluster)
        self.assertNotEqual(utts[0].cluster, utts[1].cluster)
        self.assertEqual(len(clusters), 2)

    def test_cross_episode_same_voice_links(self):
        utts = self._mk([(300, "E01", 0, 5), (300, "E02", 0, 5),
                         (301, "E01", 8, 5), (302, "E02", 8, 5)])
        clusters = clustering.cluster(utts, verbose=False)
        self.assertEqual(utts[0].cluster, utts[1].cluster,
                         "ugyanaz a színész a másik epizódban ugyanaz a klaszter")
        self.assertEqual(len(clusters), 3)

    def test_violation_detected_when_voice_differs(self):
        utts = self._mk([(400, "E01", 0, 2), (400, "E01", 2.2, 2)])
        utts[0].voice, utts[1].voice = "SUP_M1", "SUP_M2"
        self.assertEqual(len(clustering.violations(utts)), 1)
        utts[1].voice = "SUP_M1"
        self.assertEqual(clustering.violations(utts), [])

    def test_cluster_evidence_rejects_short_similar_but_distinct_speakers(self):
        # Két külön szereplő rövid, hasonló felkiáltással. A klaszterük
        # mintái külön beszélőt bizonyítanak, ezért nem hangváltás.
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.4, 0.9165, 0.0], dtype=np.float32)
        a0 = a * 0.9 + b * 0.1
        b0 = b * 0.9 + a * 0.1
        specs = [("a1", a, 0), ("a2", a, 1), ("a3", a0, 2),
                 ("b1", b0, 3), ("b2", b, 4), ("b3", b, 5)]
        utts = []
        for uid, vec, slot in specs:
            u = utterances.Utterance(uid=f"E01:{uid}", ep="E01",
                                     start=float(slot * 3),
                                     end=float(slot * 3 + 1), text="x")
            u.emb = vec / np.linalg.norm(vec)
            u.cluster = 0 if uid.startswith("a") else 1
            u.voice = "SUP_M1" if u.cluster == 0 else "SUP_M2"
            utts.append(u)
        self.assertEqual(clustering.violations(utts), [])

    def test_similar_pair_with_cluster_support_is_not_a_switch(self):
        """Két hasonló hangú, de külön klaszterű színész (mért E04-eset).

        A pár koszinusza 0,88 fölött van, de mindkét sor a **saját**
        klaszteréhez illik jobban, mint a többiekhez — tehát két beszélő, a
        hangváltás helyes. Ha ugyanez a pár klaszter-támasz nélkül áll, a
        kapu továbbra is jelez.
        """
        def mk(uid, vec, cluster, slot):
            u = utterances.Utterance(uid=f"E04:{uid}", ep="E04",
                                     start=float(slot * 4),
                                     end=float(slot * 4 + 2), text="x")
            v = np.array(vec, dtype=np.float32)
            u.emb = v / np.linalg.norm(v)
            u.cluster = cluster
            u.voice = "SUP_M2" if cluster == 0 else "SUP_M1"
            return u

        a = [1.0, 0.0, 0.0]
        b = [0.88, 0.47, 0.0]
        supported = [
            mk("a", a, 0, 0), mk("b", b, 1, 1),
            mk("a2", a, 0, 2), mk("a3", [1.0, 0.03, 0.0], 0, 3),
            mk("b2", b, 1, 4), mk("b3", [0.88, 0.46, 0.03], 1, 5),
        ]
        self.assertEqual(clustering.violations(supported), [],
                         "klaszter-támaszú, hasonló pár nem hangváltás")
        thin = [mk("a", a, 0, 0), mk("b", b, 1, 1)]
        self.assertEqual(len(clustering.violations(thin)), 1,
                         "támasz nélkül marad a jelzés")

    def test_season_context_supplies_identity_evidence(self):
        # Szoros szomszédság (ugyanaz a párbeszéd): csak a nagyon erős
        # egyezés bizonyít azonos beszélőt (cos ~0,87).
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.8, 0.6, 0.0], dtype=np.float32)
        a0 = a * 0.9 + b * 0.1
        b0 = b * 0.9 + a * 0.1
        utts = []
        for uid, vec, slot in [("a", a0, 0), ("b", b0, 1)]:
            u = utterances.Utterance(uid=f"E01:{uid}", ep="E01",
                                     start=float(slot * 2),
                                     end=float(slot * 2 + 1), text="x")
            u.emb = vec / np.linalg.norm(vec)
            u.cluster = 0 if uid == "a" else 1
            u.voice = "SUP_M1" if u.cluster == 0 else "SUP_M2"
            utts.append(u)
        context = []
        for uid, vec in [("a1", a), ("a2", a), ("b1", b), ("b2", b)]:
            u = utterances.Utterance(uid=f"E02:{uid}", ep="E02", start=0.0,
                                     end=1.0, text="x")
            u.emb = vec / np.linalg.norm(vec)
            u.cluster = 0 if uid.startswith("a") else 1
            context.append(u)
        self.assertEqual(len(clustering.violations(utts)), 1)
        self.assertEqual(clustering.violations(utts, context=context), [])

    def test_close_pair_needs_strong_evidence(self):
        """Két, ugyanabban a párbeszédben egymást követő sor 0,57 koszinusza
        nem bizonyít azonos beszélőt — a színészváltás itt helyes; távoli
        soroknál viszont a gyengébb egyezés is jelzés."""
        def pair(gap: float, k: float = 0.9):
            out = []
            for uid, vec, slot in (("a", [1.0, 0.0, 0.0], 0),
                                   ("b", [k, (1 - k * k) ** 0.5, 0.0], 1)):
                u = utterances.Utterance(uid=f"E01:{uid}", ep="E01",
                                         start=float(slot * (2 + gap)),
                                         end=float(slot * (2 + gap) + 1),
                                         text="x")
                v = np.array(vec, dtype=np.float32)
                u.emb = v / np.linalg.norm(v)
                u.cluster = 0 if uid == "a" else 1
                u.voice = "SUP_M1" if u.cluster == 0 else "SUP_M2"
                out.append(u)
            return out
        # cos 0,75 — szoros párbeszédben még nem bizonyíték, távoli sornál
        # a 0,60 is jelzi, hogy ugyanaz a színész kapott két hangot
        self.assertEqual(clustering.violations(pair(1.0, 0.75)), [])
        self.assertEqual(len(clustering.violations(pair(40.0, 0.6))), 1)


class TestVoiceUsage(unittest.TestCase):
    """Hanggyakoriság: a főszereplő kizárólagos hangot kap."""

    class _Bank:
        """Könnyű súlyú hamis hangbank (nem tölt be motort)."""

        def __init__(self):
            self.available = {v.vid: v for v in voices.CATALOG}

        def pool(self, gender):
            return [v for v in voices.CATALOG if v.gender == gender]

    def _mk(self, specs, main_share_min=0.10):
        """specs: [(időtartam, nem)] — mindegyik külön klaszter = szereplő."""
        utts, clusters = [], {}
        for i, (dur, gender) in enumerate(specs):
            u = utterances.Utterance(uid=f"E01:{i:04d}", ep="E01",
                                     start=float(i * 60),
                                     end=float(i * 60 + dur), text="x")
            u.cluster = i
            utts.append(u)
            clusters[i] = clustering.Cluster(cid=i, dur=float(dur), gender=gender,
                                             episodes=["E01"], centroid=None)
        reg = registry.Registry("/nonexistent/cast.json",
                                main_share_min=main_share_min)
        reg.assign(clusters, utts, self._Bank(), verbose=False)
        return reg, utts

    def test_main_char_voice_is_exclusive(self):
        # 1 főszereplő (100 s) + 9 mellékszereplő (5-5 s), mind női hang
        reg, _ = self._mk([(100, "F")] + [(5, "F")] * 9, main_share_min=0.10)
        main = max(reg.characters, key=lambda c: c.duration)
        holders = [c for c in reg.characters if c.voice == main.voice]
        self.assertEqual(main.role, "fő")
        self.assertTrue(main.exclusive,
                        "a főszereplő hangját más szereplő nem használhatja")
        self.assertEqual(len(holders), 1)
        # a többi fő hangját senki nem kaphatja meg
        self.assertTrue(all(c.role == "mellék" for c in reg.characters
                            if c is not main))

    def test_usage_ranks_by_speech_time(self):
        reg, _ = self._mk([(100, "M"), (50, "F"), (5, "M"), (1, "F")],
                           main_share_min=0.20)
        usage = reg.analyze()
        by_rank = sorted(reg.characters, key=lambda c: c.rank)
        self.assertEqual([c.duration for c in by_rank], [100.0, 50.0, 5.0, 1.0])
        self.assertEqual([c.rank for c in by_rank], [1, 2, 3, 4])
        self.assertAlmostEqual(sum(c.share for c in reg.characters), 1.0, places=3)
        self.assertEqual([c.role for c in by_rank], ["fő", "fő", "mellék", "mellék"])
        self.assertEqual(usage["n_main"], 2)
        self.assertEqual(usage["n_chars"], 4)
        # hangonkénti összesítés: minden hang szerepel, idő szerint csökkenően
        self.assertEqual(usage["voices"][0]["n_chars"], 1)
        self.assertTrue(all(usage["voices"][i]["seconds"] >=
                            usage["voices"][i + 1]["seconds"]
                            for i in range(len(usage["voices"]) - 1)))

    def test_minor_characters_may_share_a_voice(self):
        # nincs elég hang: a kicsik osztoznak, de egymástól pitch-térülnek
        reg, _ = self._mk([(40, "M")] + [(4, "M")] * 9, main_share_min=0.10)
        usage = reg.analyze()
        main = max(reg.characters, key=lambda c: c.duration)
        shared = [v for v in usage["voices"] if not v["exclusive"]]
        self.assertTrue(shared, "a hangszűke miatt osztozniuk KELL a kicsiknek")
        self.assertTrue(all(c.voice != main.voice
                            for c in reg.characters if c.role == "mellék"),
                        "mellékszereplő nem a főszereplő hangját kapja")
        # az osztozó szereplők hangja+pitch-je egyedi
        pairs = [(c.voice, c.pitch) for c in reg.characters]
        self.assertEqual(len(pairs), len(set(pairs)))


class TestRecast(unittest.TestCase):
    """Évadszintű újrakiosztás: fő hang kizárólagos, a többi osztozik."""

    class _Bank:
        def __init__(self):
            self.available = {v.vid: v for v in voices.CATALOG}

        def pool(self, gender):
            return [v for v in voices.CATALOG if v.gender == gender]

    def _utt(self, char, ep, start, dur, cluster):
        u = utterances.Utterance(uid=f"{ep}:{char}:{start}", ep=ep,
                                 start=float(start), end=float(start + dur),
                                 text="x")
        u.char, u.cluster = char, cluster
        return u

    def _reg(self, specs, main_share_min=0.10):
        reg = registry.Registry("/nonexistent/cast.json",
                                main_share_min=main_share_min)
        reg.voice_pool = [v.vid for v in voices.CATALOG]
        reg.characters = [registry.Character(cid, gender, "")
                          for cid, gender in specs]
        return reg

    def test_episode_durations_are_summed(self):
        reg = self._reg([("C001", "F")], main_share_min=0.4)
        utts = [self._utt("C001", "E01", 0, 10, 0),
                self._utt("C001", "E02", 100, 5, 1)]
        reg.recast(utts, self._Bank(), verbose=False)
        c = reg.characters[0]
        self.assertAlmostEqual(c.ep_durations["E01"], 10.0)
        self.assertAlmostEqual(c.ep_durations["E02"], 5.0)
        self.assertAlmostEqual(c.duration, 15.0)

    def test_main_is_exclusive_and_secondary_avoids_main_voice(self):
        reg = self._reg([("C001", "F"), ("C002", "F"), ("C003", "F"),
                         ("C010", "M"), ("C011", "M"), ("C012", "M")])
        utts = [self._utt("C001", "E01", 0, 100, 0),
                self._utt("C002", "E01", 200, 10, 1),
                self._utt("C003", "E01", 400, 10, 2),
                self._utt("C010", "E01", 600, 100, 3),
                self._utt("C011", "E01", 800, 10, 4),
                self._utt("C012", "E01", 1000, 10, 5)]
        reg.recast(utts, self._Bank(), verbose=False)
        mains = [c for c in reg.characters if c.role == "fő"]
        self.assertEqual(len(mains), 2)
        self.assertTrue(all(c.exclusive for c in mains))
        main_voices = {c.voice for c in mains}
        self.assertTrue(all(c.voice not in main_voices
                            for c in reg.characters if c.role == "mellék"))

    def test_scene_pair_does_not_share_voice_when_free_voice_exists(self):
        reg = self._reg([("C001", "F")] + [(f"C00{i}", "F") for i in range(2, 6)])
        # 1 fő + 4 mellék egy jelenetben; 4 szabad női hang van
        utts = [self._utt("C001", "E01", 0, 100, 0)]
        utts += [self._utt(f"C00{i}", "E01", 200 + i, 1, i)
                 for i in range(2, 6)]
        reg.recast(utts, self._Bank(), verbose=False)
        self.assertEqual(reg.conflicts(utts, strict=True), [])


    def test_character_tiers(self):
        reg = registry.Registry("/nonexistent/cast.json", main_share_min=0.5)
        reg.characters = [
            registry.Character("C001", "F", "SUP_F1", duration=100.0,
                               episodes=["E01", "E02"],
                               ep_durations={"E01": 50.0, "E02": 50.0}),
            registry.Character("C002", "M", "SUP_M1", duration=12.0,
                               episodes=["E01", "E02"],
                               ep_durations={"E01": 6.0, "E02": 6.0}),
            registry.Character("C003", "F", "SUP_F2", duration=12.0,
                               episodes=["E01"], ep_durations={"E01": 12.0}),
            registry.Character("C004", "M", "SUP_M2", duration=4.0,
                               episodes=["E01"], ep_durations={"E01": 4.0}),
        ]
        usage = reg.analyze()
        tiers = {c.char_id: c.tier for c in reg.characters}
        self.assertEqual(tiers, {"C001": "fő", "C002": "mellék",
                                 "C003": "epizód", "C004": "statiszta"})
        self.assertEqual(usage["tiers"], {"fő": 1, "mellék": 1,
                                          "epizód": 1, "statiszta": 1})


class TestSpeakerAudit(unittest.TestCase):
    def test_hebrew_normalization_removes_markup_and_keeps_dialogue(self):
        text = "{\\an8}" + "\u202b[דורון] שלום לכם!\u202c"
        self.assertEqual(speaker_audit.normalize_hebrew(text), "שלום לכם!")
        self.assertEqual(speaker_audit.normalize_hebrew("[מוזיקה]"), "")

    def test_word_speaker_assignment_preserves_uncertainty(self):
        turns = [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
                 {"start": 0.9, "end": 1.8, "speaker": "SPEAKER_01"}]
        words = [{"word": "א", "start": 0.2, "end": 0.6},
                 {"word": "ב", "start": 0.9, "end": 1.0},
                 {"word": "ג", "start": 2.0, "end": 2.2}]
        got = speaker_audit.turn_segment_ids(turns, words, margin_s=0.0)
        self.assertEqual(got[0]["speaker"], "SPEAKER_00")
        self.assertEqual(got[1]["speaker"], None)
        self.assertEqual(got[1]["speaker_confidence"], "ambiguous_overlap")
        self.assertEqual(got[2]["speaker_confidence"], "no_turn_overlap")

    def test_word_without_alignment_stays_unassigned(self):
        got = speaker_audit.turn_segment_ids(
            [{"start": 0, "end": 1, "speaker": "SPEAKER_00"}],
            [{"word": "x", "start": None, "end": None}])
        self.assertEqual(got[0]["speaker"], None)
        self.assertEqual(got[0]["speaker_confidence"], "no_word_timing")


class TestHebrewAlign(unittest.TestCase):
    """Helyi héber char-CTC forced-alignment: tokenizálás, Viterbi, szóhatárok."""

    VOCAB = {"[PAD]": 30, "|": 9, "ב": 10, "ו": 11, "ק": 12,
             "ר": 13, "ט": 14, "ג": 17, "י": 18, "א": 16}

    def test_tokenize_keeps_model_chars_and_drops_punctuation(self):
        items, words = speaker_audit.tokenize_hebrew("בוקר טוב, גיבור!", self.VOCAB)
        # a szavak megtartják a felirat szövegét, a vessző/kijelző nem ad CTC célt
        self.assertEqual(words, ["בוקר", "טוב,", "גיבור!"])
        self.assertEqual([w for _, w in items],
                         [0, 0, 0, 0, -1, 1, 1, 1, -1, 2, 2, 2, 2, 2])

    def test_tokenize_punctuation_only_yields_no_targets(self):
        items, _ = speaker_audit.tokenize_hebrew("… !!!", self.VOCAB)
        self.assertEqual(items, [])

    def _flat(self, n_frames):
        """log_probs where each token block clearly favours its label."""
        return np.full((n_frames, 31), -20.0)

    def test_viterbi_places_tokens_in_chronological_blocks(self):
        # target: blank,1,blank,2,blank -> L=5
        lp = self._flat(24)
        for t in range(0, 5):   lp[t, 30] = 0.0   # blank
        for t in range(5, 10):  lp[t, 1] = 0.0    # token 1
        for t in range(10, 15): lp[t, 30] = 0.0   # blank
        for t in range(15, 20): lp[t, 2] = 0.0    # token 2
        for t in range(20, 24): lp[t, 30] = 0.0   # blank
        path = speaker_audit.ctc_viterbi(lp, [1, 2], blank=30)
        self.assertTrue(all(path[t] == 1 for t in range(5, 10)))
        self.assertTrue(all(path[t] == 3 for t in range(15, 20)))
        self.assertTrue(path[-1] in (4, 3))
        self.assertTrue(np.all(np.diff(path) >= 0), "a célállapot-monotonitás sérült")

    def test_viterbi_keeps_repeated_letters_separate(self):
        # target: blank,1,blank,1,blank — a két egyforma betű nem olvadhat össze
        lp = self._flat(15)
        for t in range(0, 3):  lp[t, 30] = 0.0
        for t in range(3, 6):  lp[t, 1] = 0.0
        for t in range(6, 8):  lp[t, 30] = 0.0
        for t in range(8, 11): lp[t, 1] = 0.0
        for t in range(11, 15): lp[t, 30] = 0.0
        path = speaker_audit.ctc_viterbi(lp, [1, 1], blank=30)
        self.assertTrue(all(path[t] == 1 for t in range(3, 6)))
        self.assertTrue(all(path[t] == 2 for t in range(6, 8)),
                        "az ismétlődő betű között maradnia kell a blanknak")
        self.assertTrue(all(path[t] == 3 for t in range(8, 11)))

    def test_viterbi_rejects_window_shorter_than_text(self):
        lp = self._flat(3)
        with self.assertRaises(ValueError):
            speaker_audit.ctc_viterbi(lp, [1] * 10, blank=30)

    def test_words_from_window_returns_word_spans_and_scores(self):
        lp = self._flat(24)
        tokens, _ = speaker_audit.tokenize_hebrew("בוקר טוב", self.VOCAB)
        ids = [i for i, _ in tokens]
        # minden tokennek 2 keret + utána 1 blank keret
        for i, cid in enumerate(ids):
            base = i * 3
            lp[base:base + 2, cid] = 0.0
            lp[base + 2, 30] = 0.0
        got = speaker_audit.words_from_window(lp, "בוקר טוב", self.VOCAB, blank=30,
                                              offset_s=1.0, source_cue=5)
        self.assertEqual([w["word"] for w in got], ["בוקר", "טוב"])
        self.assertTrue(all(1.0 <= w["start"] < w["end"] <= 1.72 for w in got))
        self.assertTrue(all(0.0 < w["score"] <= 1.0 for w in got))
        self.assertTrue(all(w["source_cue"] == 5 for w in got))
        self.assertLessEqual(got[0]["end"], got[1]["start"])


class TestIdentity(unittest.TestCase):
    """Stabil C### azonosság: a diarizált címke ne számítson, a hang igen."""

    RECORDS = {
        "C003": {"char_id": "C003", "gender": "M", "voice": "SUP_M2",
                 "centroid": [1.0] + [0.0] * 31, "episodes": ["E05"], "origin": "cast"},
        "C004": {"char_id": "C004", "gender": "F", "voice": "SUP_F1",
                 "centroid": [0.0, 1.0] + [0.0] * 30, "episodes": ["E05"], "origin": "cast"},
    }
    T = {"link_cos": 0.55, "weak_cos": 0.42, "margin": 0.05, "min_exemplars": 3}

    @staticmethod
    def _exemplars(rows: dict, dim: int = 32) -> dict:
        out = {}
        for cid, rows_ in rows.items():
            out[cid] = identity._as_matrix(rows_)
        return out

    def test_exemplar_matching_ignores_cosine_inflation_of_centroids(self):
        # két különböző beszélő centroidja 0,9-re hasonlít (mért jelenség),
        # a mintáik viszont jól szétválnak -> a minta-alapú összevetés dönt
        import numpy as np
        base = np.zeros(32, dtype=np.float32)
        a = base.copy(); a[0] = 1.0
        b = base.copy(); b[0] = 0.6; b[1] = 0.8
        ex = self._exemplars({"C003": [a] * 4, "C004": [b] * 4})
        probe = base.copy(); probe[0] = 0.98; probe[2] = 0.1
        cands = identity.rank_candidates(probe, ex, self.RECORDS)
        self.assertEqual(cands[0]["char_id"], "C003")
        status, _, match = identity.decide(cands, self.T)
        self.assertEqual(status, "linked")
        self.assertEqual(match, "C003")

    def test_unrelated_voice_is_a_new_character(self):
        a = np.zeros(32, dtype=np.float32); a[0] = 1.0
        b = np.zeros(32, dtype=np.float32); b[5] = 1.0
        ex = self._exemplars({"C003": [a] * 4})
        probe = np.zeros(32, dtype=np.float32); probe[9] = 1.0
        cands = identity.rank_candidates(probe, ex, self.RECORDS)
        status, reason, _ = identity.decide(cands, self.T)
        self.assertEqual(status, "new")
        self.assertIn("hasonlóság", reason)

    def test_few_exemplars_never_link_automatically(self):
        a = np.zeros(32, dtype=np.float32); a[0] = 1.0
        ex = self._exemplars({"C003": [a]})
        cands = identity.rank_candidates(a, ex, self.RECORDS)
        status, reason, match = identity.decide(cands, self.T)
        self.assertEqual(status, "weak")
        self.assertIsNone(match)
        self.assertIn("mintája", reason)

    def test_close_second_candidate_is_ambiguous(self):
        import numpy as np
        a = np.zeros(32, dtype=np.float32); a[0] = 1.0
        b = np.zeros(32, dtype=np.float32); b[0] = 0.99; b[1] = 0.14
        ex = self._exemplars({"C010": [a] * 4, "C011": [b] * 4})
        probe = np.zeros(32, dtype=np.float32); probe[0] = 1.0; probe[2] = 0.02
        cands = identity.rank_candidates(probe, ex, {
            "C010": {"char_id": "C010", "gender": "M"},
            "C011": {"char_id": "C011", "gender": "M"}})
        status, reason, _ = identity.decide(cands, self.T)
        self.assertEqual(status, "ambiguous")
        self.assertIn("különbség", reason)

    def test_gender_conflict_never_links(self):
        import numpy as np
        a = np.zeros(32, dtype=np.float32); a[0] = 1.0
        ex = self._exemplars({"C003": [a] * 4})
        cands = identity.rank_candidates(a, ex, self.RECORDS)
        status, reason, _ = identity.decide(cands, self.T, gender="F")
        self.assertEqual(status, "weak")
        self.assertIn("neme", reason)

    def test_next_char_id_continues_the_series(self):
        self.assertEqual(identity.next_char_id(self.RECORDS), "C005")
        self.assertEqual(identity.next_char_id({}), "C001")

    def test_confirm_writes_alias_and_keeps_exemplars(self):
        gallery = {"characters": {}, "aliases": {}}
        vec = np.asarray([1.0, 0.0] + [0.0] * 30, dtype=np.float32)
        samples = [[1.0, 0.0] + [0.0] * 30, [0.99, 0.02] + [0.0] * 30]
        identity.confirm(gallery, "E06", "SPEAKER_04", "C003", vec=vec,
                         samples=samples, gender="M", voice="SUP_M2", seconds=6.0)
        self.assertEqual(gallery["aliases"]["E06"]["SPEAKER_04"], "C003")
        row = gallery["characters"]["C003"]
        self.assertEqual(row["episodes"], ["E06"])
        self.assertAlmostEqual(row["seconds"], 6.0)
        self.assertEqual(len(row["samples"]), 2)
        self.assertAlmostEqual(row["centroid"][0], 1.0, places=3)

    def test_confirm_without_samples_uses_centroid(self):
        gallery = {"characters": {}, "aliases": {}}
        vec = np.asarray([1.0, 0.0] + [0.0] * 30, dtype=np.float32)
        identity.confirm(gallery, "E06", "SPEAKER_01", "C009", vec=vec, seconds=3.0)
        self.assertAlmostEqual(gallery["characters"]["C009"]["centroid"][0], 1.0, places=4)

    def test_confirm_rejection_remembers_no_character(self):
        gallery = {"characters": {}, "aliases": {}}
        identity.confirm(gallery, "e06", "SPEAKER_09", None)
        self.assertIsNone(gallery["aliases"]["E06"]["SPEAKER_09"])
        self.assertEqual(gallery["characters"], {})


class TestVoiceDecision(unittest.TestCase):
    """Tipizált döntési modell: nem, elérhetőség, stabilitás, hangszín."""

    def _char(self, **kw):
        base = {"char_id": "C007", "gender": "M", "voice": None,
                "main": False, "evidence_seconds": 8.0}
        base.update(kw)
        return base

    def test_gender_filters_the_pool(self):
        res = voice_decision.decide_voice(self._char(gender="F"))
        self.assertTrue(res["label"].startswith("F") or res["label"] in ("anna", "berta"))
        chosen = voice_decision.BY_ID[res["choice"]]
        self.assertEqual(chosen.gender, "F")

    def test_cooccurring_character_blocks_that_voice(self):
        char = self._char(voice="SUP_M1")
        res = voice_decision.decide_voice(
            char, usage={"SUP_M1": ["C002"]}, cooccurring={"C002"})
        blocked = [c for c in res["candidates"] if c["voice"] == "SUP_M1"][0]
        self.assertTrue(blocked["blocked"])
        self.assertNotEqual(res["choice"], "SUP_M1")

    def test_existing_assignment_is_stable(self):
        char = self._char(voice="SUP_M3")
        res = voice_decision.decide_voice(char)
        self.assertEqual(res["choice"], "SUP_M3")
        self.assertGreaterEqual(res["score"], 2)

    def test_timbre_evidence_changes_the_choice(self):
        char = self._char()
        timbres = {"SUP_M1": {"cosine": 0.70}, "SUP_M2": {"cosine": 0.05},
                   "SUP_M3": {"cosine": 0.10}, "SUP_M4": {"cosine": 0.02},
                   "SUP_M5": {"cosine": 0.08}, "PIP_imre": {"cosine": 0.04}}
        res = voice_decision.decide_voice(char, timbres=timbres)
        self.assertEqual(res["choice"], "SUP_M1")
        self.assertEqual(res["score"], 3)

    def test_unknown_gender_requests_human_review(self):
        res = voice_decision.decide_voice(self._char(gender=None))
        self.assertGreaterEqual(res["noul"]["human_review"], 0.3)
        self.assertIn(res["score"], (2, 3))

    def test_probabilities_are_a_distribution(self):
        res = voice_decision.decide_voice(self._char())
        self.assertAlmostEqual(sum(res["probabilities"].values()), 1.0, places=3)
        self.assertTrue(0.0 <= res["confidence"] <= 1.0)
        self.assertEqual(res["probabilities"][res["choice"]], res["answer_confidence"])

    def test_shared_voice_asks_for_pitch_shift(self):
        res = voice_decision.decide_voice(
            self._char(voice="SUP_M4"), usage={"SUP_M4": ["C020"]})
        self.assertGreater(res["noul"]["pitch_shift"], 0.1)

    def test_timbre_similarity_mapping(self):
        self.assertEqual(voice_decision.timbre_similarity(None), 0.5)
        self.assertEqual(voice_decision.timbre_similarity(0.80), 1.0)
        self.assertEqual(voice_decision.timbre_similarity(-0.20), 0.0)

    def test_remote_backend_is_off_by_default(self):
        self.assertEqual(voice_decision.remote_system_one({}, "x", {}), {})


class TestReview(unittest.TestCase):
    """Emberi döntés: elsőbbség, galéria-frissítés, összegzés."""

    class _P:
        def __init__(self, root: Path):
            self.work = root
            self._root = root

        def ep_dir(self, tag: str) -> Path:
            d = self._root / tag
            d.mkdir(parents=True, exist_ok=True)
            return d

    def test_manual_decision_overrides_automatic_candidate(self):
        auto = {"speakers": {"SPEAKER_00": {"speaker": "SPEAKER_00", "char_id": "C003",
                                           "status": "linked", "review_required": True}}}
        merged = review.apply(auto, {"speakers": {"SPEAKER_00": {
            "decision": "corrected", "char_id": "C011", "gender": "M"}}})
        row = merged["speakers"]["SPEAKER_00"]
        self.assertEqual(row["char_id"], "C011")
        self.assertEqual(row["status"], "corrected")
        self.assertFalse(row["review_required"])
        self.assertEqual(merged["pending_review"], 0)

    def test_rejection_hides_the_character(self):
        merged = review.apply({"speakers": {}}, {"speakers": {"SPEAKER_05": {
            "decision": "rejected"}}})
        self.assertIsNone(merged["speakers"]["SPEAKER_05"]["char_id"])
        self.assertEqual(merged["counts"]["rejected_by_human"], 1)

    def test_saved_decision_survives_reload_and_updates_gallery(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            project = self._P(Path(tmp))
            review.set_speaker(project, "e06", "SPEAKER_01", "accepted",
                               char_id="C003", gender="M", voice="SUP_M2", seconds=5.0)
            again = review.load(project, "E06")
            self.assertEqual(again["speakers"]["SPEAKER_01"]["char_id"], "C003")
            self.assertEqual(review.summary(again)["accepted"], 1)
            gallery = identity.load_gallery(Path(tmp) / "identity-gallery.json")
            self.assertEqual(gallery["aliases"]["E06"]["SPEAKER_01"], "C003")

    def test_stale_key_resolves_by_subtitle_index(self):
        """Újracsoportosítás után a jelölés sorszám alapján megmarad."""
        data = {"lines": {"E04:0077#89": {"marks": [{"words": None, "kind": "char",
                                                       "char": "C002", "gender": "F"}]}}}
        k, entry = linereview.resolve(data, "E04:0075", 89, {"E04:0075", "E04:0076"})
        self.assertEqual(k, "E04:0075#89")
        self.assertEqual(entry["marks"][0]["char"], "C002")

    def test_live_other_line_is_not_a_fallback(self):
        data = {"lines": {"E04:0075#88": {"marks": [{"kind": "char", "char": "C001"}]}}}
        k, entry = linereview.resolve(data, "E04:0075", 89, {"E04:0075", "E04:0076"})
        self.assertEqual((k, entry), ("E04:0075#89", {}))

    def test_exact_key_wins_over_stale(self):
        data = {"lines": {"E04:0075#89": {"marks": [{"kind": "char", "char": "C001"}]},
                          "E04:0077#89": {"marks": [{"kind": "char", "char": "C002"}]}}}
        k, entry = linereview.resolve(data, "E04:0075", 89, {"E04:0075"})
        self.assertEqual(k, "E04:0075#89")
        self.assertEqual(entry["marks"][0]["char"], "C001")

    def test_unknown_decision_is_rejected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                review.set_speaker(self._P(Path(tmp)), "E06", "SPEAKER_01", "maybe")


class TestSubs(unittest.TestCase):
    def test_tts_text_strips_markup(self):
        self.assertEqual(subs.tts_text("<i>Szia</i> (mosolyog) [ARAB]!"), "Szia !")

    def test_tts_text_drops_music_notes(self):
        self.assertEqual(subs.tts_text("♪ Ez az álom... ♪"), "Ez az álom...")

    def test_music_lines_are_not_dialogue(self):
        for text in ("♪ Ez az álom... ♪",
                     "<i>♪ 'Cause this life I'm living... ♪</i>",
                     "[Music]", "(zene)", "Zene szól.", "♪♪"):
            self.assertTrue(subs.is_music(text), text)
        for text in ("Ez egy normális mondat.", "- Szia! - Hello.",
                     "♪ Igen, ♪ de nem.", "[kiáltás] [válasz]"):
            self.assertFalse(subs.is_music(text), text)

    def test_music_cue_is_not_dubbed(self):
        uu = utterances.Utterance(uid="E01:0001", ep="E01", start=0.0, end=2.5,
                                  cues=[_cue(1, 0.0, 1.0, "Szia."),
                                        _cue(2, 1.2, 2.5, "♪ La la ♪")],
                                  text="Szia.", char="C001", voice="F1")
        uu.cues[1].music = True
        got = plan.build([uu], verbose=False)
        self.assertEqual([e.hu for e in got], ["Szia."])
        self.assertFalse(uu.music)
        solo = utterances.Utterance(uid="E01:0002", ep="E01", start=3.0,
                                    end=4.0, cues=[_cue(3, 3.0, 4.0, "♪ La ♪")])
        solo.cues[0].music = True
        self.assertTrue(solo.music)

    def test_screen_text_detection(self):
        self.assertTrue(subs.is_screen_text("{Nem tűnik el} AZ IDŐ"))
        self.assertFalse(subs.is_screen_text("Ez egy normális mondat."))

    def test_dash_dialogue(self):
        self.assertTrue(subs.has_dash_dialogue("- A? - B."))
        self.assertEqual(subs.split_dash_dialogue("- A? - B."), ("A?", "B."))

    def test_cut_words_prefers_punctuation(self):
        left, right = subs.cut_words("Minden rendben, hazamegyünk ma este", 0.5)
        self.assertTrue(left.endswith(","))



class TestTextcheck(unittest.TestCase):
    """Szöveg-ellenőrző pont a hanggenerálás előtt: szabályok + LLM-keret."""

    def test_rules_fix_spacing_and_punctuation(self):
        fixed, flags = textcheck.rule_pass("Nem megy  az ablak , rendben ?")
        self.assertEqual(fixed, "Nem megy az ablak, rendben?")
        self.assertEqual(flags, [])
        fixed, _ = textcheck.rule_pass("Vége.Mehetünk tovább.")
        self.assertEqual(fixed, "Vége. Mehetünk tovább.")

    def test_rules_leave_valid_text_alone(self):
        # a tizedesjegy nem mondatvég, az nem kaphat szóköz mögé
        fixed, flags = textcheck.rule_pass("A 2.5-ös verzió jó.")
        self.assertEqual(fixed, "A 2.5-ös verzió jó.")
        self.assertEqual(flags, [])

    def test_repeated_word_flagged_but_emphasis_is_not(self):
        _, flags = textcheck.rule_pass("A kutya kutya ugat.")
        self.assertTrue(any("ismételt" in f for f in flags))
        self.assertEqual(textcheck.rule_pass("Nem nem megy.")[1], [])
        self.assertEqual(textcheck.rule_pass("Nem, nem, nem.")[1], [])

    def test_llm_fix_validation(self):
        cfg = dict(textcheck.DEFAULTS)
        self.assertTrue(textcheck._accepted(
            "Nem tudom meg csinálni ezt", "Nem tudom megcsinálni ezt", cfg))
        # teljesen más mondat → elutasítva
        self.assertFalse(textcheck._accepted(
            "Szia, mit keresel itt?", "A rókák télen alszanak a hó alatt.", cfg))
        # lényeges túlterjesztés → elutasítva
        self.assertFalse(textcheck._accepted(
            "Szia", "Szia! Nagyon sok plusz szöveg, amit biztosan nem kért senki.",
            cfg))

    def test_run_without_llm_applies_only_rules(self):
        from dubstudio.config import Project
        p = Project(name="teszt")
        p.llm = {"enabled": False}
        e = plan.Entry(i=1, uid="E05:0001", ep="E05", hu="Szia  világ , üdv")
        stats = textcheck.run([e], p, verbose=False)
        self.assertEqual(e.hu, "Szia világ, üdv")
        self.assertEqual(stats["rule_fixed"], 1)
        self.assertEqual(stats["llm_status"], "disabled")
        self.assertEqual(stats["remaining"], 0)
        self.assertFalse(stats["required"])


class TestAsrMapping(unittest.TestCase):
    """ASR-visszaellenőrzés: a felismert szavak időbélyeg szerinti térképezése."""

    def test_words_in_window_maps_by_timestamp(self):
        words = [{"word": "Szia", "start": 1.0}, {"word": "világ", "start": 1.4},
                 {"word": None, "start": 1.5},
                 {"word": "Másik", "start": 5.0}, {"word": "üres", "start": None}]
        self.assertEqual(asrcheck.words_in_window(words, 0.9, 1.6),
                         ["Szia", "világ"])
        self.assertEqual(asrcheck.words_in_window(words, 3.0, 4.0), [])

    def test_missing_audio_line_is_flagged(self):
        # egy sorhoz sem esik szó az ablakban → „nincs hallott szó”
        self.assertEqual(asrcheck.words_in_window([], 0.0, 2.0), [])

    def test_loop_detection_separates_whisper_repetition(self):
        # ismétlés-hurok: ugyanaz a szó/mondat darabonként visszatér
        self.assertTrue(asrcheck._is_loop("mindenki mindenki mindenki mindenki"))
        self.assertTrue(asrcheck._is_loop(
            "ezt mindenki különbözött mindenki különbözött mondta"))
        # normál szövegben nincs ilyen
        self.assertFalse(asrcheck._is_loop("Igen, köszönöm szépen a választ!"))
        self.assertFalse(asrcheck._is_loop(""))

    def test_row_kind_and_ordering(self):
        self.assertEqual(asrcheck._kind(""), "empty")
        self.assertEqual(asrcheck._kind("mindenki mindenki mindenki"), "loop")
        self.assertEqual(asrcheck._kind("Egy valódi mondat a szövegből."), "real")
        rows = [{"kind": "empty", "cer": 1.0, "i": 1},
                {"kind": "real", "cer": 0.4, "i": 2},
                {"kind": "loop", "cer": 9.0, "i": 3},
                {"kind": "real", "cer": 0.9, "i": 4}]
        out = asrcheck._sort_rows(list(rows))
        self.assertEqual([r["i"] for r in out], [4, 2, 3, 1],
                         "először a valódi találatok, azon belül CER szerint")


class TestAsrHybrid(unittest.TestCase):
    """Hibrid ASR: Parakeet elsődleges + Whisper-HU megerősítés a gyanús sorokon."""

    def test_merge_spans_joins_close_windows_and_caps_length(self):
        # közeli ablakok egy szeletbe; a 29 s-os plafon betartva
        spans = [(10.0, 12.0), (12.5, 14.0), (20.0, 22.0)]
        self.assertEqual(asrcheck._merge_spans(spans),
                         [(10.0, 14.0), (20.0, 22.0)])
        long = asrcheck._merge_spans([(0.0, 40.0)])
        self.assertEqual(len(long), 1)
        self.assertLessEqual(long[0][1] - long[0][0], asrcheck.WIN_S)
        far = asrcheck._merge_spans([(0.0, 1.0), (10.0, 11.0)])
        self.assertEqual(len(far), 2, "távoli ablakok külön hívásban")

    def test_pack_spans_packs_far_windows_into_one_call(self):
        # a távoli szeletek is egy hívásba kerülnek (csend választja el őket)
        groups = asrcheck._pack_spans([(0.0, 4.0), (10.0, 14.0),
                                       (60.0, 64.0)])
        self.assertEqual(len(groups), 1, "három szelet → egy hívás")
        self.assertEqual([g[2] for g in groups[0]], [0.0, 4.4, 8.8],
                         "a kezdés-eltolás a szelet + csend hossza szerint")
        g = groups[0]
        self.assertLessEqual(g[-1][2] + (g[-1][1] - g[-1][0]),
                             asrcheck.PACK_LIMIT_S,
                             "a hívás nem lépi túl a csomagolt keretet")

    def test_pack_spans_respects_limit_and_keeps_every_span(self):
        spans = [(float(i * 10), float(i * 10 + 5)) for i in range(8)]
        groups = asrcheck._pack_spans(spans)
        got = sorted((a, b) for grp in groups for (a, b, _o) in grp)
        self.assertEqual(got, sorted(spans), "minden szelet megmarad")
        for grp in groups:
            total = grp[-1][2] + (grp[-1][1] - grp[-1][0])
            self.assertLessEqual(total, asrcheck.PACK_LIMIT_S + 1e-6,
                                 "egy hívás sem hosszabb a keretnél")
        self.assertLess(len(groups), len(spans),
                        "a csomagolás kevesebb hívást ad")

    def test_pack_spans_splits_oversized_span(self):
        groups = asrcheck._pack_spans([(0.0, 40.0)])
        self.assertTrue(all(len(grp) == 1 and grp[0][1] - grp[0][0]
                            <= asrcheck.PACK_LIMIT_S + 1e-6 for grp in groups))
        self.assertEqual([grp[0][0] for grp in groups][0], 0.0,
                         "a felvágott darabok a globális időn maradnak")

    def test_map_words_restores_global_time(self):
        group = [(10.0, 12.0, 0.0), (20.0, 22.0, 4.25)]
        local = [{"word": "egy", "start": 1.0},
                 {"word": "kettő", "start": 5.0},
                 {"word": "semmi", "start": None}]
        out = asrcheck._map_words(group, local)
        self.assertEqual(len(out), 2, "a kezdés nélküli szót kihagyjuk")
        self.assertAlmostEqual(out[0]["start"], 11.0)
        self.assertAlmostEqual(out[1]["start"], 20.75,
                               msg="a csendbe eső szót a szomszéd szelethez")

    def test_confirm_pack_switch(self):
        self.assertTrue(asrcheck.confirm_pack(), "alapból bekapcsolva")
        os.environ["DUBSTUDIO_ASR_CONFIRM_PACK"] = "0"
        try:
            self.assertFalse(asrcheck.confirm_pack())
        finally:
            os.environ.pop("DUBSTUDIO_ASR_CONFIRM_PACK", None)

    def test_gate_threshold_is_per_model(self):
        from dubstudio.config import Project
        p = Project(name="teszt")
        p.gate = {"asr_cer_max": {"parakeet": 0.25, "whisper": 0.60}}
        par = {"model": "nemo-parakeet-tdt-0.6b-v3", "cer": 0.20,
               "bad": 3}
        m = asrcheck.apply_gate({"gates": []}, par, p)
        self.assertTrue(m["gates"][-1]["ok"], "Parakeet 0,20 ≤ 0,25")
        par_bad = dict(par, cer=0.40)
        m = asrcheck.apply_gate({"gates": []}, par_bad, p)
        self.assertFalse(m["gates"][-1]["ok"])
        wh = {"model": "sarpba/whisper-hu-large-v3-turbo-finetuned",
              "cer": 0.55, "bad": 9}
        m = asrcheck.apply_gate({"gates": []}, wh, p)
        self.assertTrue(m["gates"][-1]["ok"], "Whisper 0,55 ≤ 0,60")

    def test_gate_keeps_plain_number_threshold(self):
        from dubstudio.config import Project
        p = Project(name="teszt")
        p.gate = {"asr_cer_max": 0.60}
        m = asrcheck.apply_gate({"gates": []},
                                {"model": "nemo-parakeet", "cer": 0.30,
                                 "bad": 1}, p)
        self.assertTrue(m["gates"][-1]["ok"])

    def test_adaptive_duck_pulls_loud_bed_under_speech(self):
        # 0. ágy = az átlag (allowed), 1. ágy = +12 dB (intro)
        allowed = 0.1
        f_rms = np.array([allowed, allowed * 4.0], dtype=np.float32)
        env = np.array([0.0, 1.0], dtype=np.float32)
        duck = 10.0 ** (-mix.DUCK_DB / 20.0)
        g, target = mix._duck_curve(env, f_rms, duck, allowed)
        self.assertAlmostEqual(float(g[0]), 1.0, places=5,
                               msg="beszéd nélkül az ágy teljes szinten marad")
        self.assertAlmostEqual(float(g[1]), 0.25, places=3,
                               msg="intro alatt az ágy az átlagszintig húzódik")
        self.assertLess(float(target[1]), duck)
        self.assertAlmostEqual(float(target[0]), duck, places=5,
                               msg="átlagos ágyon a fix ducking marad")

    def test_adaptive_duck_never_boosts(self):
        f_rms = np.array([0.001, 0.0005], dtype=np.float32)
        env = np.array([1.0, 1.0], dtype=np.float32)
        duck = 10.0 ** (-mix.DUCK_DB / 20.0)
        g, _ = mix._duck_curve(env, f_rms, duck, 0.1)
        self.assertTrue((g <= 1.0 + 1e-6).all(), "a ducking sosem erősít")


class _FakeWhisper:
    """Hamis pipeline: minden szóra egy ``wN`` szót ad, sorban — így
    összehasonlítható, hogy a kötegelt és a soronkénti út ugyanazt adja."""

    def __init__(self, reject_list: bool = False):
        self.reject_list = reject_list
        self.calls = 0            # pipeline-hívások száma
        self.list_calls = 0       # listás (kötegelt) hívások száma
        self.n = 0                # kiadott szavak száma
        self._pipe = self.pipe

    def _load(self):
        pass

    def pipe(self, data, generate_kwargs=None, return_timestamps=None,
             batch_size=None):
        self.calls += 1
        items = data if isinstance(data, list) else [data]
        if isinstance(data, list):
            self.list_calls += 1
            if self.reject_list:
                raise TypeError("a pipeline nem fogad listát")
        out = []
        for _ in items:
            self.n += 1
            out.append({"chunks": [{"text": f"w{self.n}",
                                    "timestamp": (0.5, 0.7)}]})
        return out if isinstance(data, list) else out[0]


class TestAsrBatch(unittest.TestCase):
    """Kötegelt (batch) ablak-feldolgozás: forrás, útválasztás, egyenértékűség."""

    def test_batch_size_from_project_env_and_default(self):
        import os

        from dubstudio.config import Project

        p = Project(name="t")
        p.asr = {"batch": 2}
        self.assertEqual(asrcheck.batch_size(p), 2, "a projekt asr.batch adja")
        # A memóriacsúcs miatt a köteg felső korlátja MAX_BATCH: a nagyobb
        # érték nem növeli a memóriát, hanem a plafonra esik vissza.
        p.asr = {"batch": 8}
        self.assertEqual(asrcheck.batch_size(p), asrcheck.MAX_BATCH,
                         "a köteg a memória-plafonra korlátozva")
        p.asr = {}
        os.environ["DUBSTUDIO_ASR_BATCH"] = "1"
        try:
            self.assertEqual(asrcheck.batch_size(p), 1, "környezeti változó")
        finally:
            os.environ.pop("DUBSTUDIO_ASR_BATCH", None)
        self.assertEqual(asrcheck.batch_size(None), asrcheck.DEFAULT_BATCH,
                         "alapértelmezés: a régi, soronkénti út")

    def test_checkpoint_roundtrip_and_bad_lines(self):
        """A megszakított ASR ellenőrzőpontja: kész ablakok, hibás sorok kihagyva."""
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "asr_windows_1.jsonl"
            p.write_text(
                json.dumps({"i": 0, "words": [{"word": "a", "start": 0.0}]})
                + "\n" + "not-json\n"
                + json.dumps({"i": 3, "words": []}) + "\n",
                encoding="utf-8")
            done = asrcheck._load_checkpoint(p)
        self.assertEqual(sorted(done), [0, 3])
        self.assertEqual(done[0][0]["word"], "a")
        self.assertEqual(asrcheck._load_checkpoint(None), {})

    def test_split_by_speaker_uses_diarisation_and_sentence(self):
        """Kétbeszélős sor (kötőjel nélkül): diarizáció + mondathatár vág.

        Mért hiba: a 341 s-os sorban a kép szerint férfi beszél, a sort mégis
        a nő dominálta, ezért egy női hangon ment ki. A vágás ideje a
        diarizációs váltás, a szövegé a hozzá legközelebbi mondathatár.
        """
        cue = subs.Cue(index=7, start=10.0, end=16.0,
                       text="Nos, itt alhatsz. Igen, szia.")
        runs = [(1, 9.5, 12.6), (2, 12.6, 16.5)]
        parts = utterances.split_by_speaker(cue, runs)
        self.assertIsNotNone(parts)
        self.assertEqual(len(parts), 2)
        self.assertAlmostEqual(parts[0].end, 12.6, places=2)
        self.assertEqual(parts[0].text, "Nos, itt alhatsz.")
        self.assertEqual(parts[1].text, "Igen, szia.")
        # egy beszélő: nincs vágás
        self.assertIsNone(utterances.split_by_speaker(cue, [(1, 9.5, 16.5)]))
        # túl rövid sor: nem vágunk
        self.assertIsNone(utterances.split_by_speaker(cue, runs,
                                                      {"split_min_s": 30.0}))
        # rövid átszivárgó forduló nem számít külön beszélőnek
        self.assertIsNone(utterances.split_by_speaker(
            cue, [(1, 9.5, 15.6), (2, 15.6, 16.5)], min_run=1.0))

    def test_multi_diar_roundtrip_and_gate_metric(self):
        u = utterances.Utterance(uid="E01:1", ep="E01", start=0.0, end=2.0)
        self.assertEqual(u.multi_diar, 0)
        u.multi_diar = 2
        back = utterances.Utterance.from_dict(u.to_dict())
        self.assertEqual(back.multi_diar, 2)
        from dubstudio import qa as _qa
        self.assertIn("voice_overlap_max", _qa.DEFAULT_GATES)

    def test_recover_empties_hears_short_line_and_maps_time(self):
        """A 29 s-os ablakból kimaradó rövid sor második felismerése.

        Mért hiba (E04, Parakeet): 29 üres sor, mindegyikben van beszéd; a
        soronkénti kivágásra a modell ad szöveget. A teszt azt rögzíti, hogy
        (a) csak az „üres” sort hallgatja meg, (b) a szó a globális idő-
        tengelyre kerül, (c) a meglévő szavak megmaradnak.
        """
        import tempfile

        import soundfile as sf

        class _Res:
            tokens = [" Tal", "án"]
            timestamps = [0.10, 0.30]

        class _Model:
            seen: list[int] = []

            def recognize(self, seg):
                _Model.seen.append(len(seg))
                return _Res()

        class _Tr:
            model_id = "fake-parakeet"
            _model = _Model()

        e = plan.Entry(i=7, uid="E04:0007", ep="E04", hu="Talán.",
                       cue_start=0.60, cue_end=1.50)
        old = [{"word": "előtte", "start": 0.05, "end": 0.2}]
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "a.wav"
            sf.write(str(wav), np.zeros(16000 * 2, dtype=np.float32), 16000)
            merged, got = asrcheck._recover_empties(
                [e], [{"i": 7, "kind": "empty"}], list(old), wav, _Tr(),
                verbose=False)
        self.assertEqual(got, 1)
        self.assertEqual(len(_Model.seen), 1, "egy sor = egy hívás")
        self.assertEqual(merged[0]["word"], "előtte", "a régi szó megmarad")
        got_words = [w for w in merged if w["word"] == "Talán"]
        self.assertEqual(len(got_words), 1)
        # a kivágás a cue_start - RECOVER_PAD-nál kezdődik: 0,60 - 0,35 = 0,25
        self.assertAlmostEqual(got_words[0]["start"], 0.35, places=2)

    def test_recover_enabled_default_and_switch(self):
        class _P:
            def __init__(self, asr):
                self.asr = asr

        self.assertTrue(asrcheck._recover_enabled(_P({"mode": "parakeet"})))
        self.assertFalse(asrcheck._recover_enabled(_P({"recover_empty": False})))

    def test_pipe_windows_batch_is_one_call_and_same_words(self):
        segs = [np.zeros(1600, dtype=np.float32) for _ in range(4)]
        serial, batched = _FakeWhisper(), _FakeWhisper()
        a = asrcheck._pipe_windows(serial, segs, 1, verbose=False)
        b = asrcheck._pipe_windows(batched, segs, 4, verbose=False)
        self.assertEqual(serial.calls, 4, "batch=1: ablakonként egy hívás")
        self.assertEqual(batched.calls, 1, "batch=4: egy hívás a 4 ablak")
        self.assertEqual(a, b, "azonos szavak mindkét úton")
        self.assertTrue(a)

    def test_pipe_windows_falls_back_to_serial_when_batch_unsupported(self):
        segs = [np.zeros(1600, dtype=np.float32) for _ in range(4)]
        reference = asrcheck._pipe_windows(_FakeWhisper(), segs, 1,
                                           verbose=False)
        bad = _FakeWhisper(reject_list=True)
        out = asrcheck._pipe_windows(bad, segs, 4, verbose=False)
        self.assertEqual(bad.list_calls, 1, "egy próbálkozás, aztán soronként")
        self.assertEqual(bad.calls, 1 + 4, "1 sikertelen köteg + 4 soronkénti")
        self.assertEqual(out, reference, "a visszaesés ugyanazt adja")

    def test_windowed_batch_matches_serial_end_to_end(self):
        import tempfile
        from pathlib import Path

        import soundfile as sf

        sr = 16000
        t = np.arange(int(sr * 70.0), dtype=np.float32) / sr
        x = (0.1 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        with tempfile.TemporaryDirectory() as d:
            wav = Path(d) / "t.wav"
            sf.write(str(wav), x, sr)
            a, ta = asrcheck._transcribe_windowed(_FakeWhisper(), wav,
                                                  verbose=False, batch=1)
            b, tb = asrcheck._transcribe_windowed(_FakeWhisper(), wav,
                                                  verbose=False, batch=4)
        self.assertTrue(a, "vannak felismert szavak")
        self.assertEqual(a, b, "ablak-rács + igazítás: azonos szavak")
        self.assertEqual(ta, tb)


class TestWordTiming(unittest.TestCase):
    """Szó-idő ellenőrzés (``words-audit``): szerkezet, energia, tempó, támasz.

    A lényeg, hogy a felirat-sáv önmagában szűkebb a beszédnél: a 0,35 s-os
    illesztő margón belüli kilógás **nem** hiba, a margón túli viszont az.
    """

    def _seg(self, i, a, b):
        return {"id": i, "start": a, "end": b, "text": "x"}

    def test_word_outside_alignment_window_is_a_failure(self):
        segs = [self._seg(0, 1.0, 2.0)]
        ok = [{"word": "שלום", "start": 1.2, "end": 1.7, "source_cue": 0}]
        bad = [{"word": "שלום", "start": 0.3, "end": 1.0, "source_cue": 0}]
        self.assertEqual(words.check_spans(ok, segs)["rates"]["inside_window"], 1.0)
        r = words.check_spans(bad, segs)
        self.assertLess(r["rates"]["inside_window"], 1.0)
        self.assertEqual(len(r["outside_window"]), 1)

    def test_cue_overflow_alone_is_only_informational(self):
        segs = [self._seg(0, 1.0, 2.0)]
        w = [{"word": "שלום", "start": 1.7, "end": 2.3, "source_cue": 0}]
        r = words.check_spans(w, segs)
        self.assertEqual(r["rates"]["inside_window"], 1.0)
        self.assertEqual(len(r["outside_cue"]), 1)
        self.assertAlmostEqual(r["cue_overflow_s"]["max"], 0.3, places=3)

    def test_overlapping_words_break_the_no_overlap_rate(self):
        segs = [self._seg(0, 0.0, 3.0)]
        w = [{"word": "אבג", "start": 0.5, "end": 1.5, "source_cue": 0},
             {"word": "דהו", "start": 1.0, "end": 1.8, "source_cue": 0}]
        r = words.check_spans(w, segs)
        self.assertEqual(len(r["overlap"]), 1)
        self.assertEqual(r["rates"]["no_overlap"], 0.5)

    def test_silent_word_is_detected_against_its_cue_energy(self):
        segs = [self._seg(0, 0.5, 3.5)]
        w = [{"word": "אבג", "start": 1.0, "end": 1.3, "source_cue": 0},
             {"word": "דהו", "start": 3.0, "end": 3.3, "source_cue": 0}]
        rms = np.full(400, 0.05, dtype=np.float32)      # 10 ms hop, 4 s
        rms[300:330] = 0.0005                           # 3,0-3,3 s: csend
        r = words.check_spans(w, segs, rms=rms, hop_s=0.01)
        self.assertEqual([x["word"] for x in r["silent"]], ["דהו"])

    def test_tempo_outliers_are_flagged(self):
        segs = [self._seg(0, 0.0, 10.0)]
        w = [{"word": "אבגדהוזחטי", "start": 0.0, "end": 0.2, "source_cue": 0},
             {"word": "אבג", "start": 4.0, "end": 6.0, "source_cue": 0},
             {"word": "אב", "start": 8.0, "end": 8.5, "source_cue": 0}]
        r = words.check_spans(w, segs)
        self.assertEqual(len(r["too_fast"]), 1)
        self.assertEqual(len(r["too_slow"]), 1)
        self.assertAlmostEqual(r["too_fast"][0]["cps"], 50.0, places=1)

    def test_very_short_word_is_not_a_tempo_outlier(self):
        segs = [self._seg(0, 0.0, 5.0)]
        w = [{"word": "של", "start": 1.0, "end": 1.04, "source_cue": 0}]
        r = words.check_spans(w, segs)
        self.assertEqual(len(r["too_fast"]), 0, "az egy betűs szó nem tempó-hiba")
        self.assertEqual(len(r["too_short"]), 1)

    def test_weak_alignment_support_is_reported(self):
        segs = [self._seg(0, 0.0, 5.0)]
        w = [{"word": "אבג", "start": 0.0, "end": 0.5, "score": 0.01, "source_cue": 0},
             {"word": "דהו", "start": 0.5, "end": 1.0, "score": 0.02, "source_cue": 0}]
        sup = words.check_spans(w, segs)["support"]
        self.assertTrue(sup["weak"])
        self.assertAlmostEqual(sup["median"], 0.015, places=4)
        self.assertEqual(sup["share_below"], 1.0)

    def test_no_words_is_not_a_crash(self):
        r = words.check_spans([], [self._seg(0, 0.0, 1.0)])
        self.assertEqual(r["n_words"], 0)
        self.assertEqual(r["rates"]["inside_window"], 0.0)
        self.assertIsNone(r["support"])


class TestEnglishWordTiming(unittest.TestCase):
    """Angol fő hang: ASR-szóidő (Parakeet tokenek) → szóhatárok.

    A tokenek szubszavak, szókezdő szóközzel; a szó vége nem nyúlhat a
    következő szóba, és hosszú csend nem csinálhat 30 s-os "szót".
    """

    def test_subword_tokens_form_words(self):
        words = speaker_audit.tokens_to_words(
            [" And", " now", " it", "'", "s", " qu", "iet", "."],
            [11.28, 11.6, 12.0, 12.08, 12.16, 12.32, 12.72, 12.88])
        self.assertEqual([w["word"] for w in words], ["And", "now", "it's", "quiet."])
        self.assertEqual(words[0]["start"], 11.28)
        self.assertGreaterEqual(words[0]["end"], 11.28)

    def test_word_end_never_overlaps_next_word(self):
        words = speaker_audit.tokens_to_words([" a", " b", " c"], [1.0, 1.02, 1.04])
        for a, b in zip(words, words[1:]):
            self.assertLessEqual(a["end"], b["start"], "a szavak nem fedhetik egymást")

    def test_long_pause_does_not_stretch_a_word(self):
        words = speaker_audit.tokens_to_words([" hello", " world"], [5.0, 40.0])
        self.assertLess(words[0]["end"] - words[0]["start"], 1.0,
                        "a csend nem nyújthatja meg az első szót")

    def test_english_subtitle_normalisation_keeps_text(self):
        self.assertEqual(speaker_audit.normalize_english("<i>Hello</i> [door slams]"), "Hello")
        self.assertEqual(speaker_audit.normalize_english("♪ music ♪"), "♪ music ♪")


class TestReviewConfirmation(unittest.TestCase):
    """Emberi jóváhagyás → galéria: e nélkül a C### nem marad stabil.

    A galéria a tekintély: a megerősített karakter **példány-mintái** is
    eltárolódnak, mert a puszta centroid két különböző beszélőt is 0,8–0,9
    koszinusszal közelít egymáshoz.
    """

    class _Project:
        name = "teszt"

        def __init__(self, work):
            self.work = work

        def ep_dir(self, tag):
            p = self.work / tag
            p.mkdir(parents=True, exist_ok=True)
            return p

    def test_confirmed_speaker_stores_exemplars_and_alias(self):
        import tempfile

        samples = [[1.0, 0.0] + [0.0] * 6, [1.0, 0.0] + [0.0] * 6]
        with tempfile.TemporaryDirectory() as d:
            p = self._Project(Path(d))
            review.set_speaker(p, "E06", "SPEAKER_00", "accepted", char_id="C004",
                               gender="M", voice="SUP_M3", samples=samples,
                               seconds=4.0)
            gal = identity.load_gallery(Path(d) / "identity-gallery.json")
            row = gal["characters"]["C004"]
            self.assertEqual(len(row["samples"]), 2)
            self.assertEqual(row["episodes"], ["E06"])
            self.assertEqual(gal["aliases"]["E06"]["SPEAKER_00"], "C004")
            self.assertEqual(len(row["centroid"]), 8)

    def test_second_episode_alias_points_to_the_same_character(self):
        import tempfile

        samples = [[0.0, 1.0] + [0.0] * 6]
        with tempfile.TemporaryDirectory() as d:
            p = self._Project(Path(d))
            review.set_speaker(p, "E06", "SPEAKER_00", "accepted", char_id="C004",
                               samples=samples, seconds=4.0)
            review.set_speaker(p, "E07", "SPEAKER_03", "corrected", char_id="C004",
                               samples=samples, seconds=2.0)
            gal = identity.load_gallery(Path(d) / "identity-gallery.json")
            self.assertEqual(sorted(gal["characters"]["C004"]["episodes"]),
                             ["E06", "E07"])
            self.assertEqual(gal["aliases"]["E07"]["SPEAKER_03"], "C004")
            self.assertAlmostEqual(gal["characters"]["C004"]["seconds"], 6.0, places=2)

    def test_rejected_speaker_gets_no_character(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            p = self._Project(Path(d))
            review.set_speaker(p, "E06", "SPEAKER_10", "rejected")
            gal = identity.load_gallery(Path(d) / "identity-gallery.json")
            self.assertIsNone(gal["aliases"]["E06"]["SPEAKER_10"])
            self.assertEqual(gal["characters"], {})
            self.assertEqual(review.load(p, "E06")["speakers"]["SPEAKER_10"]["decision"],
                             "rejected")

    def test_review_overrides_automatic_status(self):
        ident = {"speakers": {"SPEAKER_00": {"status": "weak", "char_id": None,
                                              "candidates": [], "review_required": True}}}
        review_doc = {"speakers": {"SPEAKER_00": {"decision": "corrected",
                                                   "char_id": "C005", "gender": "F",
                                                   "voice": "SUP_F2"}}}
        out = review.apply(ident, review_doc)
        row = out["speakers"]["SPEAKER_00"]
        self.assertEqual(row["char_id"], "C005")
        self.assertEqual(row["status"], "corrected")
        self.assertFalse(row["review_required"])
        self.assertEqual(out["counts"], {"corrected": 1})
        self.assertEqual(out["pending_review"], 0)


class TestPLDA(unittest.TestCase):
    """PLDA kalibráció: fit/score/decide és visszaesés."""

    def _vecs(self):
        # 3 osztály, osztályonként 4 minta, elkülönülő irányok
        import numpy as np
        rng = np.random.default_rng(0)
        out, labels = [], []
        for c in range(3):
            center = np.zeros(8, dtype=np.float32)
            center[c * 2] = 1.0
            for _ in range(4):
                v = center + 0.05 * rng.normal(size=8)
                out.append((v / np.linalg.norm(v)).astype(np.float32))
                labels.append(c)
        return out, labels

    def test_fit_and_decide_same(self):
        vecs, labels = self._vecs()
        m = plda.PLDAModel.fit(vecs, labels)
        self.assertTrue(m.fitted)
        # ugyanazon osztályból két vektor -> same
        # (első és második minta az 1. osztályból)
        same_idx = [i for i, l in enumerate(labels) if l == 0]
        r = m.decide(vecs[same_idx[0]], vecs[same_idx[1]])
        self.assertEqual(r["decision"], "same")

    def test_decide_different(self):
        vecs, labels = self._vecs()
        m = plda.PLDAModel.fit(vecs, labels)
        i0 = [i for i, l in enumerate(labels) if l == 0][0]
        # a tipikus különböző párnál is jóval eltérőbb pár → different
        r = m.decide(vecs[i0], -np.asarray(vecs[i0]))
        self.assertEqual(r["decision"], "different")
        # tipikus különböző pár (0. vs 1. osztály): sosem mondható azonosnak
        i1 = [i for i, l in enumerate(labels) if l == 1][0]
        self.assertNotEqual(m.decide(vecs[i0], vecs[i1])["decision"], "same")

    def test_decide_uses_z_thresholds(self):
        """Beállítás nélkül a mért ``Z_SAME``/``Z_DIFF`` alapérték dönt."""
        vecs, labels = self._vecs()
        m = plda.PLDAModel.fit(vecs, labels)
        same_idx = [i for i, l in enumerate(labels) if l == 0]
        r = m.decide(vecs[same_idx[0]], vecs[same_idx[1]])
        self.assertEqual(r["z_same"], plda.Z_SAME)
        self.assertEqual(r["z_diff"], plda.Z_DIFF)
        # a hívó z-küszöbe előbbre való az alapértéknél
        r = m.decide(vecs[same_idx[0]], vecs[same_idx[1]], z_same=99.0)
        self.assertEqual(r["z_same"], 99.0)
        self.assertNotEqual(r["decision"], "same")
        # a modell mért operációs pontja (calibrate) is előbbre való
        m.z_same = 5.5
        r = m.decide(vecs[same_idx[0]], vecs[same_idx[1]])
        self.assertEqual(r["z_same"], 5.5)
        self.assertEqual(r["z_diff"], plda.Z_DIFF)

    def test_score_scale_is_null_normalized(self):
        """A pontszám a **null-eloszlásból mért szórásban** van (z)."""
        vecs, labels = self._vecs()
        m = plda.PLDAModel.fit(vecs, labels)
        self.assertTrue(m.fitted)
        self.assertGreater(m.n_nt_pairs, 0)
        self.assertGreater(m.nt_sd, 0.0)
        i0 = [i for i, l in enumerate(labels) if l == 0]
        i1 = [i for i, l in enumerate(labels) if l == 1][0]
        # azonos osztály: jóval a null-eloszlás fölött
        self.assertGreater(m.z(m.score(vecs[i0[0]], vecs[i0[1]])), 2.0)
        # tipikus különböző pár: a null-eloszlás közelében
        self.assertLess(abs(m.z(m.score(vecs[i0[0]], vecs[i1]))), 2.0)

    def test_roc_eer_and_calibrate(self):
        """A küszöb mérésből jön (EER), nem örökölt koszinusz-érték."""
        rng = np.random.default_rng(1)
        target = list(rng.normal(0.8, 0.05, 200))
        non = list(rng.normal(0.2, 0.08, 400))
        stats = plda.roc_eer(target, non)
        self.assertLess(stats["eer"], 0.05)
        self.assertGreater(stats["z"], 2.0)
        m = plda.PLDAModel.fit(*self._vecs())
        m.calibrate(target, non)
        self.assertGreater(m.z_same, m.z_diff)
        self.assertGreater(m.z_same, 2.0)
        d = m.decide(*self._vecs()[0][:2])
        self.assertIn("z", d)

    def test_roundtrip_serialization(self):
        vecs, labels = self._vecs()
        m = plda.PLDAModel.fit(vecs, labels)
        m2 = plda.PLDAModel.from_dict(m.to_dict())
        self.assertTrue(m2.fitted)
        self.assertEqual(m2.n_classes, m.n_classes)
        # döntés konzisztens
        i0 = [i for i, l in enumerate(labels) if l == 0][0]
        i1 = [i for i, l in enumerate(labels) if l == 1][0]
        self.assertEqual(m.decide(vecs[i0], vecs[i1])["decision"],
                         m2.decide(vecs[i0], vecs[i1])["decision"])

    def test_unfitted_falls_back(self):
        m = plda.PLDAModel()
        self.assertFalse(m.fitted)
        r = m.decide(np.zeros(4, dtype=np.float32), np.zeros(4, dtype=np.float32))
        self.assertIn(r["decision"], ("same", "different", "uncertain"))

    def test_fit_plda_helper(self):
        # utterance-ek klaszterrel
        import numpy as np
        vecs, labels = self._vecs()
        utts = []
        for i, (v, l) in enumerate(zip(vecs, labels)):
            u = utterances.Utterance(uid=f"E01:{i:04d}", ep="E01", start=float(i),
                                     end=float(i + 1), text="x")
            u.emb = v
            u.cluster = l
            utts.append(u)
        model = clustering.fit_plda(utts)
        self.assertIsNotNone(model)
        self.assertTrue(model.fitted)

    def test_crossfit_never_sees_the_pair(self):
        """Körkörösség ellen: a pár egyik klaszterét sem látta a modell."""
        rng = np.random.default_rng(2)
        utts = []
        for c in range(12):
            center = np.zeros(24, dtype=np.float32)
            center[c] = 1.0
            for k in range(4):
                u = utterances.Utterance(uid=f"E01:{c:02d}{k}", ep="E01",
                                         start=float(10 * c + k),
                                         end=float(10 * c + k) + 1.0, text="x")
                v = center + 0.05 * rng.normal(size=24)
                u.emb = (v / np.linalg.norm(v)).astype(np.float32)
                u.cluster = c
                utts.append(u)
        cf = clustering.plda_crossfit(utts)
        self.assertIsNotNone(cf)
        m = cf.model_for(0, 1)
        self.assertIsNotNone(m)
        self.assertNotIn(0, m.labels)
        self.assertNotIn(1, m.labels)
        # azonos foldban lévő pár is kap modellt, amely nem látta őket
        same = cf.model_for(0, 3)
        self.assertIsNotNone(same)
        self.assertNotIn(0, same.labels)
        self.assertNotIn(3, same.labels)
        # ismeretlen klaszter: nincs modell → a hívó AS-Norm-ra esik vissza
        self.assertIsNone(cf.model_for(0, 99))

    def test_verify_pair_plda_returns_plda_key(self):
        vecs, labels = self._vecs()
        m = plda.PLDAModel.fit(vecs, labels)
        a = vecs[0]
        b = vecs[[i for i, l in enumerate(labels) if l == 1][0]]
        r = clustering.verify_pair_plda(a, b, plda_model=m)
        self.assertIn("plda", r)
        self.assertIn(r["decision"], ("same", "different", "uncertain"))


class TestAsdIntegration(unittest.TestCase):
    """Képi ASD evidence alkalmazás az utterance-ekre."""

    def test_apply_evidence_attaches_asd(self):
        u = utterances.Utterance(uid="E01:0001", ep="E01", start=10.0, end=12.0,
                                 text="szia")
        rows = [{"t": 10.0, "end": 12.0, "uid": "E01:0001", "faces": 1,
                 "light": {"speaking": 0, "verdict": "arc #0 beszél (p 0.9)",
                           "cands": [{"face": 0, "prob": 0.9, "size": 100,
                                      "spans": [[10.0, 11.5]]}]}}]
        n = asd_integration.apply_asd_evidence([u], rows)
        self.assertEqual(n, 1)
        self.assertIsNotNone(u.asd)
        self.assertEqual(u.asd["verdict"], "arc #0 beszél (p 0.9)")
        self.assertAlmostEqual(u.asd["prob"], 0.9, places=3)
        self.assertEqual(u.asd["face"], 0)

    def test_metrics_counts(self):
        u1 = utterances.Utterance(uid="E01:0001", ep="E01", start=0.0, end=1.0, text="a")
        u2 = utterances.Utterance(uid="E01:0002", ep="E01", start=2.0, end=3.0, text="b")
        u1.asd = {"verdict": "arc #0 beszél (p 0.9)", "prob": 0.9, "face": 0}
        u2.asd = {"verdict": "nincs beszélő arc a képen", "prob": None, "face": None}
        m = asd_integration.asd_metrics([u1, u2])
        self.assertEqual(m["n"], 2)
        self.assertEqual(m["visible_speaker"], 1)
        # "nincs beszélő arc a képen" = képen kívüli beszélő (a riport is így
        # számolja); a korábbi kód ezt egyik kategóriába sem tette.
        self.assertEqual(m["offscreen_speaker"], 1)
        self.assertEqual(m["no_face"], 0)

    def test_load_asd_results_missing(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(asd_integration.load_asd_results(Path(tmp)), [])


class TestWordAssignment(unittest.TestCase):
    """Szó-szintű hozzárendelés."""

    def test_assign_basic(self):
        turns = [{"start": 0.0, "end": 1.0, "speaker": "S0"},
                 {"start": 0.9, "end": 1.8, "speaker": "S1"}]
        words = [{"word": "a", "start": 0.2, "end": 0.6},
                 {"word": "b", "start": 0.9, "end": 1.0},
                 {"word": "c", "start": 2.0, "end": 2.2}]
        ws = word_assignment.assign(words, turns)
        self.assertEqual(ws[0].speaker, "S0")
        self.assertEqual(ws[1].confidence, "ambiguous_overlap")
        self.assertEqual(ws[2].confidence, "no_turn_overlap")

    def test_segments_from_word_speakers(self):
        ws = [word_assignment.WordSpeaker("a", 0.0, 0.5, "S0", "x"),
              word_assignment.WordSpeaker("b", 0.6, 1.0, "S0", "x"),
              word_assignment.WordSpeaker("c", 1.1, 1.5, "S1", "x")]
        segs = word_assignment.segments_from_word_speakers(ws)
        self.assertEqual(len(segs), 2)
        self.assertEqual(segs[0]["words"], ["a", "b"])
        self.assertEqual(segs[1]["speaker"], "S1")

    def test_split_points(self):
        ws = [word_assignment.WordSpeaker("a", 0.0, 0.5, "S0", "x"),
              word_assignment.WordSpeaker("b", 0.6, 1.0, "S1", "x")]
        pts = word_assignment.split_points(ws)
        self.assertEqual(pts, [0.6])

    def test_coverage_report(self):
        ws = [word_assignment.WordSpeaker("a", 0.0, 0.5, "S0", "turn_overlap_only"),
              word_assignment.WordSpeaker("b", 0.6, 1.0, None, "ambiguous_overlap")]
        cr = word_assignment.coverage_report(ws)
        self.assertEqual(cr["total"], 2)
        self.assertEqual(cr["assigned"], 1)
        self.assertEqual(cr["ambiguous"], 1)
        self.assertAlmostEqual(cr["coverage"], 0.5)

    def test_split_utterance_by_words(self):
        u = utterances.Utterance(uid="E01:0001", ep="E01", start=0.0, end=2.0, text="a b c d")
        u.cluster = 1
        ws = [word_assignment.WordSpeaker("a", 0.0, 0.5, "S0", "turn_overlap_only"),
              word_assignment.WordSpeaker("b", 0.6, 1.0, "S0", "turn_overlap_only"),
              word_assignment.WordSpeaker("c", 1.1, 1.5, "S1", "turn_overlap_only"),
              word_assignment.WordSpeaker("d", 1.6, 2.0, "S1", "turn_overlap_only")]
        out = word_assignment.split_utterance_by_words(u, ws)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].text, "a b")
        self.assertEqual(out[1].text, "c d")

    def test_split_keeps_short_middle_text(self):
        """A rövid középső szegmens szövege nem veszhet el."""
        u = utterances.Utterance(uid="E01:0001", ep="E01", start=0.0, end=3.0,
                                 text="a b c d e f")
        ws = [word_assignment.WordSpeaker("a", 0.0, 0.5, "S0", "x"),
              word_assignment.WordSpeaker("b", 0.5, 1.0, "S0", "x"),
              word_assignment.WordSpeaker("c", 1.0, 1.4, "S1", "x"),
              word_assignment.WordSpeaker("d", 1.4, 2.0, "S0", "x"),
              word_assignment.WordSpeaker("e", 2.0, 2.5, "S0", "x"),
              word_assignment.WordSpeaker("f", 2.5, 3.0, "S0", "x")]
        out = word_assignment.split_utterance_by_words(u, ws)
        self.assertIn("c", " ".join(x.text for x in out).split())

    def test_auto_marks_splits_cue_by_char(self):
        cue = _cue(1, 0.0, 4.0, "a b c d")
        u = utterances.Utterance(uid="E01:0001", ep="E01", start=0.0, end=4.0,
                                 cues=[cue], text="a b c d")
        u.diar, u.char = 7, "C001"
        turns = [{"start": 0.0, "end": 2.0, "speaker": "7"},
                 {"start": 2.0, "end": 4.0, "speaker": "9"}]
        marks = word_assignment.auto_marks([u], turns,
                                          {"7": "C001", "9": "C002"})
        got = marks[("E01:0001", 1)]
        self.assertEqual([m["char"] for m in got], ["C001", "C002"])
        self.assertEqual(got[0]["words"], [0, 1])
        self.assertEqual(got[1]["words"], [2, 3])

    def test_auto_marks_no_split_when_same_char(self):
        """Két nyers címke, de egy szereplő: nem vághatunk ketté."""
        cue = _cue(1, 0.0, 4.0, "a b c d")
        u = utterances.Utterance(uid="E01:0001", ep="E01", start=0.0, end=4.0,
                                 cues=[cue], text="a b c d")
        u.diar, u.char = 7, "C001"
        turns = [{"start": 0.0, "end": 2.0, "speaker": "7"},
                 {"start": 2.0, "end": 4.0, "speaker": "9"}]
        marks = word_assignment.auto_marks([u], turns,
                                          {"7": "C001", "9": "C001"})
        self.assertEqual(marks, {})

    def test_plan_build_applies_auto_marks(self):
        cue = _cue(1, 0.0, 4.0, "a b c d")
        u = utterances.Utterance(uid="E01:0001", ep="E01", start=0.0, end=4.0,
                                 cues=[cue], text="a b c d")
        u.char, u.voice, u.cluster = "C001", "V1", 1
        marks = {("E01:0001", 1): [
            {"char": "C001", "words": [0, 1], "kind": "word_auto"},
            {"char": "C002", "words": [2, 3], "kind": "word_auto"}]}
        entries = plan.build([u], verbose=False,
                             voices={"C001": "V1", "C002": "V2"},
                             auto_marks=marks)
        self.assertEqual([e.char for e in entries], ["C001", "C002"])
        self.assertFalse(any(e.manual for e in entries))

    def test_plan_manual_mark_beats_auto_marks(self):
        cue = _cue(1, 0.0, 4.0, "a b c d")
        u = utterances.Utterance(uid="E01:0001", ep="E01", start=0.0, end=4.0,
                                 cues=[cue], text="a b c d")
        u.char, u.voice, u.cluster = "C001", "V1", 1
        auto = {("E01:0001", 1): [
            {"char": "C001", "words": [0, 1], "kind": "word_auto"},
            {"char": "C002", "words": [2, 3], "kind": "word_auto"}]}
        labels = {("E01:0001", 1): [{"char": "C003", "words": None,
                                     "kind": "char"}]}
        entries = plan.build([u], verbose=False,
                             voices={"C001": "V1", "C002": "V2", "C003": "V3"},
                             labels=labels, auto_marks=auto)
        self.assertEqual({e.char for e in entries}, {"C003"})
        self.assertTrue(all(e.manual for e in entries))

    def test_marks_saved_in_prepare_are_read_back(self):
        """A vágásjelölt a `prepare`-ben mentődik, a `dub` csak beolvassa."""
        import tempfile

        class _P:
            def __init__(self, root):
                self.root = Path(root)

            def ep_dir(self, tag):
                return self.root / tag

        with tempfile.TemporaryDirectory() as tmp:
            p = _P(tmp)
            cue = _cue(1, 0.0, 4.0, "a b c d")
            u = utterances.Utterance(uid="E01:0001", ep="E01", start=0.0, end=4.0,
                                     cues=[cue], text="a b c d")
            u.diar, u.char = 7, "C001"
            u2 = utterances.Utterance(uid="E01:0002", ep="E01", start=5.0, end=6.0,
                                      text="e")
            u2.diar, u2.char = 9, "C002"
            turns = [{"start": 0.0, "end": 2.0, "speaker": "7"},
                     {"start": 2.0, "end": 4.0, "speaker": "9"}]
            labeled = word_assignment.segments_by_label([u, u2], turns)
            word_assignment.save_marks(p, "E01", labeled)
            self.assertTrue(word_assignment.marks_path(p, "E01").is_file())
            marks = word_assignment.marks_for_project(p, "E01", [u, u2])
            got = marks[("E01:0001", 1)]
            self.assertEqual([m["char"] for m in got], ["C001", "C002"])
            self.assertEqual(got[0]["words"], [0, 1])
            self.assertEqual(got[1]["words"], [2, 3])

    def test_marks_for_project_without_file(self):
        import tempfile

        class _P:
            def __init__(self, root):
                self.root = Path(root)

            def ep_dir(self, tag):
                return self.root / tag

        with tempfile.TemporaryDirectory() as tmp:
            u = utterances.Utterance(uid="E01:0001", ep="E01", start=0.0, end=1.0,
                                     text="a")
            self.assertEqual(
                word_assignment.marks_for_project(_P(tmp), "E01", [u]), {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
