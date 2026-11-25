// Emberi ellenőrző felület: film + feliratok, soronkénti címkézés.
(function () {
  'use strict';
  const esc = (v) => Auris.esc(v);
  const $ = (id) => document.getElementById(id);
  const state = {
    projects: [],
    project: null,
    episode: null,
    data: null,
    lines: [],
    cast: [],
    voices: [],
    byKey: new Map(),
    active: null,
    sel: null,                 // {key, w0, w1} — w0 null = a teljes sor
    player: null,
    audioId: null,
    seq: 0,
    prepTimer: null,
    follow: true,
    previewTimer: null,
    editorId: null,
    filterPending: false,
    showAllPeople: false,
    busy: false,
  };
  //: ennyi szereplő fér ki kényelmesen a panelre (a billentyűk is ezekhez 1–9)
  const PEOPLE_LIMIT = 9;

  const params = () => new URLSearchParams({ project: state.project || '', episode: state.episode || '' });
  const projectEntry = () => state.projects.find((p) => (p.id || p.name) === state.project) || null;

  function setStatus(text, kind = '') {
    const el = $('rev-status');
    if (!el) return;
    el.textContent = text || '';
    el.className = 'review-status' + (kind ? ` is-${kind}` : '');
  }

  function fmt(t) {
    const s = Math.max(0, Number(t) || 0);
    const m = Math.floor(s / 60);
    return `${String(m).padStart(2, '0')}:${(s - m * 60).toFixed(1).padStart(4, '0')}`;
  }

  function colorFor(charId) {
    const person = state.cast.find((p) => p.char_id === charId);
    return person ? person.hue : 210;
  }

  function personName(charId) {
    const person = state.cast.find((p) => p.char_id === charId);
    return (person && person.name) || charId || '';
  }

  // ── projekt- és epizódválasztás ─────────────────────────────────────────
  async function loadProjects() {
    const payload = await Auris.api('/api/dub/projects');
    state.projects = payload.projects || [];
    const select = $('rev-project');
    if (!state.projects.length) {
      select.innerHTML = '<option value="">nincs projekt</option>';
      setStatus('Nincs dubstudio projekt (projects/*.json).', 'error');
      return;
    }
    const counts = {};
    state.projects.forEach((p) => { counts[p.name] = (counts[p.name] || 0) + 1; });
    select.innerHTML = state.projects.map((p) =>
      `<option value="${esc(p.id || p.name)}">${esc(p.name)}` +
      `${counts[p.name] > 1 ? ` · ${esc(p.id || p.name)}` : ''}</option>`).join('');
    const withEpisodes = state.projects.find((p) => (p.episodes || []).length) || state.projects[0];
    state.project = withEpisodes.id || withEpisodes.name;
    select.value = state.project;
    renderEpisodeOptions();
  }

  function renderEpisodeOptions() {
    const entry = projectEntry();
    const select = $('rev-episode');
    const episodes = (entry && entry.episodes) || [];
    if (!episodes.length) {
      select.innerHTML = '<option value="">nincs előkészített epizód</option>';
      state.episode = null;
      clearVideo();
      $('rev-lines').innerHTML = '';
      $('rev-empty').hidden = false;
      $('rev-empty').innerHTML = '<strong>Nincs előkészített epizód.</strong>' +
        '<span>Futtasd a Szinkron stúdióban az <em>Előkészítés</em> lépést, utána itt ' +
        'megjelennek a sorok.</span>';
      setStatus('Ehhez a projekthez még nincs kész felirat-előkészítés.', 'warn');
      return;
    }
    select.innerHTML = episodes.map((e) => `<option value="${esc(e)}">${esc(e)}</option>`).join('');
    state.episode = episodes[episodes.length - 1];
    select.value = state.episode;
  }

  // ── sorok betöltése ────────────────────────────────────────────────────
  async function loadEpisode() {
    if (!state.project || !state.episode) return;
    setStatus('Feliratok betöltése…');
    $('rev-empty').hidden = true;
    try {
      const data = await Auris.api(`/api/dub/lines?${params().toString()}`);
      state.data = data;
      state.lines = data.lines || [];
      state.cast = data.cast || [];
      state.voices = data.voices || [];
      state.byKey = new Map(state.lines.map((l) => [l.key, l]));
      state.active = null;
      state.sel = null;
      renderLines();
      renderCounts(data.stats || {});
      renderPeople();
      renderPanel();
      if (!data.has_utterances) {
        setStatus('Ehhez az epizódhoz nincs előkészített felirat (utterances.json).', 'warn');
      } else if (!state.lines.length) {
        setStatus('Az előkészítés nem adott feliratsort.', 'warn');
      } else {
        setStatus(`${state.lines.length} sor · ${data.asd_ready ? 'képi bizonyíték elérhető' : 'nincs képi bizonyíték'}`, 'ok');
      }
    } catch (err) {
      setStatus(`A feliratok nem tölthetők be: ${err.message}`, 'error');
      return;
    }
    await loadPlayer();
  }

  // ── lejátszó (a meglévő proxy-végpontokon) ─────────────────────────────
  function clearVideo() {
    if (state.prepTimer) clearTimeout(state.prepTimer);
    state.prepTimer = null;
    state.seq += 1;
    state.player = null;
    state.audioId = null;
    const video = $('rev-video');
    video.pause();
    video.removeAttribute('src');
    video.load();
    const select = $('rev-audio');
    select.innerHTML = '<option value="">— nincs betöltött média —</option>';
    select.disabled = true;
    $('rev-prep').hidden = true;
  }

  function showEmpty(message) {
    const box = $('rev-empty');
    if (!message) {
      box.hidden = true;
      box.textContent = '';
      return;
    }
    box.hidden = false;
    box.innerHTML = `<strong>A film nem játszható le.</strong><span>${esc(message)}</span>`;
  }

  function showPrep(show, text) {
    $('rev-prep').hidden = !show;
    if (text) $('rev-prep-text').textContent = text;
  }

  function setVideoSource(track) {
    const video = $('rev-video');
    showEmpty('');
    $('rev-empty').hidden = true;
    showPrep(false);
    const at = video.currentTime || 0;
    video.src = track.url;
    video.load();
    video.addEventListener('loadedmetadata', () => {
      if (at > 0 && at < video.duration) video.currentTime = at;
    }, { once: true });
  }

  async function activateAudio(audioId) {
    const track = ((state.player || {}).audio || []).find((a) => a.id === audioId);
    if (!track) return;
    state.audioId = audioId;
    const seq = ++state.seq;
    const select = $('rev-audio');
    if (select) select.value = audioId;
    if (track.cached) { setVideoSource(track); return; }
    showPrep(true, `A hangsáv előkészítése… (${track.label})`);
    try {
      const st = await Auris.api('/api/dub/player/prepare', {
        method: 'POST',
        body: { project: state.project, episode: state.episode, audio: audioId },
      });
      if (seq !== state.seq) return;
      if (st.state === 'ready') { track.cached = true; setVideoSource(track); return; }
      if (st.state === 'error') {
        showPrep(true, `A hangsáv nem készíthető el: ${st.error || 'ismeretlen hiba'}`);
        return;
      }
      pollPrep(audioId, track, seq);
    } catch (err) {
      if (seq !== state.seq) return;
      showPrep(true, `Az előkészítés nem indult: ${err.message}`);
    }
  }

  function pollPrep(audioId, track, seq) {
    if (seq !== state.seq) return;
    const q = new URLSearchParams({ project: state.project, episode: state.episode, audio: audioId });
    Auris.api(`/api/dub/player/status?${q.toString()}`).then((st) => {
      if (seq !== state.seq) return;
      if (st.state === 'ready') { track.cached = true; setVideoSource(track); return; }
      if (st.state === 'error') {
        showPrep(true, `A hangsáv nem készíthető el: ${st.error || 'ismeretlen hiba'}`);
        return;
      }
      const mb = (Number(st.part_bytes || 0) / (1024 * 1024)).toFixed(0);
      showPrep(true, `A hangsáv előkészítése… ${Number(st.elapsed || 0).toFixed(0)} mp` +
        (mb > 0 ? ` · ${mb} MB` : '') + ' (egyszer kell, utána azonnal indul)');
      state.prepTimer = setTimeout(() => pollPrep(audioId, track, seq), 1200);
    }).catch(() => {
      if (seq === state.seq) showPrep(true, 'Az állapot nem kérdezhető le — újrapróbálom…');
      if (seq === state.seq) state.prepTimer = setTimeout(() => pollPrep(audioId, track, seq), 2500);
    });
  }

  async function loadPlayer() {
    clearVideo();
    try {
      const info = await Auris.api(`/api/dub/player?${params().toString()}`);
      state.player = info;
      const select = $('rev-audio');
      select.innerHTML = (info.audio || []).map((a) =>
        `<option value="${esc(a.id)}">${esc(a.label)}</option>`).join('');
      select.disabled = !(info.audio || []).length;
      // a magyar szinkron az alapértelmezett: ezt kell ellenőrizni
      const preferred = (info.audio || []).find((a) => a.kind === 'dub') ||
        (info.audio || []).find((a) => a.kind === 'source') ||
        (info.audio || [])[0];
      if (preferred) await activateAudio(preferred.id);
      else showEmpty('A forrásnak nincs hangsávja.');
    } catch (err) {
      showEmpty(err.message);
    }
  }

  // ── a feliratok rajzolása ──────────────────────────────────────────────
  function wordsOf(text) {
    return String(text || '').split(/\s+/).filter(Boolean);
  }

  function markFor(line, index) {
    const marks = line.marks || [];
    const ranged = marks.find((m) => Array.isArray(m.words) && m.words[0] <= index && index <= m.words[1]);
    if (ranged) return ranged;
    return marks.find((m) => !Array.isArray(m.words)) || null;
  }

  //: egy sor akkor van eldöntve, ha valamelyik mark szereplőt nevez meg
  function hasCharMark(line) {
    return (line.marks || []).some((m) => m.kind === 'char' && m.char);
  }

  function isPending(line) {
    return !hasCharMark(line);
  }

  function markChip(mark) {
    if (!mark) return '';
    if (mark.kind === 'char') {
      return `<span class="rev-chip" style="--rev-hue:${colorFor(mark.char)}">${esc(mark.name || personName(mark.char))}</span>`;
    }
    if (mark.kind === 'offscreen') return '<span class="rev-chip is-off">képen kívül</span>';
    return '<span class="rev-chip is-unknown">nem tudom</span>';
  }

  function renderLine(line) {
    const li = document.createElement('li');
    li.className = 'review-line';
    li.dataset.key = line.key;
    const tokens = String(line.text || '').split(/(\s+)/).filter((t) => t.length);
    let w = -1;
    const text = tokens.map((token) => {
      if (/^\s+$/.test(token)) return token;
      w += 1;
      const mark = markFor(line, w);
      return `<span class="rev-word${mark ? ' has-mark' : ''}" data-w="${w}"` +
        (mark && mark.kind === 'char' ? ` style="--rev-hue:${colorFor(mark.char)}"` : '') +
        `>${esc(token)}</span>`;
    }).join('');
    const chips = (line.marks || []).map(markChip).join('');
    const auto = !line.marks.length && line.char
      ? `<span class="rev-chip is-auto" style="--rev-hue:${colorFor(line.char)}">gép: ${esc(line.name || line.char)}</span>`
      : '';
    const asd = line.asd
      ? `<span class="rev-asd is-${esc(line.asd.kind)}" title="Képi beszélő-ellenőrzés (bizonyíték)">${esc(line.asd.text)}</span>`
      : '';
    li.innerHTML =
      `<button type="button" class="review-line-time" data-seek="${line.start}" title="Ugrás ide">${fmt(line.start)}</button>` +
      `<div class="review-line-body"><p class="review-line-text">${text}</p>` +
      `<div class="review-line-meta">${chips}${auto}${asd}` +
      (line.offset ? `<span class="rev-chip is-offset">${line.offset > 0 ? '+' : ''}${line.offset.toFixed(2)} s</span>` : '') +
      `</div></div>`;
    if (state.filterPending && !isPending(line)) li.classList.add('is-filtered');
    return li;
  }

  function renderLines() {
    const list = $('rev-lines');
    const fragment = document.createDocumentFragment();
    state.lines.forEach((line) => fragment.appendChild(renderLine(line)));
    list.replaceChildren(fragment);
    paintSelection();
  }

  function replaceLine(line) {
    if (!line) return;
    const at = state.lines.findIndex((l) => l.key === line.key);
    if (at < 0) return;
    state.lines[at] = line;
    state.byKey.set(line.key, line);
    const old = $('rev-lines').querySelector(`li[data-key="${CSS.escape(line.key)}"]`);
    if (old) old.replaceWith(renderLine(line));
    paintSelection();
  }

  function lineEl(key) {
    return $('rev-lines').querySelector(`li[data-key="${CSS.escape(key)}"]`);
  }

  function paintSelection() {
    $('rev-lines').querySelectorAll('li.review-line').forEach((li) => {
      const key = li.dataset.key;
      li.classList.toggle('is-active', key === state.active);
      li.classList.toggle('is-selected', Boolean(state.sel && state.sel.key === key));
      li.querySelectorAll('.rev-word').forEach((span) => {
        const i = Number(span.dataset.w);
        const inSel = state.sel && state.sel.key === key && state.sel.w0 !== null &&
          i >= state.sel.w0 && i <= state.sel.w1;
        span.classList.toggle('is-sel', Boolean(inSel));
      });
    });
  }

  function renderCounts(stats) {
    const total = state.lines.length;
    const marked = state.lines.filter((l) => (l.marks || []).length).length;
    const offsets = state.lines.filter((l) => l.offset).length;
    const pending = state.lines.filter(isPending).length;
    $('rev-counts').textContent =
      `${marked} / ${total} sor megjelölve · ${offsets} elcsúszás` +
      (pending ? ` · ${pending} döntésre vár` : '') +
      (stats && stats.unknown ? ` · ${stats.unknown} „nem tudom”` : '');
  }

  // ── aktív sor és kijelölés ─────────────────────────────────────────────
  function lineAt(t) {
    let best = null;
    for (const line of state.lines) {
      if (line.start <= t && t <= line.end) return line;
      if (line.start <= t) best = line;
    }
    return best || state.lines[0] || null;
  }

  function setActive(key, opts = {}) {
    if (key === state.active && !opts.force) {
      paintSelection();
      return;
    }
    state.active = key;
    const line = state.byKey.get(key);
    if (line) {
      if (opts.seek) {
        const video = $('rev-video');
        if (video.src) video.currentTime = Math.max(0, line.start - 0.12);
      }
      if (opts.scroll !== false) scrollToLine(key);
      renderNow(line);
      renderPanel();
    }
    paintSelection();
  }

  function scrollToLine(key) {
    if (!state.follow) return;
    const li = lineEl(key);
    if (!li) return;
    // a görgethető elem maga a lista (nem a körítő doboz)
    const list = $('rev-lines');
    const row = li.getBoundingClientRect();
    const box = list.getBoundingClientRect();
    const delta = (row.top + row.height / 2) - (box.top + box.height / 2);
    list.scrollTo({ top: Math.max(0, list.scrollTop + delta), behavior: 'smooth' });
  }

  // a lejátszófejhez igazodó kiemelés: ugrásnál (seek) is azonnal
  function syncPlayhead() {
    const line = lineAt($('rev-video').currentTime);
    if (!line || line.key === state.active) return;
    state.active = line.key;
    if (state.follow) scrollToLine(line.key);
    renderNow(line);
    renderPanel();
    paintSelection();
  }

  function renderNow(line) {
    const box = $('rev-now');
    if (!line) { box.textContent = ''; return; }
    const name = line.marks && line.marks.length
      ? line.marks.map((m) => m.kind === 'char' ? esc(m.name || personName(m.char))
        : m.kind === 'offscreen' ? 'képen kívül' : 'nem tudom').join(' · ')
      : (line.char ? `gép szerint: ${esc(line.name || line.char)}` : 'nincs megjelölés');
    box.innerHTML = `<span class="review-now-time">${fmt(line.start)}</span>` +
      `<strong>${name}</strong><span class="review-now-text">${esc(line.text)}</span>`;
  }

  function selectionLine() {
    const key = (state.sel && state.sel.key) || state.active;
    return key ? state.byKey.get(key) || null : null;
  }

  function renderSelectionLabel() {
    const line = selectionLine();
    if (!line) {
      $('rev-selection').textContent = 'Nincs kijelölt sor';
      $('rev-clear-sel').hidden = true;
      return;
    }
    const words = wordsOf(line.text);
    let text = `#${line.n} · ${fmt(line.start)} · ${line.text}`;
    if (state.sel && state.sel.key === line.key && state.sel.w0 !== null) {
      const part = words.slice(state.sel.w0, state.sel.w1 + 1).join(' ');
      text = `#${line.n} · kijelölve: „${part}”`;
    }
    $('rev-selection').textContent = text;
    $('rev-clear-sel').hidden = false;
  }

  function renderPeople() {
    const line = selectionLine();
    const suggested = line && line.char ? line.char : null;
    const parts = [];
    if (suggested) {
      parts.push(`<button type="button" class="rev-person is-accept" data-accept="1"` +
        ` style="--rev-hue:${colorFor(suggested)}">Gép szerint: ${esc(personName(suggested))}` +
        ` <small>Enter</small></button>`);
    }
    const shown = state.showAllPeople ? state.cast : state.cast.slice(0, PEOPLE_LIMIT);
    parts.push(shown.map((p) => {
      const i = state.cast.indexOf(p);
      return `<span class="rev-person-wrap">` +
      `<button type="button" class="rev-person${p.char_id === suggested ? ' is-suggested' : ''}"` +
      ` data-char="${esc(p.char_id)}" data-voice="${esc(p.voice || '')}"` +
      ` data-pitch="${Number(p.pitch || 0)}" style="--rev-hue:${p.hue}"` +
      ` title="${esc(p.char_id)}${p.voice ? ' · ' + esc(p.voice) : ' · nincs hang'}` +
      ` — egy kattintás: a kijelölt rész meghallgatása, dupla: áttérés erre a hangra, Alt+kattintás: átnevezés">` +
      `<span class="rev-person-key">${i < 9 ? i + 1 : ''}</span>` +
      `<span class="rev-person-dot"></span>${esc(p.name || p.char_id)}` +
      `<small>${p.gender === 'F' ? 'nő' : p.gender === 'M' ? 'férfi' : ''}</small>` +
      (p.voice ? '' : `<small class="rev-novoice">nincs hang</small>`) +
      `</button>` +
      `<button type="button" class="rev-person-edit" data-edit="${esc(p.char_id)}"` +
      ` title="Név és magyar hang szerkesztése" aria-label="${esc(p.name || p.char_id)} szerkesztése">✎</button>` +
      `</span>`;
    }).join(''));
    if (state.cast.length > PEOPLE_LIMIT) {
      parts.push(`<button type="button" class="rev-person is-more" data-more="1">` +
        `${state.showAllPeople ? 'Kevesebb szereplő' : `További ${state.cast.length - PEOPLE_LIMIT} szereplő`}</button>`);
    }
    $('rev-people').innerHTML = parts.join('');
  }

  function renderPanel() {
    renderSelectionLabel();
    renderPeople();
    const line = selectionLine();
    $('rev-offset').value = line ? Number(line.offset || 0).toFixed(2) : '0';
    $('rev-offset-info').textContent = line && line.offset
      ? `mentett: ${line.offset > 0 ? '+' : ''}${line.offset.toFixed(2)} s`
      : '';
  }

  // ── a kijelölt feliratrész meghallgatása egy hangon ────────────────────
  function stopPreview() {
    const audio = $('rev-voice');
    if (audio) audio.pause();
  }

  function previewPerson(button) {
    const charId = button.dataset.char;
    const person = state.cast.find((p) => p.char_id === charId) || null;
    const voice = (person && person.voice) || button.dataset.voice || '';
    if (!voice) {
      setStatus(`${personName(charId)} — ehhez a szereplőhöz még nincs magyar hang.`, 'warn');
      return;
    }
    const line = selectionLine();
    if (!line) {
      setStatus('Előbb válassz egy feliratrészt: kattints a sorra, vagy jelölj ki szavakat.', 'warn');
      return;
    }
    const words = state.sel && state.sel.key === line.key && state.sel.w0 !== null
      ? [state.sel.w0, state.sel.w1] : null;
    const query = new URLSearchParams({
      project: state.project, episode: state.episode,
      uid: String(line.uid), cue: String(line.cue), voice,
      pitch: String(Number((person && person.pitch) || button.dataset.pitch || 0)),
    });
    if (words) query.set('words', `${words[0]}-${words[1]}`);
    const url = `/api/dub/lines/preview?${query.toString()}`;
    const audio = $('rev-voice');
    $('rev-video').pause();
    if (audio.src !== new URL(url, window.location.href).href) audio.src = url;
    else audio.currentTime = 0;
    setStatus(`${personName(charId)} · ${voice} — a kijelölt rész betöltése… ` +
      '(az első kérés betöltheti a TTS-motort, ez 1–2 perc lehet)', 'busy');
    audio.play()
      .then(() => setStatus(`${personName(charId)} · ${voice} — a kijelölt feliratrész szól.`, 'ok'))
      .catch(() => setStatus(`A hangminta nem indítható (${voice}).`, 'error'));
  }

  // ── mentés ─────────────────────────────────────────────────────────────
  async function post(url, body) {
    return Auris.api(url, { method: 'POST', body });
  }

  async function applyLabel(kind, person) {
    const line = selectionLine();
    if (!line || state.busy) return;
    const words = state.sel && state.sel.key === line.key && state.sel.w0 !== null
      ? [state.sel.w0, state.sel.w1] : null;
    const body = {
      project: state.project, episode: state.episode,
      uid: line.uid, cue: line.cue, kind, words,
    };
    if (person) {
      body.char_id = person.char_id;
      body.gender = person.gender || null;
      body.name = person.name || null;
    }
    state.busy = true;
    try {
      const res = await post('/api/dub/lines/label', body);
      replaceLine(res.line);
      renderCounts(res.stats || {});
      renderPanel();
      const what = kind === 'clear' ? 'törölve'
        : kind === 'offscreen' ? 'képen kívüli'
          : kind === 'unknown' ? 'nem tudom' : (person ? person.name : '');
      setStatus(`#${line.n} — ${what}`, 'ok');
    } catch (err) {
      setStatus(`A megjelölés nem menthető: ${err.message}`, 'error');
    } finally {
      state.busy = false;
    }
  }

  async function applyOffset(value) {
    const line = selectionLine();
    if (!line || state.busy) return;
    const offset = Math.max(-10, Math.min(10, Number(value) || 0));
    state.busy = true;
    try {
      const res = await post('/api/dub/lines/offset', {
        project: state.project, episode: state.episode,
        uid: line.uid, cue: line.cue, offset,
      });
      replaceLine(res.line);
      renderCounts(res.stats || {});
      renderPanel();
      setStatus(`#${line.n} elcsúszás: ${offset > 0 ? '+' : ''}${offset.toFixed(2)} s ` +
        '(a következő szinkronfutásnál érvényes)', 'ok');
    } catch (err) {
      setStatus(`Az elcsúszás nem menthető: ${err.message}`, 'error');
    } finally {
      state.busy = false;
    }
  }

  async function createPerson(gender) {
    const line = selectionLine();
    if (!line || state.busy) return;
    const name = $('rev-new-name').value.trim();
    const words = state.sel && state.sel.key === line.key && state.sel.w0 !== null
      ? [state.sel.w0, state.sel.w1] : null;
    state.busy = true;
    try {
      const res = await post('/api/dub/lines/label', {
        project: state.project, episode: state.episode,
        uid: line.uid, cue: line.cue, kind: 'new', gender, name, words,
      });
      const newChar = ((res.line || {}).marks || []).find((m) => m.char);
      if (res.cast) state.cast = res.cast;
      if (res.voices) state.voices = res.voices;
      state.data.cast = state.cast;
      $('rev-new-name').value = '';
      replaceLine(res.line);
      renderCounts(res.stats || {});
      renderPanel();
      const created = newChar && state.cast.find((p) => p.char_id === newChar.char);
      setStatus(`Új szereplő: ${newChar ? newChar.name : name}` +
        (created && created.voice ? ` · hang: ${created.voice}` : ' · nincs hang') +
        ' — a ✎ gombbal más hangot választhatsz.', 'ok');
    } catch (err) {
      setStatus(`Az új szereplő nem menthető: ${err.message}`, 'error');
    } finally {
      state.busy = false;
    }
  }

  // ── szereplő szerkesztése: név + magyar hang ───────────────────────────
  function applyPersonPayload(res) {
    if (res.cast) { state.data.cast = res.cast; state.cast = res.cast; }
    if (res.voices) state.voices = res.voices;
    if (res.lines) {
      state.lines = res.lines;
      state.byKey = new Map(state.lines.map((l) => [l.key, l]));
    }
    renderLines();
    renderCounts(res.stats || {});
    renderPanel();
  }

  function closePersonEditor() {
    state.editorId = null;
    const box = $('rev-person-editor');
    if (!box) return;
    box.hidden = true;
    box.innerHTML = '';
  }

  function openPersonEditor(charId) {
    const person = state.cast.find((p) => p.char_id === charId);
    const box = $('rev-person-editor');
    if (!person || !box) return;
    state.editorId = charId;
    const current = person.voice || '';
    const options = state.voices.map((v) =>
      `<option value="${esc(v.vid)}"${v.vid === current ? ' selected' : ''}>` +
      `${esc(v.label)} · ${v.gender === 'F' ? 'nő' : 'férfi'} · ${esc(v.vid)}` +
      `${v.model_present === false ? ' · nincs modell' : ''}</option>`).join('');
    box.hidden = false;
    box.innerHTML =
      `<span class="review-editor-title">${esc(person.name || charId)}` +
      ` <small>${esc(charId)} · ${person.gender === 'F' ? 'nő' : person.gender === 'M' ? 'férfi' : '?'}</small></span>` +
      `<label class="review-editor-field">Név` +
      ` <input id="rev-edit-name" class="text-input" maxlength="24" value="${esc(person.name || '')}"` +
      ` autocomplete="off"></label>` +
      `<label class="review-editor-field">Magyar hang` +
      ` <select id="rev-edit-voice" class="select-input">` +
      `<option value="">— nincs hang —</option>${options}</select></label>` +
      `<button id="rev-edit-save" class="btn btn-sm btn-primary" type="button">Mentés</button>` +
      `<button id="rev-edit-cancel" class="btn btn-sm btn-ghost" type="button">Mégse</button>`;
    $('rev-edit-save').addEventListener('click', savePersonEditor);
    $('rev-edit-cancel').addEventListener('click', closePersonEditor);
    const input = $('rev-edit-name');
    input.focus();
    input.select();
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); savePersonEditor(); }
      if (e.key === 'Escape') { e.preventDefault(); closePersonEditor(); }
    });
  }

  async function savePersonEditor() {
    const charId = state.editorId;
    if (!charId) return;
    const name = ($('rev-edit-name').value || '').trim();
    const voice = $('rev-edit-voice').value || '';
    try {
      const res = await post('/api/dub/lines/person', {
        project: state.project, episode: state.episode,
        char_id: charId, name, voice,
      });
      applyPersonPayload(res);
      closePersonEditor();
      setStatus(`${name || charId} — hang: ${voice || 'nincs'}`, 'ok');
    } catch (err) {
      setStatus(`A szereplő mentése nem sikerült: ${err.message}`, 'error');
    }
  }

  // ── események ──────────────────────────────────────────────────────────
  function bind() {
    $('rev-project').addEventListener('change', (e) => {
      state.project = e.target.value;
      renderEpisodeOptions();
      loadEpisode();
    });
    $('rev-episode').addEventListener('change', (e) => {
      state.episode = e.target.value;
      loadEpisode();
    });
    $('rev-audio').addEventListener('change', (e) => activateAudio(e.target.value));
    $('rev-reload').addEventListener('click', () => loadEpisode());

    const video = $('rev-video');
    $('rev-play').addEventListener('click', () => {
      if (video.paused) video.play().catch(() => setStatus('A böngésző nem engedte a lejátszást.', 'error'));
      else video.pause();
    });
    video.addEventListener('play', () => {
      $('rev-play').textContent = '❚❚ Szünet';
      stopPreview();
    });
    video.addEventListener('pause', () => { $('rev-play').textContent = '▶ Lejátszás'; });
    $('rev-back').addEventListener('click', () => { video.currentTime = Math.max(0, video.currentTime - 2); });
    $('rev-fwd').addEventListener('click', () => { video.currentTime = video.currentTime + 2; });
    $('rev-follow').addEventListener('change', (e) => {
      state.follow = e.target.checked;
      if (state.follow && state.active) scrollToLine(state.active);
    });
    video.addEventListener('timeupdate', () => {
      $('rev-time').textContent = fmt(video.currentTime);
      syncPlayhead();
    });
    // a csúszka/ugrás után a feliratsáv is odaugorjon (a timeupdate késhet)
    video.addEventListener('seeked', () => {
      $('rev-time').textContent = fmt(video.currentTime);
      syncPlayhead();
    });

    $('rev-prev').addEventListener('click', () => stepLine(-1));
    $('rev-next').addEventListener('click', () => stepLine(1));

    $('rev-lines').addEventListener('click', (e) => {
      const li = e.target.closest('li[data-key]');
      if (!li) return;
      const key = li.dataset.key;
      const line = state.byKey.get(key);
      if (!line) return;
      const word = e.target.closest('.rev-word');
      if (word) {
        const i = Number(word.dataset.w);
        if (e.shiftKey && state.sel && state.sel.key === key && state.sel.w0 !== null) {
          state.sel = { key, w0: Math.min(state.sel.w0, i), w1: Math.max(state.sel.w0, i) };
        } else {
          state.sel = { key, w0: i, w1: i };
        }
        setActive(key, { seek: true, scroll: false, force: true });
      } else {
        state.sel = { key, w0: null, w1: null };
        setActive(key, { seek: true, force: true });
        stopPreview();
      }
      paintSelection();
      renderPanel();
    });

    $('rev-clear-sel').addEventListener('click', () => {
      state.sel = null;
      paintSelection();
      renderPanel();
    });

    $('rev-people').addEventListener('click', (e) => {
      if (e.target.closest('[data-more]')) {
        state.showAllPeople = !state.showAllPeople;
        renderPanel();
        return;
      }
      const edit = e.target.closest('[data-edit]');
      if (edit) {
        stopPreview();
        openPersonEditor(edit.dataset.edit);
        return;
      }
      const accept = e.target.closest('[data-accept]');
      const line = selectionLine();
      if (accept && line && line.char) {
        applyLabel('char', state.cast.find((p) => p.char_id === line.char) ||
          { char_id: line.char, name: line.name, gender: line.gender });
        return;
      }
      const button = e.target.closest('[data-char]');
      if (!button) return;
      const person = state.cast.find((p) => p.char_id === button.dataset.char);
      if (e.detail >= 2) {           // dupla kattintás: átváltás erre a hangra
        clearTimeout(state.previewTimer);
        stopPreview();
        applyLabel('char', person || { char_id: button.dataset.char });
        return;
      }
      if (e.altKey) {                // Alt+kattintás: név átírása
        stopPreview();
        openPersonEditor(button.dataset.char);
        return;
      }
      // egy kattintás: a kijelölt feliratrész meghallgatása ezen a hangon
      stopPreview();
      clearTimeout(state.previewTimer);
      state.previewTimer = setTimeout(() => previewPerson(button), 260);
    });

    $('rev-offscreen').addEventListener('click', () => applyLabel('offscreen'));
    $('rev-unknown').addEventListener('click', () => applyLabel('unknown'));
    $('rev-clear').addEventListener('click', () => applyLabel('clear'));

    document.querySelectorAll('[data-off]').forEach((button) => {
      button.addEventListener('click', () => {
        const line = selectionLine();
        const base = line ? Number(line.offset || 0) : 0;
        $('rev-offset').value = (base + Number(button.dataset.off)).toFixed(2);
        applyOffset($('rev-offset').value);
      });
    });
    $('rev-offset-save').addEventListener('click', () => applyOffset($('rev-offset').value));
    $('rev-offset-reset').addEventListener('click', () => {
      $('rev-offset').value = '0';
      applyOffset(0);
    });
    $('rev-new-f').addEventListener('click', () => createPerson('F'));
    $('rev-new-m').addEventListener('click', () => createPerson('M'));
    $('rev-new-name').addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); createPerson('F'); }
    });

    $('rev-filter').addEventListener('change', (e) => {
      state.filterPending = e.target.checked;
      $('rev-lines').querySelectorAll('li.review-line').forEach((li) => {
        const line = state.byKey.get(li.dataset.key);
        li.classList.toggle('is-filtered', Boolean(state.filterPending && line && !isPending(line)));
      });
    });

    const nextPendingBtn = $('rev-next-pending');
    if (nextPendingBtn) nextPendingBtn.addEventListener('click', nextPending);
    document.addEventListener('keydown', onKey);
  }

  /** A következő, még el nem döntött sorra ugrás (körben, a jelenlegitől). */
  function nextPending() {
    const list = state.lines;
    if (!list.length) return;
    const at = list.findIndex((l) => l.key === state.active);
    for (let k = 1; k <= list.length; k += 1) {
      const line = list[(at + k + list.length) % list.length];
      if (!isPending(line)) continue;
      state.sel = { key: line.key, w0: null, w1: null };
      setActive(line.key, { seek: true, force: true });
      renderPanel();
      setStatus(`#${line.n} — még nincs szereplő megjelölve.`, 'warn');
      return;
    }
    setStatus('Minden sor szereplője meg van jelölve — nincs több döntésre váró sor.', 'ok');
  }

  function stepLine(delta) {
    const at = state.lines.findIndex((l) => l.key === state.active);
    const next = state.lines[Math.max(0, Math.min(state.lines.length - 1, at + delta))];
    if (!next) return;
    state.sel = { key: next.key, w0: null, w1: null };
    setActive(next.key, { seek: true, force: true });
    renderPanel();
  }

  function typing(e) {
    const tag = (e.target && e.target.tagName) || '';
    return tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA' || e.target.isContentEditable;
  }

  function onKey(e) {
    if (typing(e)) return;
    const video = $('rev-video');
    if (e.key === ' ' || e.code === 'Space') {
      e.preventDefault();
      if (video.src) {
        if (video.paused) video.play().catch(() => {});
        else video.pause();
      }
      return;
    }
    if (e.key === 'ArrowLeft') { e.preventDefault(); video.currentTime = Math.max(0, video.currentTime - 2); return; }
    if (e.key === 'ArrowRight') { e.preventDefault(); video.currentTime = Math.min(video.duration || 1e9, video.currentTime + 2); return; }
    if (e.key === 'ArrowUp') { e.preventDefault(); stepLine(-1); return; }
    if (e.key === 'ArrowDown') { e.preventDefault(); stepLine(1); return; }
    if (e.key === 'Enter') {
      const line = selectionLine();
      if (line && line.char) {
        e.preventDefault();
        applyLabel('char', state.cast.find((p) => p.char_id === line.char) ||
          { char_id: line.char, name: line.name, gender: line.gender });
      }
      return;
    }
    if (e.key.toLowerCase() === 'o') { e.preventDefault(); applyLabel('offscreen'); return; }
    if (e.key.toLowerCase() === 'n') { e.preventDefault(); applyLabel('unknown'); return; }
    if (e.key.toLowerCase() === 'j') { e.preventDefault(); nextPending(); return; }
    if (e.key === '0' || e.key === 'Backspace') { e.preventDefault(); applyLabel('clear'); return; }
    if (/^[1-9]$/.test(e.key)) {
      const person = state.cast[Number(e.key) - 1];
      if (person) { e.preventDefault(); applyLabel('char', person); }
    }
  }

  async function start() {
    bind();
    try {
      await loadProjects();
      await loadEpisode();
    } catch (err) {
      setStatus(`Az indítás nem sikerült: ${err.message}`, 'error');
    }
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
  else start();
})();
