// Szinkron stúdió: beszélő-azonosítás és hangkiosztás emberi felülvizsgálata.
(function () {
  'use strict';
  const esc = (v) => Auris.esc(v);
  const $ = (id) => document.getElementById(id);
  const state = {
    projects: [],
    project: null,
    episode: null,
    audit: null,
    data: null,
    library: null,
    player: null,
    playerAudioId: null,
    playerSeq: 0,
    playerPrepTimer: null,
    playerSampleTimer: null,
    speakerFilter: 'all',
    config: null,
    runTimer: null,
    runState: 'idle',
    libTick: 0,                  // a könyvtár utolsó frissítése futás közben
    audioMode: 'clip',           // 'clip' | 'preview' — mit játszik a #sync-audio
  };

  // a workflow-lépések kattintása a saját szekciójára görget
  const PANEL_FOR = {
    load: 'panel-load', prepare: 'panel-prepare', review: 'panel-review',
    cluster: 'panel-run', voices: 'panel-voices', dub: 'panel-run',
    qa: 'panel-run', mux: 'panel-run',
  };
  // mely szereplőknél volt nyitva a kézi hang-választó (újrarajz után is maradjon)
  const manualOpen = new Set();

  function setStatus(text, kind = '') {
    const el = $('sync-status');
    el.textContent = text || '';
    el.className = 'sync-status' + (kind ? ` is-${kind}` : '');
  }

  function pillClass(status) {
    return String(status || 'unknown');
  }

  function fmtTime(t) {
    const s = Number(t || 0);
    const m = Math.floor(s / 60);
    const rest = (s - m * 60).toFixed(2).padStart(5, '0');
    return `${String(m).padStart(2, '0')}:${rest}`;
  }

  function scoreStars(score) {
    const n = Math.max(0, Math.min(3, Number(score) || 0));
    return '★'.repeat(n) + '☆'.repeat(3 - n);
  }

  function projectEntry() {
    return state.projects.find((p) => (p.id || p.name) === state.project) || null;
  }

  // ── betöltés ────────────────────────────────────────────────────────────
  function renderProjectOptions(selected) {
    const sel = $('sync-project');
    const nameCount = {};
    state.projects.forEach((p) => { nameCount[p.name] = (nameCount[p.name] || 0) + 1; });
    sel.innerHTML = state.projects.map((p) =>
      `<option value="${esc(p.id || p.name)}">${esc(p.name)}${
        nameCount[p.name] > 1 ? ` · ${esc(p.id || p.name)}` : ''}</option>`).join('');
    const wanted = state.projects.find((p) => (p.id || p.name) === selected)
      || state.projects[0];
    state.project = wanted ? (wanted.id || wanted.name) : null;
    if (state.project) sel.value = state.project;
  }

  async function loadProjects() {
    const payload = await Auris.api('/api/dub/projects');
    state.projects = payload.projects || [];
    // friss klón: üres projects/ — a film-varázsló (lent) hozza létre az elsőt
    renderProjectOptions(state.project);
    if (!state.projects.length) {
      setStatus('Még nincs projekt. Hozz létre egyet az „Új film” űrlappal — '
        + 'a beállítások a dubstudio alapértékeiből jönnek.', 'busy');
      fillEpisodes();
      return;
    }
    await loadLibrary();
    fillEpisodes();
    await loadPlayer();
  }

  async function refreshProjects(selected) {
    const payload = await Auris.api('/api/dub/projects');
    state.projects = payload.projects || [];
    renderProjectOptions(selected);
  }

  function fillEpisodes() {
    const entry = projectEntry();
    const sel = $('sync-episode');
    const libEpisodes = ((state.library || {}).episodes || []).map((e) => e.tag);
    const episodes = libEpisodes.length ? libEpisodes : ((entry && entry.episodes) || []);
    sel.innerHTML = episodes.map((e) => `<option value="${esc(e)}">${esc(e)}</option>`).join('')
      || '<option value="">— nincs előkészített epizód —</option>';
    const current = state.episode && episodes.includes(state.episode) ? state.episode : null;
    state.episode = current || (episodes.length ? episodes[episodes.length - 1] : null);
    if (state.episode) sel.value = state.episode;
    const subTag = $('subtitle-episode');
    if (subTag) subTag.value = state.episode || '';
  }

  // ── könyvtár (sorozat/film) ─────────────────────────────────────────────
  const LIB_STATUS = {
    dubbed: 'kész', prepared: 'előkészítve', ready: 'indítható', waiting: 'anyagra vár',
  };

  function baseName(path) {
    const text = String(path || '');
    const cut = Math.max(text.lastIndexOf('/'), text.lastIndexOf('\\'));
    return text.slice(cut + 1);
  }

  function renderLibrary(lib) {
    state.library = lib;
    const mode = $('library-mode');
    if (mode) {
      mode.textContent = lib.is_movie ? 'Film' : `Sorozat · S${String(lib.season).padStart(2, '0')}`;
    }
    const counts = lib.counts || {};
    const cEl = $('library-counts');
    if (cEl) {
      const pending = (lib.episodes || []).filter((e) =>
        e.media && !e.ready && !e.output).length;
      cEl.textContent = `${counts.episodes || 0} rész · ${counts.ready || 0} indítható · ` +
        `${counts.dubbed || 0} kész` + (pending ? ` · ${pending} intake-re vár` : '');
    }
    const list = $('library-list');
    if (!list) return;
    const episodes = lib.episodes || [];
    if (!episodes.length) {
      list.innerHTML = '<p class="sync-empty">Még nincs epizód. Add meg a mappát vagy az első fájlt lent.</p>';
    } else {
      list.innerHTML = episodes.map((e) => {
        const status = e.status || 'waiting';
        const intake = e.intake || {};
        const canIntake = e.media && !e.ready && !e.output;
        const intakeInfo = intake.error
          ? `<span class="sync-libintake is-error" title="${esc(intake.error)}">intake hiba — kattints az újrapróbálkozásért</span>`
          : intake.state === 'partial'
            ? '<span class="sync-libintake is-warn">intake részben kész</span>'
            : canIntake
              ? '<span class="sync-libintake">felirat/hang előkészíthető</span>'
              : '';
        const missing = (e.missing || []).length
          ? `<span class="sync-libmissing">hiányzik: ${e.missing.map(esc).join(', ')}</span>`
          : '<span class="sync-libok">minden anyag megvan</span>';
        return `<article class="sync-librow is-${esc(status)}" data-lib-episode="${esc(e.tag)}" role="listitem">
          <button class="sync-libmain" type="button" data-lib-select="${esc(e.tag)}">
            <span class="sync-libtag">${esc(e.tag)}</span>
            <span class="sync-libname">${esc(e.media_name || 'nincs média')}</span>
            <span class="sync-libstate">${esc(LIB_STATUS[status] || status)}</span>
          </button>
          <div class="sync-libmeta">
            ${missing}
            ${intakeInfo}
            ${e.output ? `<span class="sync-libout">kimenet: ${esc(baseName(e.output))}</span>` : ''}
          </div>
          <div class="sync-libactions">
            <button class="btn btn-xs btn-ghost" type="button" data-lib-select="${esc(e.tag)}">Megnyitás</button>
            ${canIntake
              ? `<button class="btn btn-xs btn-ghost" type="button" data-lib-intake="${esc(e.tag)}">Anyagok pótlása</button>`
              : ''}
            ${e.ready && !e.output
              ? `<button class="btn btn-xs btn-primary" type="button" data-lib-run="${esc(e.tag)}">Szinkron indítása</button>`
              : ''}
          </div>
        </article>`;
      }).join('');
    }
    const watch = lib.watch || {};
    const toggle = $('watch-toggle');
    const poll = $('watch-poll');
    if (toggle) toggle.checked = !!watch.enabled;
    if (poll && watch.poll_s) poll.value = String(watch.poll_s);
    const info = $('watch-info');
    if (info) {
      info.textContent = watch.enabled
        ? `Figyelés bekapcsolva (${watch.poll_s || 30} mp)`
        : 'A figyelés most kikapcsolva.';
    }
  }

  async function loadLibrary() {
    if (!state.project) return;
    try {
      const lib = await Auris.api(`/api/dub/library?project=${encodeURIComponent(state.project)}`);
      renderLibrary(lib);
    } catch (err) {
      setStatus(`A könyvtár nem tölthető be: ${err.message}`, 'error');
    }
  }

  async function selectEpisode(tag) {
    if (!tag) return;
    state.episode = tag;
    state.audit = null;
    const sel = $('sync-episode');
    if (sel) sel.value = tag;
    const subTag = $('subtitle-episode');
    if (subTag) subTag.value = tag;
    await loadState();
    await loadPlayer();
  }

  async function addToLibrary() {
    const path = ($('library-path').value || '').trim();
    if (!path) { setStatus('Add meg a mappa vagy a fájl útvonalát.', 'error'); return; }
    await withBusy($('library-add'), 'Hozzáadás…', async () => {
      try {
        const res = await Auris.api('/api/dub/series/add', {
          method: 'POST', body: { project: state.project, path },
        });
        const added = res.added || [];
        const skipped = res.skipped || [];
        $('library-add-result').innerHTML = (
          added.length ? `<span class="ok">Hozzáadva: ${added.map((a) => `${esc(a.tag)} (${esc(a.mode)})`).join(', ')}</span>` : ''
        ) + (skipped.length ? `<span class="warn">Kihagyva: ${skipped.map((s) => `${esc(s.tag || s.path)} — ${esc(s.reason)}`).join('; ')}</span>` : '');
        await loadLibrary();
        fillEpisodes();
        if (added.length) await selectEpisode(added[added.length - 1].tag);
      } catch (err) {
        $('library-add-result').innerHTML = `<span class="err">${esc(err.message)}</span>`;
        setStatus(`A hozzáadás nem sikerült: ${err.message}`, 'error');
      }
    });
  }

  async function scanLibrary() {
    const folder = ($('library-path').value || '').trim();
    await withBusy($('library-scan'), 'Keresés…', async () => {
      try {
        const res = await Auris.api('/api/dub/series/scan', {
          method: 'POST', body: { project: state.project, folder },
        });
        const fresh = res.new || [];
        $('library-add-result').innerHTML = fresh.length
          ? `<span class="ok">${fresh.length} új fájl: ${fresh.map((c) => esc(c.tag || c.name)).join(', ')}. Nyomd meg a Hozzáadás gombot.</span>`
          : '<span class="warn">Nincs új, felismerhető epizód.</span>';
      } catch (err) {
        $('library-add-result').innerHTML = `<span class="err">${esc(err.message)}</span>`;
      }
    });
  }

  async function processReady(tag) {
    const episodes = tag ? [tag] : [];
    setStatus('Szinkron indítása…', 'busy');
    try {
      await Auris.api('/api/dub/series/process', {
        method: 'POST', body: { project: state.project, episodes },
      });
      state.runState = 'running';
      await pollRun(true);
    } catch (err) {
      setStatus(`A szinkron nem indult el: ${err.message}`, 'error');
    }
  }

  async function runIntake(tag) {
    const body = { project: state.project, then_run: false };
    if (tag) body.episodes = [tag];
    setStatus(tag ? `${tag}: felirat és hang előkészítése…` : 'Hiányzó anyagok előkészítése…', 'busy');
    try {
      const res = await Auris.api('/api/dub/intake', { method: 'POST', body });
      state.runState = 'running';
      setStatus(`Intake elindult: ${(res.episodes || []).join(', ')}. A napló a 4. szakaszban látszik.`);
      await pollRun(true);
    } catch (err) {
      setStatus(`Az intake nem indult el: ${err.message}`, 'error');
    }
  }

  async function createMovie() {
    const name = ($('movie-name').value || '').trim();
    const path = ($('movie-path').value || '').trim();
    if (!name || !path) {
      $('movie-result').innerHTML = '<span class="err">A film címét és a fájl útvonalát is add meg.</span>';
      return;
    }
    await withBusy($('movie-create'), 'Létrehozás…', async () => {
      try {
        const res = await Auris.api('/api/dub/movies', {
          method: 'POST',
          body: { name, path, source_lang: $('movie-source-lang').value,
                  template: state.project },
        });
        const id = (res.project || {}).id;
        $('movie-result').innerHTML =
          `<span class="ok">Filmprojekt elkészült: ${esc(name)} (${esc(id)}), ` +
          `fájl: ${esc(res.mode || '')}.</span>` +
          (res.warning ? `<span class="warn">${esc(res.warning)}</span>` : '') +
          '<span class="sync-configinfo">Az intake és az első szinkron a 4. szakasz naplójában fut.</span>';
        await refreshProjects(id);
        state.episode = 'E01';
        await loadLibrary();
        fillEpisodes();
        await loadState();
        await loadPlayer();
        state.runState = 'running';
        await pollRun(true);
      } catch (err) {
        $('movie-result').innerHTML = `<span class="err">${esc(err.message)}</span>`;
        setStatus(`A filmprojekt nem készült el: ${err.message}`, 'error');
      }
    });
  }

  async function uploadSubtitle() {
    const fileInput = $('subtitle-file');
    const file = fileInput && fileInput.files && fileInput.files[0];
    const tag = (($('subtitle-episode') && $('subtitle-episode').value) || '').trim().toUpperCase();
    if (!file) {
      $('subtitle-result').innerHTML = '<span class="err">Válassz .srt vagy .vtt fájlt.</span>';
      return;
    }
    if (!/^E\d{2,3}$/.test(tag)) {
      $('subtitle-result').innerHTML = '<span class="err">Add meg az epizód-azonosítót (pl. E04).</span>';
      return;
    }
    const form = new FormData();
    form.append('project', state.project || '');
    form.append('episode', tag);
    form.append('lang', $('subtitle-lang').value || 'eng');
    form.append('replace', $('subtitle-replace').checked ? '1' : '0');
    form.append('then_run', $('subtitle-thenrun').checked ? '1' : '0');
    form.append('file', file, file.name);
    await withBusy($('subtitle-upload'), 'Feltöltés…', async () => {
      try {
        const res = await Auris.api('/api/dub/subtitle', { method: 'POST', body: form });
        $('subtitle-result').innerHTML =
          `<span class="ok">${esc(res.tag)} ${esc(res.lang)} felirat elmentve: ` +
          `${res.cues} sor${res.replaced ? ' (a korábbi mentve)' : ''}.</span>` +
          (res.warning ? `<span class="warn">${esc(res.warning)}</span>` : '') +
          (res.job ? '<span class="sync-configinfo">Az intake + szinkron a 4. szakasz naplójában fut.</span>' : '');
        await loadLibrary();
        fillEpisodes();
        await selectEpisode(tag);
        if (res.job) {
          state.runState = 'running';
          await pollRun(true);
        }
      } catch (err) {
        $('subtitle-result').innerHTML = `<span class="err">${esc(err.message)}</span>`;
        setStatus(`A felirat feltöltése nem sikerült: ${err.message}`, 'error');
      }
    });
  }

  async function startYoutube() {
    const url = ($('youtube-url').value || '').trim();
    const name = ($('youtube-name').value || '').trim();
    if (!url) {
      $('youtube-result').innerHTML = '<span class="err">Add meg a YouTube URL-t.</span>';
      return;
    }
    await withBusy($('youtube-start'), 'Letöltés…', async () => {
      try {
        const res = await Auris.api('/api/dub/youtube', {
          method: 'POST',
          body: { url, name, source_lang: $('youtube-source-lang').value,
                  template: state.project },
        });
        const id = (res.project || {}).id;
        $('youtube-result').innerHTML =
          `<span class="ok">YouTube-projekt elkészült: ${esc((res.project || {}).name || id)} ` +
          `(${esc(id)}).</span>` +
          (res.warning ? `<span class="warn">${esc(res.warning)}</span>` : '') +
          '<span class="sync-configinfo">A letöltés, az intake és a szinkron a 4. szakasz naplójában fut.</span>';
        await refreshProjects(id);
        state.episode = 'E01';
        await loadLibrary();
        fillEpisodes();
        await loadState();
        await loadPlayer();
        state.runState = 'running';
        await pollRun(true);
      } catch (err) {
        $('youtube-result').innerHTML = `<span class="err">${esc(err.message)}</span>`;
        setStatus(`A YouTube-letöltés nem indult el: ${err.message}`, 'error');
      }
    });
  }

  async function saveWatch() {
    try {
      const res = await Auris.api('/api/dub/series/watch', {
        method: 'POST',
        body: { project: state.project, enabled: $('watch-toggle').checked,
                poll_s: Number($('watch-poll').value || 30) },
      });
      const w = res.watch || {};
      const info = $('watch-info');
      if (info) info.textContent = w.enabled ? `Figyelés bekapcsolva (${w.poll_s || 30} mp)` : 'A figyelés most kikapcsolva.';
      setStatus(w.enabled ? 'A program figyeli az új részeket.' : 'A figyelés kikapcsolva.');
    } catch (err) {
      setStatus(`A figyelés mentése nem sikerült: ${err.message}`, 'error');
    }
  }

  // ── lejátszó ────────────────────────────────────────────────────────────
  function playerStatus(text) {
    const el = $('player-status');
    if (el) el.textContent = text;
  }

  function clearPlayer(text) {
    stopPlayerTimers();
    const video = $('sync-player');
    if (video) { video.pause(); video.removeAttribute('src'); video.load(); }
    ['player-audio', 'player-subtitle'].forEach((id) => {
      const sel = $(id);
      if (!sel) return;
      sel.innerHTML = '<option value="">— nincs betöltött média —</option>';
      sel.disabled = true;
    });
    state.playerAudioId = null;
    showPlayerPrep(false);
    playerStatus(text || 'nincs betöltött epizód');
  }

  function stopPlayerTimers() {
    if (state.playerPrepTimer) clearTimeout(state.playerPrepTimer);
    if (state.playerSampleTimer) clearTimeout(state.playerSampleTimer);
    state.playerPrepTimer = null;
    state.playerSampleTimer = null;
  }

  function showPlayerPrep(show, text) {
    const box = $('player-prep');
    if (!box) return;
    box.hidden = !show;
    const line = $('player-prep-text');
    if (line && text) line.textContent = text;
    if (!show) {
      const bar = box.querySelector('.sync-prepbar');
      if (bar) bar.classList.remove('is-error');
    }
  }

  function prepText(payload) {
    if (payload && payload.state === 'error') {
      return `A hangsáv előkészítése nem sikerült: ${payload.error || 'ismeretlen hiba'}`;
    }
    const mb = Number(payload && payload.part_bytes || 0) / (1024 * 1024);
    const secs = Number(payload && payload.elapsed || 0);
    return `A hangsáv előkészítése… ${secs.toFixed(0)} mp` +
      (mb > 0 ? ` · ${mb.toFixed(0)} MB` : '') +
      ' (a kész proxy a következő váltásnál már azonnal indul)';
  }

  function setPlayerSource(track, opts = {}) {
    const video = $('sync-player');
    if (!track || !video) return;
    showPlayerPrep(false);
    const at = opts.keepTime === false ? 0 : (video.currentTime || 0);
    video.src = track.url;
    video.load();
    video.addEventListener('loadedmetadata', () => {
      if (opts.sample) {
        const sample = opts.sample || {};
        const start = Number(sample.start || 0);
        const dur = Math.max(1, Number(sample.duration || 8));
        if (start > 0 && start < video.duration) video.currentTime = start;
        state.playerSampleTimer = setTimeout(() => {
          video.pause();
          setStatus('A magyar minta véget ért — a lejátszó a teljes filmre áll.');
        }, dur * 1000);
        video.play().catch(() => setStatus('A böngésző nem engedte a mintalejátszást.', 'error'));
      } else if (at > 0 && at < video.duration) {
        video.currentTime = at;
      }
    }, { once: true });
    playerStatus(`${state.episode} · ${track.label}`);
  }

  async function pollPlayerPrep(audioId, seq, attempt = 0) {
    if (seq !== state.playerSeq) return;
    const params = new URLSearchParams({ project: state.project, episode: state.episode, audio: audioId });
    try {
      const st = await Auris.api(`/api/dub/player/status?${params.toString()}`);
      if (seq !== state.playerSeq) return;
      if (st.state === 'ready') {
        const track = ((state.player || {}).audio || []).find((a) => a.id === audioId);
        if (track) { track.cached = true; setPlayerSource(track, state.playerRequestOpts || {}); }
        state.playerRequestOpts = null;
        return;
      }
      if (st.state === 'error') {
        showPlayerPrep(true, prepText(st));
        const bar = ($('player-prep') || {}).querySelector && $('player-prep').querySelector('.sync-prepbar');
        if (bar) bar.classList.add('is-error');
        playerStatus('a hangsáv előkészítése nem sikerült');
        return;
      }
      showPlayerPrep(true, prepText(st));
      playerStatus(`előkészítés: ${audioId}`);
      state.playerPrepTimer = setTimeout(() => pollPlayerPrep(audioId, seq, attempt + 1), 900);
    } catch (err) {
      if (seq !== state.playerSeq) return;
      showPlayerPrep(true, `Az állapot nem kérdezhető le: ${err.message}`);
    }
  }

  async function activatePlayerAudio(audioId, opts = {}) {
    const info = state.player || {};
    const track = (info.audio || []).find((a) => a.id === audioId);
    if (!track) return;
    stopPlayerTimers();
    const seq = ++state.playerSeq;
    state.playerAudioId = audioId;
    state.playerRequestOpts = opts;
    const sel = $('player-audio');
    if (sel) sel.value = audioId;
    if (track.cached) {
      setPlayerSource(track, opts);
      return;
    }
    showPlayerPrep(true, `A hangsáv előkészítése… (${track.label})`);
    playerStatus(`${state.episode} · előkészítés: ${track.label}`);
    try {
      const st = await Auris.api('/api/dub/player/prepare', {
        method: 'POST',
        body: { project: state.project, episode: state.episode, audio: audioId },
      });
      if (seq !== state.playerSeq) return;
      if (st.state === 'ready') { track.cached = true; setPlayerSource(track, opts); return; }
      if (st.state === 'error') {
        showPlayerPrep(true, prepText(st));
        playerStatus(`a hangsáv nem készíthető el: ${st.error || ''}`);
        return;
      }
      showPlayerPrep(true, prepText(st));
      pollPlayerPrep(audioId, seq);
    } catch (err) {
      if (seq !== state.playerSeq) return;
      showPlayerPrep(true, `A hangsáv előkészítése nem indult: ${err.message}`);
      playerStatus('az előkészítés nem indult el');
    }
  }

  async function loadPlayer() {
    if (!state.project || !state.episode) { clearPlayer(); return; }
    playerStatus('sávok betöltése…');
    const params = new URLSearchParams({ project: state.project, episode: state.episode });
    try {
      const info = await Auris.api(`/api/dub/player?${params.toString()}`);
      state.player = info;
      const audio = $('player-audio');
      const subs = $('player-subtitle');
      const chosen = videoCurrentAudio();
      audio.innerHTML = (info.audio || []).map((a) =>
        `<option value="${esc(a.id)}">${esc(a.label)}</option>`).join('');
      audio.disabled = !(info.audio || []).length;
      const audioId = (info.audio || []).some((a) => a.id === chosen) ? chosen
        : ((info.audio.find((a) => a.default) || info.audio[0] || {}).id || '');
      if (audioId) audio.value = audioId;
      subs.innerHTML = '<option value="">kikapcsolva</option>' + (info.subtitles || []).map((s) =>
        `<option value="${esc(s.id)}">${esc(s.label)}</option>`).join('');
      subs.disabled = !(info.subtitles || []).length;
      if (audioId) {
        state.player = info;
        const label = (info.audio.find((a) => a.id === audioId) || {}).label || '';
        playerStatus(`${state.episode} · ${Math.round(Number(info.duration || 0) / 60)} perc · ${label}`);
        await activatePlayerAudio(audioId, { keepTime: false });
      } else {
        clearPlayer('a forrásnak nincs hangsávja');
      }
    } catch (err) {
      clearPlayer(err.message);
      setStatus(`A lejátszó nem tölthető be: ${err.message}`, 'error');
    }
  }

  function videoCurrentAudio() {
    return state.playerAudioId || '';
  }

  function applyPlayerAudio(audioId) {
    return activatePlayerAudio(audioId, { keepTime: true });
  }

  async function playDubSample() {
    const info = state.player || {};
    const dub = (info.audio || []).find((a) => a.kind === 'dub');
    if (!dub) { setStatus('Ehhez még nincs kész magyar hangsáv (dub.wav).', 'error'); return; }
    const video = $('sync-player');
    if (!video) return;
    await withBusy($('player-dub-sample'), 'Minta…', async () => {
      setStatus('Rövid magyar minta készítése…', 'busy');
      try {
        const res = await Auris.api('/api/dub/player/sample', {
          method: 'POST', body: { project: state.project, episode: state.episode },
        });
        stopPlayerTimers();
        state.playerSeq += 1;
        state.playerAudioId = 'dub';
        const sel = $('player-audio');
        if (sel) sel.value = 'dub';
        showPlayerPrep(false);
        const dur = Math.max(1, Number(res.duration || 8));
        video.src = res.url;
        video.load();
        video.addEventListener('loadedmetadata', () => {
          video.play().catch(() => setStatus('A böngésző nem engedte a mintalejátszást.', 'error'));
          state.playerSampleTimer = setTimeout(() => {
            video.pause();
            setStatus('A magyar minta véget ért — válassz teljes hangsávot a folytatáshoz.');
          }, dur * 1000);
        }, { once: true });
        playerStatus(`${state.episode} · magyar minta (${Math.round(dur)} mp)`);
        setStatus('Magyar minta lejátszása…');
      } catch (err) {
        setStatus(`A magyar minta nem készíthető el: ${err.message}`, 'error');
      }
    });
  }

  function applyPlayerSubtitle(subId) {
    const video = $('sync-player');
    video.querySelectorAll('track').forEach((t) => t.remove());
    if (!subId) return;
    const info = state.player || {};
    const track = (info.subtitles || []).find((s) => s.id === subId);
    if (!track) return;
    const el = document.createElement('track');
    el.kind = 'subtitles';
    el.label = track.label;
    el.srclang = track.lang || 'hu';
    el.src = track.url;
    el.default = true;
    video.appendChild(el);
    if (video.textTracks && video.textTracks.length) {
      video.textTracks[video.textTracks.length - 1].mode = 'showing';
    }
  }

  async function loadState() {
    if (!state.project || !state.episode) { renderEmpty(); return; }
    const params = new URLSearchParams({ project: state.project, episode: state.episode });
    if (state.audit) params.set('audit', state.audit);
    let data;
    try {
      data = await Auris.api(`/api/dub/state?${params.toString()}`);
    } catch (err) {
      setStatus(`Nem sikerült betölteni: ${err.message}`, 'error');
      return;
    }
    state.data = data;
    state.audit = data.audit ? data.audit.file : null;
    renderAuditSelect(data.audits || []);
    renderStages(data);
    renderBadges(data);
    renderStats(data);
    renderScene(data);
    renderSpeakers(data);
    renderCasting(data);
    renderSound(data.sound || {});
    renderModel(data);
    const when = data.audit ? `Próba: ${data.audit.file} (${data.audit.duration} mp)` : 'Még nincs próba ehhez az epizódhoz.';
    setStatus(when);
  }

  function renderEmpty() {
    const sp = $('sync-speakers');
    if (sp) sp.innerHTML = '<p class="sync-empty">Nincs kiválasztott epizód.</p>';
    const st = $('sync-stats');
    if (st) st.innerHTML = '';
    const open = $('voices-open');
    if (open) open.innerHTML = '<p class="sync-empty">Nincs kiválasztott epizód.</p>';
  }

  function frameUrl(t, w = 360) {
    const params = new URLSearchParams({
      project: state.project, episode: state.episode,
      t: Number(t).toFixed(2), w: String(w),
    });
    return `/api/dub/frame?${params.toString()}`;
  }

  function renderScene(data) {
    const el = $('sync-scene');
    if (!el) return;
    const a = data.audit || {};
    if (a.start == null || !a.duration) { el.innerHTML = ''; return; }
    const start = Number(a.start), dur = Number(a.duration), n = 6;
    const cells = Array.from({ length: n }, (_, i) => {
      const rel = ((i + 0.5) * dur) / n;
      return `<figure class="sync-frame">
        <img loading="lazy" alt="" src="${frameUrl(start + rel, 240)}">
        <button class="mini-play" type="button" title="Innen hallgatás" data-play-start="${rel.toFixed(2)}" data-play-duration="5">▶</button>
        <figcaption>${fmtTime(rel)}</figcaption>
      </figure>`;
    }).join('');
    el.innerHTML = `<div class="sync-scene-head">Jelenet — a kocka megmutatja, hol tartunk; a ▶ onnan játszik</div>
      <div class="sync-frames">${cells}</div>`;
  }

  function renderStages(data) {
    const el = $('sync-steps');
    if (!el) return;
    const stages = data.stages || [];
    if (!stages.length) { el.innerHTML = ''; return; }
    el.innerHTML = stages.map((s, i) => {
      const mark = s.state === 'done' ? '✓' : String(i + 1);
      return `<button class="sync-step is-${esc(s.state)}" data-goto="${esc(PANEL_FOR[s.id] || '')}"
        type="button" title="Ugrás ide: ${esc(s.label)}">
        <span class="n" aria-hidden="true">${mark}</span>
        <span class="txt"><span class="l">${esc(s.label)}</span>
          <span class="i">${esc(s.info || (s.state === 'done' ? 'kész' : 'függő'))}</span>
        </span>
      </button>`;
    }).join('');
  }

  function renderAuditSelect(audits) {
    const sel = $('sync-audit');
    sel.innerHTML = audits.map((a) =>
      `<option value="${esc(a.file)}">${esc(a.file)} — ${esc(a.start)}–${esc(Number(a.start) + Number(a.duration))} mp</option>`
    ).join('') || '<option value="">—</option>';
    if (state.audit) sel.value = state.audit;
  }

  function renderBadges(data) {
    const bits = [];
    const t = data.timbres || {};
    bits.push(`<span class="badge-soft ${t.ready ? 'ok' : 'warn'}">Hangszín: ${t.ready ? `${t.voices} hang` : 'nincs mérés'}</span>`);
    if (data.project && !data.project.media) bits.push('<span class="badge-soft err">Forrásmédia hiányzik</span>');
    if (data.audit && data.audit.pending_review) {
      bits.push(`<span class="badge-soft warn">${data.audit.pending_review} felülvizsgálandó</span>`);
    }
    const review = data.review || {};
    if (review.total) bits.push(`<span class="badge-soft ok">${review.total} emberi döntés</span>`);
    $('sync-badges').innerHTML = bits.join('');
  }

  function renderStats(data) {
    const speakers = data.speakers || [];
    const linked = speakers.filter((s) => s.char_id && !s.review_required).length;
    const pending = speakers.filter((s) => s.review_required).length;
    const chars = data.characters || [];
    const voiced = chars.filter((c) => c.voice).length;
    const cells = [
      [speakers.length, 'beszélő a jelenetben'],
      [linked, 'már azonosítva'],
      [pending, 'megnézendő'],
      chars.length ? [`${voiced}/${chars.length}`, 'szereplőnek van hangja']
                   : ['—', 'szereplő'],
    ];
    $('sync-stats').innerHTML = cells.map(([v, l]) =>
      `<div class="sync-stat"><div class="num-value">${esc(v)}</div><div class="num-label">${esc(l)}</div></div>`
    ).join('');
  }

  // a beszélő-státuszok laikus magyar megfelelője (a CSS-osztály a nyers kód)
  const STATUS_LABEL = {
    linked: 'rögzítve', confirmed: 'megerősítve', corrected: 'javítva',
    accepted: 'elfogadva', weak: 'bizonytalan', ambiguous: 'bizonytalan',
    new: 'új beszélő', no_evidence: 'nincs bizonyíték',
    rejected_by_human: 'nincs karaktere', unknown: 'ismeretlen',
  };
  function statusLabel(status) {
    return STATUS_LABEL[String(status || '')] || status || 'ismeretlen';
  }

  function candidateBars(cands) {
    if (!cands || !cands.length) return '<p class="sync-reason">Nincs összehasonlítható felvétel róla.</p>';
    return `<div class="sync-cands">${cands.map((c, i) => {
      const pct = Math.max(0, Math.min(100, (Number(c.cosine) + 1) * 50));
      return `<div class="sync-cand${i === 0 ? ' is-best' : ''}">
        <span>${esc(c.char_id)}${c.gender ? ` · ${esc(c.gender)}` : ''}</span>
        <span class="bar"><span style="width:${pct.toFixed(1)}%"></span></span>
        <span class="cos">${Number(c.cosine).toFixed(2)}</span>
      </div>`;
    }).join('')}</div>`;
  }

  function voiceBlock(sp) {
    const p = sp.proposal;
    if (!p) {
      return `<div class="sync-voice"><div class="sync-voice-head"><span class="sync-voice-name muted">Még nincs hangjavaslat</span></div>
        <p class="sync-reason">Előbb igazold le, ki ez a beszélő — csak utána tud hangot javasolni.</p></div>`;
    }
    const options = Object.keys(p.probabilities || {});
    const reasons = (p.candidates && p.candidates[0] && p.candidates[0].reasons) || [];
    const flags = [];
    if (p.noul && p.noul.human_review >= 0.4) flags.push(`<span class="sync-flag">nézd meg tüzetesebben</span>`);
    if (p.noul && p.noul.pitch_shift >= 0.3) flags.push(`<span class="sync-flag">a hangszíne eltér — állítsd be</span>`);
    if (!flags.length) flags.push('<span class="sync-flag calm">nincs probléma</span>');
    return `<div class="sync-voice">
      <div class="sync-voice-head">
        <span class="sync-voice-name">Javaslat: ${esc(p.label)} <small>${esc(p.choice)}</small></span>
        <span class="sync-score" title="illetés: 0 rossz … 3 ideális">${scoreStars(p.score)}</span>
      </div>
      <div class="sync-prob" title="mennyire biztos"><span style="width:${(Number(p.answer_confidence || 0) * 100).toFixed(1)}%"></span></div>
      <p class="sync-meta">${(Number(p.answer_confidence || 0) * 100).toFixed(0)}% biztos</p>
      ${p.hint ? `<p class="sync-reason">${esc(p.hint)}</p>` : ''}
      <details class="sync-advanced sync-why"><summary>Miért?</summary>
        <ul class="sync-reasons">${reasons.slice(0, 4).map((r) => `<li>${esc(r)}</li>`).join('')}</ul>
      </details>
      <div class="sync-flags">${flags.join('')}</div>
      <div class="sync-actions">
        <button class="btn btn-sm btn-primary" data-voice="${esc(p.choice)}">Hang elfogadása</button>
        <button class="btn btn-sm btn-ghost" data-preview-voice="${esc(p.choice)}" type="button"
          title="Magyar mintamondat meghallgatása ezzel a hanggal">▲ Mintahallgatás</button>
        <select class="select-input" data-voice-select>
          ${options.map((v) => `<option value="${esc(v)}"${v === p.choice ? ' selected' : ''}>${esc(v)} (${(p.probabilities[v] * 100).toFixed(0)}%)</option>`).join('')}
        </select>
        <button class="btn btn-sm btn-ghost" data-voice-apply="1">Kiválasztott hang</button>
      </div>
    </div>`;
  }

  function speakerCard(sp, characters) {
    const done = !sp.review_required;
    const cls = sp.status === 'rejected_by_human' ? 'is-rejected' : done ? 'is-done' : 'is-pending';
    const charLabel = sp.name ? `${sp.name}${sp.char_id ? ` · ${sp.char_id}` : ''}`
      : (sp.char_id || 'ismeretlen beszélő');
    // frame_t már abszolút filmidő (a szerver adja a jelenet elejével együtt)
    const shot = sp.frame_t != null
      ? `<img class="sync-shot" loading="lazy" alt="képkocka a beszélő egyik fordulójából"
           src="${frameUrl(Number(sp.frame_t), 400)}">`
      : '';
    const lines = (sp.lines || []).slice(0, 4).map((l) =>
      `<li><span class="t">${fmtTime(l.start)}</span>${esc(l.text)}</li>`).join('');
    const dialogue = lines
      ? `<ul class="sync-lines">${lines}</ul>`
      : '<p class="sync-reason">Nincs feliratszöveg ehhez a részhez.</p>';
    const suggested = !sp.char_id && sp.suggested_new_id
      ? `<span class="sync-reason">új szereplő lehet: ${esc(sp.suggested_new_id)}</span>` : '';
    const acceptLabel = sp.char_id ? 'Elfogadom' : `Új karakter (${esc(sp.suggested_new_id || '—')})`;
    const charOptions = ['<option value="">— válassz —</option>'].concat(
      characters.map((c) => `<option value="${esc(c.char_id)}"${c.char_id === sp.char_id ? ' selected' : ''}>${esc(c.char_id)}${c.gender ? ` · ${esc(c.gender)}` : ''}${c.voice ? ` · ${esc(c.voice)}` : ''}</option>`)
    ).join('');
    const spans = (sp.spans || []).slice(0, 6);
    const longest = spans.slice().sort((a, b) => (b[1] - b[0]) - (a[1] - a[0]))[0];
    // a jelenetminta legfeljebb 10 s — pont az, amiből emberi döntés születik
    const play = longest
      ? `<button class="btn btn-sm btn-ghost" data-play-start="${longest[0]}"
           data-play-duration="${Math.min(10, (longest[1] - longest[0])).toFixed(2)}">▶ Meghallgatás</button>`
      : '<span class="sync-reason">nincs hangrészlet</span>';
    return `<article class="sync-card ${cls}" data-speaker="${esc(sp.speaker)}">
      <header class="sync-card-head">
        <div>
          <div class="sync-char">${esc(charLabel)}</div>
          <div class="sync-speaker">${esc(sp.speaker)}</div>
          ${suggested}
        </div>
        <span class="sync-pill ${pillClass(sp.status)}">${esc(statusLabel(sp.status))}</span>
      </header>
      <p class="sync-meta">
        <span>${Number(sp.evidence_seconds || 0).toFixed(1)} mp beszéd</span>
        <span>${esc(sp.evidence_turns || 0)} mondat</span>
        ${sp.gender ? `<span>nem: ${esc(sp.gender === 'F' ? 'nő' : sp.gender === 'M' ? 'férfi' : sp.gender)}</span>` : ''}
        ${sp.voice ? `<span>hangja: ${esc(sp.voice)}</span>` : ''}
      </p>
      ${sp.reason ? `<p class="sync-reason">${esc(sp.reason)}</p>` : ''}
      ${sp.manual_note ? `<p class="sync-reason">Megjegyzés: ${esc(sp.manual_note)}</p>` : ''}
      ${shot}
      ${dialogue}
      ${candidateBars(sp.candidates)}
      <div class="sync-actions">
        ${play}
        <button class="btn btn-sm btn-primary" data-accept="${esc(sp.char_id || sp.suggested_new_id || '')}"
          ${sp.char_id || sp.suggested_new_id ? '' : 'disabled'}>${acceptLabel}</button>
        <button class="btn btn-sm btn-ghost" data-reject="1">Nincs karaktere</button>
      </div>
      <div class="sync-actions">
        <select class="select-input" data-char-select>${charOptions}</select>
        <button class="btn btn-sm btn-ghost" data-correct="1">Javítás</button>
      </div>
      <div class="sync-actions">
        <input class="text-input" data-name-input maxlength="60" placeholder="szereplő neve (pl. Doron)"
          value="${esc(sp.name || '')}" aria-label="szereplő neve">
      </div>
      ${voiceBlock(sp)}
    </article>`;
  }

  function renderSpeakers(data) {
    const speakers = data.speakers || [];
    const characters = data.characters || [];
    const filtered = speakers.filter((sp) => {
      if (state.speakerFilter === 'pending') return sp.review_required;
      if (state.speakerFilter === 'done') return !sp.review_required;
      return true;
    });
    if (!filtered.length) {
      $('sync-speakers').innerHTML = `<p class="sync-empty">${speakers.length
        ? 'Ebben a szűrőben nincs beszélő.'
        : 'Ehhez az epizódhoz még nincs beszélő-próba. Indíts egyet vagy válassz korábbit.'}</p>`;
      return;
    }
    $('sync-speakers').innerHTML = filtered.map((sp) => speakerCard(sp, characters)).join('');
  }

  // ── Hangkiosztás: döntés-központú nézet ───────────────────────────────────
  // A cél: az automatizmus dönt ról, az ember csak a nyitott kérdésnél
  // mond Igent/Nemet, és minden hangot mielőtt hallgasson meg.

  function voiceLabel(vid) {
    const v = ((state.data || {}).voice_pool || []).find((x) => x.vid === vid);
    return v ? `${vid} (${esc(v.gender)})` : vid;
  }

  function shareBar(share) {
    const pct = Math.max(0, Math.min(100, Number(share || 0) * 100)).toFixed(1);
    return `<span class="sync-sharebar"><span style="width:${pct}%"></span></span><span class="share-num">${pct}%</span>`;
  }

  function rolePill(role) {
    const cls = role === 'fő' ? 'is-main' : 'is-side';
    return `<span class="sync-rolepill ${cls}">${esc(role || 'mellék')}</span>`;
  }

  function castOptionsRow(row, opts) {
    if (!opts || !opts.length) {
      return '<p class="sync-reason">Nincs hangkínálat (hangszín-mérés futtandó).</p>';
    }
    return `<div class="sync-cands">${opts.map(([vid, pct], i) =>
      `<div class="sync-cand${i === 0 ? ' is-best' : ''}">
        <span class="vid">${esc(vid)}</span>
        <span class="bar"><span style="width:${Math.max(2, Math.min(100, pct)).toFixed(0)}%"></span></span>
        <span class="cos">${pct}%</span>
      </div>`).join('')}</div>`;
  }

  function manualPickerRow(sp, reason) {
    const pool = (state.data || {}).voice_pool || [];
    const options = pool.map((v) =>
      `<option value="${esc(v.vid)}"${v.vid === sp.voice ? ' selected' : ''}>${esc(v.vid)} · ${esc(v.gender)}${v.model_present ? '' : ' · nincs modell'}</option>`).join('');
    return `<div class="sync-manual">
      <p class="sync-reason">${esc(reason)}</p>
      <div class="sync-actions">
        <select class="select-input" data-cast-select>${options}</select>
        <button class="btn btn-sm btn-primary" data-cast-apply="1">Hang beállítása</button>
        <button class="btn btn-sm btn-ghost" data-cast-close="1">Mégse</button>
      </div>
    </div>`;
  }

  // char_id → a beszélő legjobb képkockája (az azonosított beszélőből)
  function frameMapFromSpeakers(speakers) {
    const map = {};
    // 1) a szerver által adott karakter-képkockák (galéria beszélő-indexből)
    for (const [cid, t] of Object.entries((state.data || {}).char_frames || {})) {
      if (t != null) map[cid] = Number(t);
    }
    // 2) az azonosított beszélők saját képkockája (felülírja)
    for (const sp of speakers || []) {
      if (sp.char_id && sp.frame_t != null) map[sp.char_id] = Number(sp.frame_t);
    }
    return map;
  }

  // szereplőnkénti hangszín: csak akkor kell, ha két szereplő egyformán szól
  function pitchControl(c) {
    const pitch = c.pitch_override != null ? Number(c.pitch_override) : Number(c.pitch || 0);
    const label = `${pitch > 0 ? '+' : ''}${pitch.toFixed(1)}`;
    return `<span class="sync-pitch" title="Hangszín: ha két szereplő egyformán szól, itt eltolhatod (a következő futásnál érvényes)">
      <span class="sync-pitchlabel">hangszín</span>
      <input type="range" min="-4" max="4" step="0.5" value="${pitch}"
             data-pitch="${esc(c.char_id)}" aria-label="Hangszín">
      <span class="sync-pitchval" data-pitchval="${esc(c.char_id)}">${label}</span>
      <button class="btn btn-xs btn-ghost" type="button" data-pitch-reset="${esc(c.char_id)}"
        ${c.pitch_override == null ? 'hidden' : ''} title="Vissza az automatikusra">auto</button>
    </span>`;
  }

  function castThumb(t) {
    if (t == null) return '';
    return `<img class="sync-castthumb" loading="lazy" alt="" src="${frameUrl(Number(t), 200)}">`;
  }

  function castOpenRow(row) {
    const isReady = row.ready;
    const key = row.speaker;
    const noul = row.noul || {};
    const reasons = [];
    if (noul.human_review >= 0.4) reasons.push('nem egyértelmű — nézd meg tüzetesebben');
    if (noul.pitch_shift >= 0.3) reasons.push('a hangszíne eltér a többiektől');
    if (row.hint) reasons.push(row.hint);
    if (isReady) reasons.push('biztos javaslat');
    const decision = isReady
      ? '<span class="sync-flag calm">elfogadható — a fenti gombbal egyben is</span>'
      : '<span class="sync-flag">döntés rád vár</span>';
    return `<div class="sync-castopen${isReady ? ' is-ready' : ''}" data-cast="${esc(key)}">
      <div class="sync-castopen-head">
        ${castThumb(row.frame_t)}
        <span class="sync-castname">${esc(row.name || row.char_id || row.speaker)}${row.char_id ? ` <small>${esc(row.char_id)}</small>` : ''}</span>
        ${rolePill(row.role)}
        <span class="sync-castweight" title="beszédidő-arány a szereplők között">${shareBar(row.share)} · ${row.evidence_seconds}s bizonyíték</span>
      </div>
      <div class="sync-castproposal">
        Javasolt hang: <strong>${esc(row.choice || '—')}</strong>
        <button class="btn btn-xs btn-ghost" data-preview-voice="${esc(row.choice || '')}" type="button">▲ meghallgatás</button>
        ${row.voice ? `<span class="sync-reason"> most: ${esc(row.voice)}</span>` : ''}
        <span class="sync-score">${scoreStars(row.score)}</span>
        ${(Number(row.answer_confidence || 0) * 100).toFixed(0)}% biztos
      </div>
      ${castOptionsRow(row, row.options)}
      <ul class="sync-reasons">${reasons.slice(0, 4).map((r) => `<li>${esc(r)}</li>`).join('')}</ul>
      ${decision}
      <div class="sync-actions">
        <button class="btn btn-sm btn-primary" data-cast-yes="1" type="button">Igen — jó a javaslat</button>
        <button class="btn btn-sm btn-ghost" data-cast-no="1" type="button">Nem — kézi választás</button>
      </div>
      ${manualOpen.has(key) ? manualPickerRow(row, 'Kézi hangkiosztás — melyik magyar hangot kapja?') : ''}
    </div>`;
  }

  function castAssignedRow(row) {
    const key = row.speaker;
    return `<div class="sync-castassigned" data-cast="${esc(key)}">
      ${castThumb(row.frame_t)}
      <span class="sync-castname" title="${esc(row.speaker)}">${esc(row.name || row.char_id || row.speaker)}</span>
      ${rolePill(row.role)}
      <button class="btn btn-xs btn-ghost" data-preview-voice="${esc(row.voice || '')}" type="button" title="${esc(row.voice || '')} mintamondat">🔊 ${esc(row.voice || '—')}</button>
      ${shareBar(row.share)}
      <div class="sync-actions">
        <button class="btn btn-xs btn-ghost" data-cast-no="1" data-cast="${esc(key)}" type="button">átrendelés</button>
      </div>
      ${manualOpen.has(key) ? manualPickerRow(row, 'Kézi hangkiosztás — melyik magyar hangot kapja?') : ''}
    </div>`;
  }

  function castCharRow(c, frameMap) {
    const role = c.role_override || c.role || 'mellék';
    const override = c.role_override ? ' (kézi)' : '';
    const title = c.name ? `${c.name} (${c.char_id})` : c.char_id;
    return `<div class="sync-castchar">
      ${castThumb(frameMap ? frameMap[c.char_id] : null)}
      <span class="sync-castname" title="${esc(title)}">${esc(c.name || c.char_id)}${c.name ? ` <small>${esc(c.char_id)}</small>` : ''}${c.gender ? ` <small>${esc(c.gender)}</small>` : ''}</span>
      ${pitchControl(c)}
      ${rolePill(role)}${override ? '<span class="sync-reason">kézi</span>' : ''}
      ${shareBar(c.share || (c.duration ? c.duration / (Number((state.data || {}).total_speech || 0) || 1) : 0))}
      <span class="sync-castvoice">${c.voice
        ? `<button class="btn btn-xs btn-ghost" data-preview-voice="${esc(c.voice)}" type="button">🔊 ${esc(c.voice)}</button>`
        : '<span class="sync-reason">nincs hangja</span>'}</span>
      <div class="sync-actions">
        <button class="btn btn-xs btn-ghost" data-cast-role="1" data-char-id="${esc(c.char_id)}"
          data-role="${role === 'fő' ? 'mellék' : 'fő'}"
          title="Szerep felülírása: a főszereplő hangja kizárólagos, a mellékszereplő osztozhat">${c.role_override
            ? 'vissza az automatára' : (role === 'fő' ? 'tegyük mellékké' : 'tegyük fővé')}</button>
      </div>
    </div>`;
  }

  function poolChip(v) {
    const users = (v.users || []);
    return `<button type="button" class="sync-poolchip${v.model_present ? '' : ' warn'}"
      data-preview-voice="${esc(v.vid)}" title="${esc(v.engine)} · ${esc(v.gender)}${v.model_present ? '' : ' · nincs modell-mappa'}${users.length ? ` · kiosztva ${users.length}x` : ''}"
      ${v.model_present ? '' : ' disabled'}>${esc(v.vid)}
      ${users.length ? `<span class="pool-n">${users.length}x</span>` : ''}</button>`;
  }

  function renderCasting(data) {
    const zone = $('panel-voices');
    if (!zone) return;
    const casting = data.casting || {};
    const summary = casting.summary || {};
    const voicePool = data.voice_pool || [];
    const bits = [];
    if (summary.n_main !== undefined && summary.n_main !== null) {
      bits.push(`<span class="badge-soft ok">${summary.n_main} fő · ${summary.n_side} mellék</span>`);
    }
    bits.push(`<span class="badge-soft ${(summary.n_open || 0) + (summary.n_ready || 0) ? 'warn' : 'ok'}">${(summary.n_open || 0) + (summary.n_ready || 0)} döntés rád vár</span>`);
    bits.push(`<span class="badge-soft ok">${summary.n_assigned || 0} automatikusan kiosztva</span>`);
    $('voices-summary').innerHTML = bits.join(' ');
    const nReady = summary.n_ready || 0;
    const bulkBtn = $('voice-accept-ready');
    if (bulkBtn) {
      bulkBtn.hidden = nReady === 0;
      bulkBtn.textContent = nReady === 1
        ? 'Biztos javaslat elfogadása (1 db)'
        : `Biztos javaslatok elfogadása (${nReady} db)`;
    }
    // ── nyitott döntések: ez az egyetlen hely, ahol az ember kér ─────────
  const open = (casting.open || []).concat((casting.ready || []).map((r) => ({ ...r, ready: true })));
  const assigned = casting.assigned || [];
  $('voices-open-count').textContent = open.length
    ? open.length === 1 ? '1 döntés' : `${open.length} döntés`
    : 'nincs megoldandó döntés — az automatizmus lezárta a kiosztást';
  // ready-elemek: a ready-ket a köteggel is elfogadhatod — de itt is megjelenik, mert szembetűnő legyen
  $('voices-open').innerHTML = open.length
    ? open.map((row) => castOpenRow(row)).join('')
    : '<p class="sync-empty">Semmi dolgod: minden beszélőnek az automatizmus adott hangot, vagy a döntésed már rögzített.</p>';
    // ── kiosztottak csak vizuális követésre ──────────────────────────────
    $('voices-assigned').innerHTML = assigned.length
      ? assigned.map(castAssignedRow).join('')
      : '<p class="sync-empty">Még nincs hangja senkinek — a döntések után itt látszik a kiosztás.</p>';
    // ── szereplők súly szerint ───────────────────────────────────────────
    const characters = data.characters || [];
    const frameMap = frameMapFromSpeakers(data.speakers || []);
    const ep = data.episode;
    // csak az ebben az epizódban szereplőket mutatjuk előre; a többi lenyitható
    const inEpisode = characters.filter((c) => (c.episodes || []).includes(ep));
    const elsewhere = characters.filter((c) => !(c.episodes || []).includes(ep));
    const isMain = (c) => (c.role_override || c.role || 'mellék') === 'fő';
    const mainChars = inEpisode.filter(isMain);
    const sideChars = inEpisode.filter((c) => !isMain(c));
    $('voices-main').innerHTML = mainChars.length
      ? mainChars.map((c) => castCharRow(c, frameMap)).join('')
      : '<p class="sync-reason">Nincs főszereplő ebben az epizódban.</p>';
    const sideHtml = sideChars.length
      ? sideChars.map((c) => castCharRow(c, frameMap)).join('')
      : '<p class="sync-reason">Nincs mellékszereplő ebben az epizódban.</p>';
    $('voices-side').innerHTML = sideHtml + (elsewhere.length
      ? `<details class="sync-advanced"><summary>Korábbi / más epizódok szereplői (${elsewhere.length})</summary>
           ${elsewhere.map((c) => castCharRow(c, frameMap)).join('')}</details>`
      : '');
    // ── hangkészlet minden mintával ──────────────────────────────────────
    $('voices-pool').innerHTML = voicePool.map(poolChip).join('');
  }

  function findCastRow(speaker) {
    const c = (state.data || {}).casting || {};
    const all = [...(c.open || []), ...(c.ready || []), ...(c.assigned || [])];
    return all.find((r) => r.speaker === speaker) || null;
  }

  async function sendCastVoice(speaker, voice) {
    try {
      await Auris.api('/api/dub/voice', {
        method: 'POST',
        body: {
          project: state.project, episode: state.episode,
          speaker, voice, accepted: true,
          note: 'Hangkiosztás döntés',
        },
      });
      await loadState();
      setStatus(`Hang rögzítve: ${speaker} → ${voice}.`);
    } catch (err) {
      setStatus(`A hang nem rögzíthető: ${err.message}`, 'error');
    }
  }

  async function sendCastRole(charId, role) {
    try {
      await Auris.api('/api/dub/char', {
        method: 'POST',
        body: { project: state.project, char_id: charId, role },
      });
      await loadState();
      setStatus(`${charId} szerepe ${role ? (role === 'fő' ? 'főszereplő' : 'mellékszereplő') : 'automatikus'} — a hangzár ennek megfelelően tartalékol.`);
    } catch (err) {
      setStatus(`A szerep nem menthető: ${err.message}`, 'error');
    }
  }

  async function bulkAccept() {
    await withBusy($('voice-accept-ready'), 'Elfogad…', async () => {
      try {
        const res = await Auris.api('/api/dub/voice/bulk', {
          method: 'POST',
          body: { project: state.project, episode: state.episode },
        });
        const acc = (res.accepted || []).map((a) => `${a.speaker} → ${a.voice}`);
        const sk = res.skipped || {};
        const parts = [`Kötegelt elfogadás: ${acc.length} hang kiosztva.`];
        if ((sk.already || 0)) parts.push(`${sk.already} már megvolt`);
        if ((sk.uncertain || 0)) parts.push(`${sk.uncertain} bizonytalan — döntés vár rád`);
        if ((sk.needs_identity || 0)) parts.push(`${sk.needs_identity} azonosítás nélkül`);
        if ((sk.needs_human || 0)) parts.push(`${sk.needs_human} emberi döntéses`);
        setStatus(parts.join(' · '));
      } catch (err) {
        setStatus(`A kötegelt elfogadás nem sikerült: ${err.message}`, 'error');
      }
    });
  }

  // ── hang és keverés: három fogaskerék, alapból automatikus ───────────────
  function renderSound(sound) {
    state.sound = sound || {};
    const duck = Number(sound.duck_db != null ? sound.duck_db : 4);
    const over = Number(sound.dialogue_over_bed_db != null ? sound.dialogue_over_bed_db : 4);
    const bed = Number(sound.bed_db != null ? sound.bed_db : 0);
    const sduck = Number(sound.bed_speech_duck_db != null ? sound.bed_speech_duck_db : 0);
    const d = $('sound-duck'), o = $('sound-over'), q = $('sound-quality');
    const fmt = (v) => `${Number(v).toFixed(1).replace('.0', '')} dB`;
    if (d) { d.value = duck; $('sound-duck-val').textContent = fmt(duck); }
    if (o) { o.value = over; $('sound-over-val').textContent = fmt(over); }
    const b = $('sound-bed');
    if (b) { b.value = bed; $('sound-bed-val').textContent = bed ? fmt(bed) : '0 dB'; }
    const sd = $('sound-speechduck');
    if (sd) {
      sd.value = sduck;
      $('sound-speechduck-val').textContent = sduck ? fmt(-sduck) : 'ki';
    }
    // kézi magyar hang: csak egyszemélyes videónál érvényesül, sorozatnál a
    // szereplőnkénti kiosztás dönt — a választót letiltjuk és figyelmeztetünk
    const vs = $('sound-voice');
    const current = sound.voice_override || '';
    const speakers = Number(sound.voice_speakers != null ? sound.voice_speakers : 0);
    const single = sound.voice_single !== false;
    if (vs) {
      const pool = (state.data || {}).voice_pool || [];
      vs.innerHTML = '<option value="">Automatikus (a program választ)</option>'
        + pool.map((v) => `<option value="${esc(v.vid)}"${v.vid === current ? ' selected' : ''}>`
          + `${esc(v.vid)} · ${esc(v.gender)}${v.model_present ? '' : ' · nincs modell'}</option>`).join('');
      vs.value = current;
      if (current && !pool.some((v) => v.vid === current)) {
        vs.insertAdjacentHTML('beforeend',
          `<option value="${esc(current)}" selected>${esc(current)}</option>`);
        vs.value = current;
      }
      vs.disabled = !single;
    }
    const voicePlayBtn = $('sound-voice-play');
    if (voicePlayBtn) voicePlayBtn.disabled = !single;
    const note = $('sound-voice-note');
    if (note) {
      note.hidden = single;
      note.textContent = single ? ''
        : `Ez a projekt ${speakers} szereplős — a kézi hang csak egyszemélyes `
          + 'videónál érvényesül, itt a Szereplők panelenkénti kiosztás dönt.'
          + (current ? ` A mentett kézi hang (${current}) ezért nem szól.` : '');
    }
    if (q) q.value = sound.quality || 'auto';
    const steps = $('sound-steps');
    if (steps) {
      steps.textContent = sound.synth_steps ? `${sound.synth_steps} lépés` : 'automatikus';
    }
    const info = $('sound-info');
    if (info) {
      info.textContent = (!sound.quality || sound.quality === 'auto')
        ? 'gyári értékek' : 'kézi beállítás';
    }
  }

  async function saveSound(body) {
    if (!state.project) return;
    try {
      const res = await Auris.api('/api/dub/sound', {
        method: 'POST', body: { project: state.project, ...body },
      });
      renderSound(res.sound || {});
      setStatus('Hang-beállítás mentve — a következő futásnál érvényes.', 'ok');
    } catch (err) {
      setStatus(err.message || 'A mentés nem sikerült.', 'err');
    }
  }

  async function savePitch(charId, pitch) {
    if (!state.project) return;
    let ok = true;
    try {
      await Auris.api('/api/dub/pitch', {
        method: 'POST',
        body: { project: state.project, char_id: charId, pitch },
      });
    } catch (err) {
      ok = false;
      setStatus(err.message || 'A hangszín mentése nem sikerült.', 'err');
    }
    // az újratöltés felülírja a státuszsort, ezért az üzenet utána jön
    await loadState();
    if (ok) {
      setStatus(pitch == null
        ? `${charId}: hangszín visszaállítva automatikusra.`
        : `${charId}: hangszín ${pitch > 0 ? '+' : ''}${pitch} félhang — a következő futásnál érvényes.`, 'ok');
    }
  }

  function renderModel(data) {
    const audit = data.audit || {};
    const method = audit.method || {};
    const limits = (audit.limitations || []).map((l) => `<li>${esc(l)}</li>`).join('');
    const d = (data.speakers || []).find((s) => s.proposal) || {};
    const model = d.proposal || null;
    $('sync-model').innerHTML = `
      <h3>Hogyan dönt a modell?</h3>
      <p>Tipizált döntési modell (<code>choice</code> / <code>score</code> / <code>noul</code>) — nem nyelvi modell.
      A hangot mérhető bizonyítékok alapján választja: nem, hangszín-koszinusz (a TTS-hangok és a
      forrás-színész CAM++ beágyazása), a hang szabad-e, nem szólal-e meg ugyanabban a jelenetben,
      és hogy a szereplő korábbi hangja megmaradjon. Az indoklás minden kártyán látszik.</p>
      <p>Illesztés: ${esc(method.alignment || 'nincs adat')}<br>
      Diarizálás: ${esc(method.diarization || 'nincs adat')}<br>
      Azonosság: ${esc(method.identity_link || 'nincs adat')}
      ${model ? `<br>Hangszín-forrás: ${esc(model.candidates && model.candidates[0] && model.candidates[0].timbre_cosine !== null ? 'mért CAM++ koszinusz' : 'nincs mérés (semleges)')}` : ''}</p>
      <ul>${limits}</ul>`;
  }

  // ── műveletek ───────────────────────────────────────────────────────────
  async function withBusy(button, label, fn) {
    const old = button ? button.textContent : '';
    if (button) { button.disabled = true; button.textContent = label; }
    try { return await fn(); }
    finally { if (button) { button.disabled = false; button.textContent = old; } }
  }

  async function runAudit() {
    if (!state.project || !state.episode) { setStatus('Válassz projektet és epizódot.', 'error'); return; }
    const start = Number($('sync-start').value || 0);
    const duration = Number($('sync-duration').value || 30);
    setStatus(`Próba fut (${start}–${start + duration} mp)… ez a diarizálástól és az igazítástól függően 1–3 percig tart.`, 'busy');
    await withBusy($('sync-audit-run'), 'Fut…', async () => {
      try {
        const res = await Auris.api('/api/dub/audit', {
          method: 'POST',
          body: {
            project: state.project, episode: state.episode, start, duration,
            language: ($('sync-language') || {}).value || 'en',
          },
        });
        state.audit = res.audit;
        await loadState();
        setStatus(`Kész: ${res.audit}`);
      } catch (err) {
        setStatus(`A próba nem sikerült: ${err.message}`, 'error');
      }
    });
  }

  async function measureTimbres() {
    setStatus('Hangszín-mérés fut (minden elérhető TTS-hang egyszer)…', 'busy');
    await withBusy($('sync-timbres'), 'Mér…', async () => {
      try {
        const res = await Auris.api('/api/dub/timbres', {
          method: 'POST', body: { project: state.project },
        });
        await loadState();
        setStatus(`Hangszín kész: ${res.voices} hang.`);
      } catch (err) {
        setStatus(`Hangszín-mérés hiba: ${err.message}`, 'error');
      }
    });
  }

  async function sendSpeaker(sp, decision, charId) {
    const card = document.querySelector(`[data-speaker="${CSS.escape(sp.speaker)}"]`);
    try {
      await Auris.api('/api/dub/speaker', {
        method: 'POST',
        body: {
          project: state.project,
          episode: state.episode,
          speaker: sp.speaker,
          decision,
          char_id: charId || null,
          gender: sp.gender || null,
          voice: sp.voice || null,
          note: '',
          name: (card && card.querySelector('[data-name-input]') || {}).value || null,
          evidence_seconds: sp.evidence_seconds || 0,
        },
      });
      await loadState();
      setStatus(`${sp.speaker}: ${decision} rögzítve.`);
    } catch (err) {
      setStatus(`Nem sikerült rögzíteni: ${err.message}`, 'error');
      if (card) card.classList.remove('is-pending');
    }
  }

  async function sendVoice(sp, voice) {
    try {
      await Auris.api('/api/dub/voice', {
        method: 'POST',
        body: {
          project: state.project, episode: state.episode, speaker: sp.speaker,
          voice, accepted: true, char_id: sp.char_id || null, gender: sp.gender || null,
          evidence_seconds: sp.evidence_seconds || 0,
        },
      });
      await loadState();
      setStatus(`${sp.speaker}: hang = ${voice}.`);
    } catch (err) {
      setStatus(`Hang rögzítése nem sikerült: ${err.message}`, 'error');
    }
  }

  // ── Beállítások (projekt-JSON a WebUI-ból) ──────────────────────────────

  const SETTING_GROUPS = [
    { key: 'meta',    label: 'Projekt',    flat: ['name', 'season', 'tts_sr', 'source_lang'] },
    { key: 'models',  label: 'Modellek',   note: 'útvonalak (relatív a dubstudio mappához vagy abszolút)' },
    { key: 'tuning',  label: 'Klaszterezés (tuning)' },
    { key: 'gate',    label: 'QA-kapuk (gate)' },
    { key: 'llm',     label: 'Szöveg-ellenőrző LLM (llm)' },
    { key: 'asr',     label: 'ASR-visszaellenőrzés (asr)' },
  ];

  function flatten(obj, prefix, out) {
    out = out || {};
    for (const [k, v] of Object.entries(obj || {})) {
      const path = prefix ? `${prefix}.${k}` : k;
      if (v && typeof v === 'object' && !Array.isArray(v) && Object.keys(v).length) {
        flatten(v, path, out);
      } else {
        out[path] = Array.isArray(v) ? JSON.stringify(v) : v;
      }
    }
    return out;
  }

  function unflatten(flat) {
    const out = {};
    for (const [path, value] of Object.entries(flat)) {
      const keys = path.split('.');
      let node = out;
      keys.forEach((k, i) => {
        if (i === keys.length - 1) node[k] = value;
        else { node[k] = node[k] && typeof node[k] === 'object' ? node[k] : {}; node = node[k]; }
      });
    }
    return out;
  }

  async function loadConfig() {
    const payload = await Auris.api(`/api/dub/config?project=${encodeURIComponent(state.project || '')}`);
    state.config = payload;
    renderConfig(payload);
  }

  function fieldInput(path, value) {
    if (typeof value === 'boolean') {
      return `<input class="text-input" data-path="${esc(path)}" type="checkbox" ${value ? 'checked' : ''}>`;
    }
    const num = typeof value === 'number' && Number.isFinite(value);
    const shown = (value === null || value === undefined) ? '' : value;
    return `<input class="text-input" data-path="${esc(path)}" type="${num ? 'number' : 'text'}"
      ${num ? 'step="any"' : ''} value="${esc(String(shown))}">`;
  }

  function renderConfig(payload) {
    const el = $('sync-config-form');
    const settings = payload.settings || {};
    if (state.project && payload.project !== state.project) {
      // a szerver az első projektet adta vissza — jelezzük, melyiket látjuk
    }
    const meta = Object.assign({
      name: payload.project, season: 1, tts_sr: 44100, source_lang: 'heb',
    }, payload.meta || {});
    const groups = SETTING_GROUPS.map(({ key, label, note, flat }) => {
      let rows = '';
      if (key === 'meta') {
        rows = flat.map((f) => `<label class="sync-form-row">${esc(f)}${fieldInput(`meta.${f}`, meta[f])}</label>`).join('');
      } else {
        const flat2 = flatten(settings[key] || {}, '');
        const names = Object.keys(flat2).sort();
        rows = names.length
          ? names.map((p) => `<label class="sync-form-row">${esc(p)}${fieldInput(`${key}.${p}`, flat2[p])}</label>`).join('')
          : '<p class="sync-reason">nincs bejegyzés</p>';
      }
      return `<details class="sync-config-group">
        <summary>${esc(label)}</summary>${note ? `<p class="sync-reason">${esc(note)}</p>` : ''}${rows}</details>`;
    }).join('');
    const voices = (payload.voices || []).map((v) =>
      `<button type="button"
        class="sync-voicechip ${v.model_present ? 'ok' : 'warn'}"
        data-preview-voice="${esc(v.vid)}"
        title="${esc(v.engine)} · nem: ${esc(v.gender)}${v.model_present ? '' : ' · nincs modell-mappa — Beállítások → Modellek'}"${v.model_present ? '' : ' disabled'}
        >${esc(v.vid)}
      </button>`
    ).join('');
    el.innerHTML = `${groups}
      <p class="sync-hint">Hangkészlet (modell-mappa jelenléte szerint): <span class="sync-voicerow">${voices}</span></p>`;
    $('sync-config-info').textContent = `projekt: ${esc(state.project || payload.project)}`;
  }

  function collectConfig() {
    const flat = {};
    document.querySelectorAll('#sync-config-form [data-path]').forEach((input) => {
      const v = input.dataset.path;
      if (input.type === 'checkbox') { flat[v] = input.checked; return; }
      let value = input.value.trim();
      if (value === '') return;                                  // üres: nem bántjuk
      if (input.type === 'number') {
        const num = Number(value);
        if (!Number.isFinite(num)) return;
        value = num;
      } else if (value.startsWith('[') && value.endsWith(']')) {
        // lista-érték (pl. glob-minta): visszaalakítjuk JSON-ból
        try { value = JSON.parse(value); } catch { /* marad szöveg */ }
      }
      flat[v] = value;
    });
    const config = unflatten(flat);
    // gap_merge 0 → a szerver üres/hiányzó mezőt változatlanul hagy; a true/false marad
    const meta = config.meta || {};
    delete config.meta;
    return Object.assign(meta, config);
  }

  async function saveConfig() {
    if (!state.project) { setStatus('Nincs kiválasztott projekt.', 'error'); return; }
    await withBusy($('sync-config-save'), 'Ment…', async () => {
      try {
        const res = await Auris.api('/api/dub/config', {
          method: 'POST',
          body: { project: state.project, config: collectConfig() },
        });
        setStatus(`Beállítások mentve (biztonsági mentés: ${res.backup}).`);
        await loadConfig();
      } catch (err) {
        setStatus(`A mentés nem sikerült: ${err.message}`, 'error');
      }
    });
  }

  // ── Futtatás (dubstudio CLI) ────────────────────────────────────────────

  async function runJob(step) {
    if (!state.project || !state.episode && step !== 'cluster') {
      setStatus('Válassz epizódot (a klaszterezés az évadról megy).', 'error'); return;
    }
    const entry = projectEntry();
    const episodes = step === 'cluster'
      ? ((entry && entry.episodes) || []).slice()
      : [state.episode];
    const body = {
      project: state.project, step, episodes,
      limit: Number(($('run-limit') || {}).value || 0) || 0,
      no_mux: !!($('run-nomux') || {}).checked,
      allow_fail: !!($('run-allowfail') || {}).checked,
    };
    setStatus(`Elindul: ${step} (${episodes.join(', ')})…`, 'busy');
    try {
      await Auris.api('/api/dub/run', { method: 'POST', body });
      state.runState = 'running';
      await pollRun(true);
    } catch (err) {
      setStatus(`A futtatás nem indult el: ${err.message}`, 'error');
    }
  }

  async function stopRun() {
    try {
      await Auris.api('/api/dub/run/stop', { method: 'POST', body: {} });
      setStatus('Leállítás kérve.');
      await pollRun(true);
    } catch (err) {
      setStatus(`A leállítás nem sikerült: ${err.message}`, 'error');
    }
  }

  function runButtonsEnabled(enabled) {
    ['intake', 'prepare', 'cluster', 'dub', 'run', 'asr'].forEach((s) => {
      const btn = $(`run-${s}`);
      if (btn) btn.disabled = !enabled;
    });
    const stop = $('run-stop');
    if (stop) stop.hidden = enabled;
  }

  async function pollRun(once = false) {
    try {
      const job = await Auris.api('/api/dub/run');
      const st = $('sync-run-state');
      const log = $('sync-run-log');
      if (job.log_tail && log) {
        log.textContent = job.log_tail.join('\n');
        log.scrollTop = log.scrollHeight;
      }
      const running = job.state === 'running';
      if (running) {
        if (st) st.textContent = `fut: ${job.step} (${(job.episodes || []).join(', ')}) — elindítva ${job.started_at}`;
        runButtonsEnabled(false);
        // hosszú futás alatt is kövesse a könyvtár sora a készülő anyagokat
        const now = Date.now();
        if (!state.libTick || now - state.libTick > 30000) {
          state.libTick = now;
          loadLibrary();
        }
      } else {
        if (st) st.textContent = job.state === 'idle' ? 'nincs futó feladat'
          : `befejeződött (${job.state}, kimenet: ${job.returncode}) — ${job.step}`;
        runButtonsEnabled(true);
      }
      const wasRunning = state.runState === 'running';
      if (wasRunning && !running && job.state !== 'idle') {
        if (job.state === 'finished') setStatus(`${job.step} kész. Ugyanaz a napló: ${job.log}`);
        else setStatus(`${job.step} hibával zárt (kód: ${job.returncode}). Lásd: ${job.log}`, 'error');
        await loadState();
        await loadLibrary();
        await loadPlayer();
      }
      state.runState = running ? 'running' : job.state || 'idle';
    } catch { /* a státusz lekérés kimaradhat — a UI marad */ }
    if (!once) state.runTimer = null;                              // az időzítő az auditálás után újraindul
  }

  function startRunPolling() {
    setInterval(pollRun, 3000);
    pollRun(true);
  }

  function playClip(start, duration) {
    const audio = $('sync-audio');
    const params = new URLSearchParams({
      project: state.project, episode: state.episode,
      start: String(start), duration: String(duration),
    });
    if (state.data && state.data.audit && state.data.audit.audio_stream !== undefined) {
      params.set('stream', String(state.data.audit.audio_stream));
    }
    audio.src = `/api/dub/clip?${params.toString()}`;
    audio.play().catch(() => setStatus('A hang nem indítható (böngésző vagy ffmpeg hiba).', 'error'));
  }

  // ── hang-minta-előhallgatás (magyar mintamondat) ─────────────────────────
  function previewVoiceUrl(vid) {
    const params = new URLSearchParams({ project: state.project || '', voice: vid });
    return `/api/dub/voicepreview?${params.toString()}`;
  }

  async function previewVoice(vid) {
    if (!vid) { setStatus('Nincs kijelölt hang.', 'error'); return; }
    const audio = $('sync-audio');
    if (audio.dataset.previewing === vid && !audio.paused) {
      audio.pause();
      audio.dataset.previewing = '';
      return;
    }
    setStatus(`${vid} minta betöltése… (az első kérés betölti a TTS-motort, ez 1–2 perc lehet)`, 'busy');
    audio.src = previewVoiceUrl(vid);
    audio.dataset.previewing = vid;
    state.audioMode = 'preview';
    try {
      await audio.play();
      setStatus(`${vid} — a magyar mintamondat szólal meg; a ▲ gombbal bármikor ment zajlik.`);
      audio.dataset.previewing = '';
    } catch {
      setStatus('A hangminta nem indítható (böngésző vagy szerverhiba).', 'error');
      audio.dataset.previewing = '';
    }
  }

  // ── film betöltése ─────────────────────────────────────────────────────
  function suggestTagFromName(name) {
    let m = /(?:[Ss](\d{1,2}))[\. _-]?[Ee][\. ]?(\d{1,3})/.exec(name || '');
    if (!m) m = /(?:[^\dk]|^)(?:[Ee](\d{1,3})(?!\d))/.exec(name || '');
    if (!m) m = /(?:epiz[óo]d[ \-.:_]*(\d{1,3}))/i.exec(name || '');
    if (!m) return null;
    return `E${String(Number(m[2] || m[1])).padStart(2, '0')}`;
  }

  async function loadMovie() {
    const path = ($('load-path').value || '').trim();
    const tag = ($('load-tag').value || '').trim().toUpperCase();
    if (!path) { setStatus('Add meg a filmfájl útvonalát.', 'error'); return; }
    if (!tag) { setStatus('Add meg az epizód-azonosítót (pl. E05).', 'error'); return; }
    await withBusy($('load-movie'), 'Betölt…', async () => {
      try {
        const res = await Auris.api('/api/dub/loadmovie', {
          method: 'POST', body: { project: state.project, path, episode: tag },
        });
        const p = res.probe || {};
        $('load-result').innerHTML =
          `<span class="ok">Betöltve (${esc(res.mode)}): ${esc(res.movie)}</span>` +
          (p.duration ? ` — ${Math.round(Number(p.duration) / 60)} perc` : '') +
          (p.audio && p.audio.length ? ` · ${p.audio.length} hangsáv` : '') +
          (p.subtitles && p.subtitles.length ? ` · feliratsávok: ${p.subtitles.map(esc).join(', ')}` : '');
        setStatus(`Film betöltve: ${tag}. Az „Előkészítés + próba” léptéssel indul az azonosítás.`);
        await loadLibrary();
        fillEpisodes();
        await selectEpisode(res.episode || tag);
      } catch (err) {
        $('load-result').innerHTML = `<span class="err">${esc(err.message)}</span>`;
        setStatus(`A film betöltése nem sikerült: ${err.message}`, 'error');
      }
    });
  }

  // ── események ───────────────────────────────────────────────────────────
  // ── helyi fájlválasztó (a szerver gépén) ───────────────────────────────
  async function pickPath(kind, initial, button) {
    const old = button ? button.textContent : '';
    if (button) { button.disabled = true; button.textContent = 'Választás…'; }
    try {
      const res = await Auris.api('/api/dub/pick', {
        method: 'POST', body: { kind, initial: initial || '' },
      });
      return res.path || '';
    } catch (err) {
      setStatus(`A fájlválasztó nem mûködött: ${err.message}`, 'error');
      return '';
    } finally {
      if (button) { button.disabled = false; button.textContent = old; }
    }
  }

  // ── médiaszerver: beállítás, böngészés, visszamásolás ───────────────────
  async function loadMediaConfig() {
    if (!state.project) return;
    try {
      const cfg = await Auris.api(`/api/dub/media/config?project=${encodeURIComponent(state.project)}`);
      $('media-root').value = cfg.path || '';
      $('media-deliver').checked = Boolean(cfg.deliver);
      $('media-status').textContent = !cfg.path ? 'nincs beállítva'
        : cfg.exists ? 'elérhető' : 'nem érhető el';
      if (cfg.path && cfg.exists) await listMedia(cfg.path);
      else $('media-files').innerHTML = '';
    } catch (err) {
      $('media-status').textContent = `hiba: ${err.message}`;
    }
  }

  async function saveMediaConfig() {
    const path = ($('media-root').value || '').trim();
    const deliver = $('media-deliver').checked;
    await withBusy($('media-root-save'), 'Mentés…', async () => {
      try {
        const res = await Auris.api('/api/dub/media/config', {
          method: 'POST', body: { project: state.project, path, deliver },
        });
        $('media-status').textContent = res.exists ? 'elérhető' : 'nem érhető el';
        $('media-result').innerHTML = `<span class="ok">Médiaszerver mentve: ${esc(res.path || '—')}` +
          `${deliver ? ' · visszamásolás bekapcsolva' : ''}</span>`;
        if (res.path && res.exists) await listMedia(res.path);
      } catch (err) {
        $('media-result').innerHTML = `<span class="err">${esc(err.message)}</span>`;
      }
    });
  }

  async function listMedia(path) {
    if (!state.project) return;
    const q = new URLSearchParams({ project: state.project });
    if (path) q.set('path', path);
    try {
      const res = await Auris.api(`/api/dub/media/list?${q.toString()}`);
      $('media-subpath').value = res.path;
      $('media-up').hidden = !res.parent;
      $('media-up').dataset.parent = res.parent || '';
      const rows = [];
      for (const dir of res.dirs || []) {
        rows.push(`<div class="sync-librow" role="listitem">` +
          `<button class="btn btn-sm btn-ghost" type="button" data-media-dir="${esc(dir.path)}">` +
          `📁 ${esc(dir.name)}</button></div>`);
      }
      for (const file of res.files || []) {
        const mb = Math.round(Number(file.size || 0) / (1024 * 1024));
        rows.push(`<div class="sync-librow" role="listitem">` +
          `<span class="sync-libname">🎬 ${esc(file.name)}` +
          `<small>${file.tag ? esc(file.tag) : 'nincs epizódszám'} · ${mb} MB</small></span>` +
          `<button class="btn btn-sm btn-primary" type="button" data-media-add="${esc(file.path)}"` +
          ` data-media-tag="${esc(file.tag || '')}">Hozzáadás</button></div>`);
      }
      $('media-files').innerHTML = rows.join('') ||
        '<p class="sync-empty">Ebben a mappában nincs videó vagy alkönyvtár.</p>';
      $('media-result').textContent = `${res.path} — ${(res.dirs || []).length} mappa, ` +
        `${(res.files || []).length} videó`;
    } catch (err) {
      $('media-files').innerHTML = '';
      $('media-result').innerHTML = `<span class="err">${esc(err.message)}</span>`;
    }
  }

  async function mediaAdd(path, tag) {
    $('media-result').textContent = 'Másolás a laptopra…';
    try {
      const res = await Auris.api('/api/dub/series/add', {
        method: 'POST', body: { project: state.project, path, episode: tag || undefined },
      });
      const added = res.added || [];
      const skipped = res.skipped || [];
      $('media-result').innerHTML = (
        added.length ? `<span class="ok">Átmásolva: ${added.map((a) => `${esc(a.tag)} (${esc(a.mode)})`).join(', ')}</span>` : ''
      ) + (skipped.length ? `<span class="warn">Kihagyva: ${skipped.map((s) => esc(s.reason)).join('; ')}</span>` : '');
      await loadLibrary();
      fillEpisodes();
      if (added.length) await selectEpisode(added[added.length - 1].tag);
    } catch (err) {
      $('media-result').innerHTML = `<span class="err">${esc(err.message)}</span>`;
    }
  }

  async function deliverNow() {
    if (!state.episode) { setStatus('Előbb válassz epizódot.', 'error'); return; }
    await withBusy($('media-deliver-now'), 'Másolás…', async () => {
      try {
        const res = await Auris.api('/api/dub/media/deliver', {
          method: 'POST', body: { project: state.project, episode: state.episode },
        });
        $('media-result').innerHTML = `<span class="ok">Visszamásolva: ${esc(res.target)}` +
          `</span><span class="sync-configinfo">biztonsági másolat: ${esc(res.backup)}</span>`;
      } catch (err) {
        $('media-result').innerHTML = `<span class="err">${esc(err.message)}</span>`;
      }
    });
  }

  function wire() {
    // ── hangszín-csúszka (szereplőnként, a sorok újrarajzolódnak) ──────────
    document.addEventListener('input', (ev) => {
      const el = ev.target.closest && ev.target.closest('input[data-pitch]');
      if (!el) return;
      const out = document.querySelector(`[data-pitchval="${el.dataset.pitch}"]`);
      const v = Number(el.value);
      if (out) out.textContent = `${v > 0 ? '+' : ''}${v.toFixed(1)}`;
    });
    document.addEventListener('change', (ev) => {
      const el = ev.target.closest && ev.target.closest('input[data-pitch]');
      if (!el) return;
      savePitch(el.dataset.pitch, Number(el.value));
    });
    document.addEventListener('click', (ev) => {
      const btn = ev.target.closest && ev.target.closest('[data-pitch-reset]');
      if (!btn) return;
      savePitch(btn.dataset.pitchReset, null);
    });

    // ── hang és keverés ────────────────────────────────────────────────────
    const duckEl = $('sound-duck');
    if (duckEl) {
      duckEl.addEventListener('input', () => {
        $('sound-duck-val').textContent =
          `${Number(duckEl.value).toFixed(1).replace('.0', '')} dB`;
      });
    }
    const overEl = $('sound-over');
    if (overEl) {
      overEl.addEventListener('input', () => {
        $('sound-over-val').textContent =
          `${Number(overEl.value).toFixed(1).replace('.0', '')} dB`;
      });
    }
    const bedEl = $('sound-bed');
    if (bedEl) {
      bedEl.addEventListener('input', () => {
        const v = Number(bedEl.value);
        $('sound-bed-val').textContent = v ? `${v} dB` : '0 dB';
      });
    }
    const sduckEl = $('sound-speechduck');
    if (sduckEl) {
      sduckEl.addEventListener('input', () => {
        const v = Number(sduckEl.value);
        $('sound-speechduck-val').textContent = v ? `-${v} dB` : 'ki';
      });
    }
    const voicePlay = $('sound-voice-play');
    if (voicePlay) {
      voicePlay.addEventListener('click', () => {
        if (voicePlay.disabled) return;
        previewVoice($('sound-voice').value);
      });
    }
    const soundSave = $('sound-save');
    if (soundSave) {
      soundSave.addEventListener('click', () => saveSound({
        duck_db: Number($('sound-duck').value),
        dialogue_over_bed_db: Number($('sound-over').value),
        bed_db: Number($('sound-bed').value),
        bed_speech_duck_db: Number($('sound-speechduck').value),
        voice: $('sound-voice').value,
        quality: $('sound-quality').value,
      }));
    }
    const soundReset = $('sound-reset');
    if (soundReset) {
      soundReset.addEventListener('click', () => saveSound({
        duck_db: null, dialogue_over_bed_db: null, bed_db: null,
        bed_speech_duck_db: null, voice: null,
        quality: 'auto', synth_steps: null,
      }));
    }

    $('sync-project').addEventListener('change', (e) => {
      state.project = e.target.value;
      state.audit = null;
      loadConfig().catch((err) => setStatus(`A beállítások nem tölthetők be: ${err.message}`, 'error'));
      loadLibrary().then(() => { fillEpisodes(); loadState(); loadPlayer(); loadMediaConfig(); });
    });
    $('sync-episode').addEventListener('change', (e) => {
      state.episode = e.target.value || null;
      state.audit = null;
      const subTag = $('subtitle-episode');
      if (subTag) subTag.value = state.episode || '';
      loadState();
      loadPlayer();
    });
    $('sync-audit').addEventListener('change', (e) => {
      state.audit = e.target.value || null;
      loadState();
    });
    $('sync-audit-run').addEventListener('click', runAudit);
    $('sync-timbres').addEventListener('click', measureTimbres);
    const loadBtn = $('load-movie');
    if (loadBtn) loadBtn.addEventListener('click', loadMovie);
    const libAdd = $('library-add');
    if (libAdd) libAdd.addEventListener('click', addToLibrary);
    const libScan = $('library-scan');
    if (libScan) libScan.addEventListener('click', scanLibrary);
    const libPick = $('library-pick');
    if (libPick) libPick.addEventListener('click', async () => {
      const path = await pickPath('file', $('library-path').value, libPick);
      if (path) $('library-path').value = path;
    });
    const libPickDir = $('library-pick-dir');
    if (libPickDir) libPickDir.addEventListener('click', async () => {
      const path = await pickPath('dir', $('library-path').value, libPickDir);
      if (path) $('library-path').value = path;
    });
    const moviePick = $('movie-pick');
    if (moviePick) moviePick.addEventListener('click', async () => {
      const path = await pickPath('file', $('movie-path').value, moviePick);
      if (path) $('movie-path').value = path;
    });
    const loadPick = $('load-pick');
    if (loadPick) loadPick.addEventListener('click', async () => {
      const path = await pickPath('file', $('load-path').value, loadPick);
      if (path) {
        $('load-path').value = path;
        const guess = suggestTagFromName(path);
        if (guess) $('load-tag').value = guess;
      }
    });
    const mediaRootPick = $('media-root-pick');
    if (mediaRootPick) mediaRootPick.addEventListener('click', async () => {
      const path = await pickPath('dir', $('media-root').value, mediaRootPick);
      if (path) $('media-root').value = path;
    });
    const mediaRootSave = $('media-root-save');
    if (mediaRootSave) mediaRootSave.addEventListener('click', saveMediaConfig);
    const mediaListBtn = $('media-list');
    if (mediaListBtn) mediaListBtn.addEventListener('click',
      () => listMedia(($('media-subpath').value || '').trim()));
    const mediaSub = $('media-subpath');
    if (mediaSub) mediaSub.addEventListener('keydown', (ev) => {
      if (ev.key === 'Enter') { ev.preventDefault(); listMedia((mediaSub.value || '').trim()); }
    });
    const mediaUp = $('media-up');
    if (mediaUp) mediaUp.addEventListener('click', () => listMedia(mediaUp.dataset.parent || ''));
    const mediaDeliverBtn = $('media-deliver-now');
    if (mediaDeliverBtn) mediaDeliverBtn.addEventListener('click', deliverNow);
    const mediaFiles = $('media-files');
    if (mediaFiles) mediaFiles.addEventListener('click', (ev) => {
      const dir = ev.target.closest('[data-media-dir]');
      if (dir) { listMedia(dir.dataset.mediaDir); return; }
      const add = ev.target.closest('[data-media-add]');
      if (add) mediaAdd(add.dataset.mediaAdd, add.dataset.mediaTag || '');
    });
    const libProcess = $('library-process');
    if (libProcess) libProcess.addEventListener('click', () => processReady(null));
    const libIntake = $('library-intake');
    if (libIntake) libIntake.addEventListener('click', () => runIntake(null));
    const movieCreate = $('movie-create');
    if (movieCreate) movieCreate.addEventListener('click', createMovie);
    const subtitleUpload = $('subtitle-upload');
    if (subtitleUpload) subtitleUpload.addEventListener('click', uploadSubtitle);
    const youtubeStart = $('youtube-start');
    if (youtubeStart) youtubeStart.addEventListener('click', startYoutube);
    const watchSave = $('watch-save');
    if (watchSave) watchSave.addEventListener('click', saveWatch);
    const playerAudio = $('player-audio');
    if (playerAudio) playerAudio.addEventListener('change', (e) => applyPlayerAudio(e.target.value));
    const playerSub = $('player-subtitle');
    if (playerSub) playerSub.addEventListener('change', (e) => applyPlayerSubtitle(e.target.value));
    const playerReload = $('player-reload');
    if (playerReload) playerReload.addEventListener('click', loadPlayer);
    const playerSample = $('player-dub-sample');
    if (playerSample) playerSample.addEventListener('click', playDubSample);
    const loadPath = $('load-path');
    if (loadPath) loadPath.addEventListener('input', () => {
      const guess = suggestTagFromName(loadPath.value);
      if (guess) $('load-tag').value = guess;
    });
    ['intake', 'prepare', 'cluster', 'dub', 'run', 'asr'].forEach((s) => {
      const btn = $(`run-${s}`);
      if (btn) btn.addEventListener('click', () => runJob(s));
    });
    const stopBtn = $('run-stop');
    if (stopBtn) stopBtn.addEventListener('click', stopRun);
    const saveCfg = $('sync-config-save');
    if (saveCfg) saveCfg.addEventListener('click', saveConfig);
    const reloadCfg = $('sync-config-reload');
    if (reloadCfg) reloadCfg.addEventListener('click', () => {
      withBusy(reloadCfg, 'Betölt…', loadConfig)
        .then(() => setStatus('Beállítások visszaolvasva az utolsó mentésről.'))
        .catch((err) => setStatus(`A visszaolvasás nem sikerült: ${err.message}`, 'error'));
    });

    document.querySelectorAll('[data-filter]').forEach((btn) => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('[data-filter]').forEach((b) => b.classList.remove('is-active'));
        btn.classList.add('is-active');
        state.speakerFilter = btn.dataset.filter;
        renderSpeakers(state.data || {});
      });
    });

    document.addEventListener('click', async (event) => {
      const target = event.target.closest('button');
      if (!target) return;
      const card = target.closest('[data-speaker]');
      if (target.dataset.goto !== undefined && target.dataset.goto) {
        const panel = document.getElementById(target.dataset.goto);
        if (panel) panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
        return;
      }
      if (target.dataset.previewVoice !== undefined) {
        await previewVoice(target.dataset.previewVoice);
        return;
      }
      if (target.dataset.playStart !== undefined) {
        playClip(Number(target.dataset.playStart), Number(target.dataset.playDuration || 3));
        return;
      }
      if (target.dataset.libSelect !== undefined) {
        await selectEpisode(target.dataset.libSelect);
        return;
      }
      if (target.dataset.libRun !== undefined) {
        await processReady(target.dataset.libRun);
        return;
      }
      if (target.dataset.libIntake !== undefined) {
        await runIntake(target.dataset.libIntake);
        return;
      }
      // ── Hangkiosztás döntések (nem beszélő-kártyák) ─────────────────────
      const castRow = target.closest('[data-cast]');
      if (target.dataset.castYes !== undefined && castRow) {
        const row = findCastRow(castRow.dataset.cast);
        if (row && row.choice) {
          manualOpen.delete(castRow.dataset.cast);
          await sendCastVoice(castRow.dataset.cast, row.choice);
        }
        return;
      }
      if (target.dataset.castNo !== undefined && castRow) {
        manualOpen.has(castRow.dataset.cast)
          ? manualOpen.delete(castRow.dataset.cast)
          : manualOpen.add(castRow.dataset.cast);
        if (state.data) { renderCasting(state.data); }
        return;
      }
      if (target.dataset.castClose !== undefined && castRow) {
        manualOpen.delete(castRow.dataset.cast);
        if (state.data) { renderCasting(state.data); }
        return;
      }
      if (target.dataset.castApply !== undefined && castRow) {
        const picker = castRow.querySelector('[data-cast-select]');
        if (picker && picker.value) {
          manualOpen.delete(castRow.dataset.cast);
          await sendCastVoice(castRow.dataset.cast, picker.value);
        }
        return;
      }
      const bulk = $('voice-accept-ready');
      if (bulk && target.id === 'voice-accept-ready') {
        await bulkAccept();
        return;
      }
      if (target.dataset.castRole !== undefined) {
        await sendCastRole(target.dataset.charId, target.dataset.role === 'fő' ? 'fő' : 'mellék');
        return;
      }
      if (!card) return;
      const speaker = card.dataset.speaker;
      const sp = ((state.data || {}).speakers || []).find((s) => s.speaker === speaker);
      if (!sp) return;
      if (target.dataset.accept !== undefined) {
        await sendSpeaker(sp, 'accepted', target.dataset.accept || sp.char_id);
      } else if (target.dataset.reject !== undefined) {
        await sendSpeaker(sp, 'rejected', null);
      } else if (target.dataset.correct !== undefined) {
        const select = card.querySelector('[data-char-select]');
        if (!select || !select.value) { setStatus('A javításhoz válassz karaktert a listából.', 'error'); return; }
        await sendSpeaker(sp, 'corrected', select.value);
      } else if (target.dataset.voice !== undefined) {
        await sendVoice(sp, target.dataset.voice);
      } else if (target.dataset.voiceApply !== undefined) {
        const picker = card.querySelector('[data-voice-select]');
        if (!picker || !picker.value) return;
        await sendVoice(sp, picker.value);
      }
    });
  }

  document.addEventListener('DOMContentLoaded', async () => {
    wire();
    try { await loadProjects(); await loadState(); await loadMediaConfig(); }
    catch (err) { setStatus(`Hiba: ${err.message}`, 'error'); }
    try { await loadConfig(); } catch (err) {
      setStatus(`A beállítások nem töltöltek be: ${err.message}`, 'error');
    }
    startRunPolling();
  });
})();
