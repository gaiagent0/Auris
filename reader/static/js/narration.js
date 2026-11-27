// Narráció: egyszereplős videó külön lánca (nincs szereplőkezelés).
(function () {
  'use strict';
  const esc = (v) => Auris.esc(v);
  const $ = (id) => document.getElementById(id);
  const state = { voices: [], cloneAvailable: false, timer: null, running: false };

  function badge(text, kind) {
    const el = $('narr-badge');
    el.textContent = text;
    el.className = 'badge-soft' + (kind ? ` ${kind}` : '');
  }

  function note(text, kind) {
    const el = $('narr-source-note');
    el.textContent = text || '';
    el.className = 'sync-loadresult' + (kind ? ` is-${kind}` : '');
  }

  function fmtDur(sec) {
    const s = Math.max(0, Math.round(Number(sec) || 0));
    const m = Math.floor(s / 60);
    return `${m}:${String(s % 60).padStart(2, '0')}`;
  }

  function renderVoices() {
    const select = $('narr-voice');
    const clone = $('narr-clone').checked;
    const list = state.voices.filter((v) => (clone ? v.clone : !v.clone));
    select.innerHTML = list.map((v) => (
      `<option value="${esc(v.vid)}">${esc(v.label)} — ${v.gender === 'F' ? 'nő' : 'férfi'} (${esc(v.engine)})</option>`
    )).join('');
    if (!clone) {
      const pref = list.find((v) => v.vid === 'SUP_M1') || list[0];
      if (pref) select.value = pref.vid;
    }
    $('narr-voice-info').textContent = clone
      ? 'klónhang a mintából'
      : `${list.length} kész hang`;
  }

  function syncClone() {
    const on = $('narr-clone').checked;
    $('narr-clonebox').hidden = !on;
    $('narr-clonebox').style.display = on ? 'flex' : 'none';
    $('narr-clone-hint').hidden = on;
    if (on && !state.cloneAvailable) {
      note('Ezen a gépen nincs XTTS klónmodell — a klón most nem választható.', 'err');
      $('narr-clone').checked = false;
      $('narr-clonebox').hidden = true;
      $('narr-clonebox').style.display = 'none';
      return;
    }
    renderVoices();
  }

  async function pick(kind, targetId) {
    try {
      const data = await Auris.api('/api/narration/pick', {
        method: 'POST', body: { kind },
      });
      if (data.path) {
        $(targetId).value = data.path;
        note(`Kiválasztva: ${data.path}`);
      }
    } catch (err) {
      note(err.message, 'err');
    }
  }

  async function upload(kind, inputId, targetId) {
    const input = $(inputId);
    const file = input.files && input.files[0];
    if (!file) return;
    note(`Feltöltés: ${file.name}…`);
    const form = new FormData();
    form.append('kind', kind);
    form.append('file', file);
    try {
      const data = await Auris.api('/api/narration/upload', { method: 'POST', body: form });
      $(targetId).value = data.path;
      note(`Feltöltve: ${data.name}`);
      if (kind === 'audio') {
        const audio = $('narr-ref-audio');
        if (audio) { audio.src = `/api/narration/file?path=${encodeURIComponent(data.path)}`; audio.hidden = false; }
      }
    } catch (err) {
      note(err.message, 'err');
    } finally {
      input.value = '';
    }
  }

  function sourceValue() {
    const url = $('narr-url').value.trim();
    if (url) return { source: url, youtube: true };
    return { source: $('narr-path').value.trim(), youtube: false };
  }

  async function start() {
    const picked = sourceValue();
    if (!picked.source) {
      note('Adj meg YouTube URL-t vagy válassz videófájlt.', 'err');
      return;
    }
    const clone = $('narr-clone').checked;
    const body = {
      source: picked.source,
      voice: $('narr-voice').value,
      ref: clone ? $('narr-ref').value.trim() : '',
      subtitle: $('narr-srt').value.trim(),
      language: 'hu',
      keep_original: $('narr-keep').checked,
      limit: $('narr-limit-on').checked ? Number($('narr-limit').value || 0) : 0,
    };
    if (clone && !body.ref) {
      note('Klónhanghoz add meg a hangmintát.', 'err');
      return;
    }
    $('narr-start').disabled = true;
    note('Indítás…');
    try {
      const started = await Auris.api('/api/narration/start', { method: 'POST', body });
      state.running = true;
      if (started && started.reused) {
        note('Korábbi munkamappa újrahasznosítva — a kész blokkok a gyorsítótárból jönnek.', 'ok');
      }
      $('narr-stop').hidden = false;
      $('narr-result').hidden = true;
      badge('fut', 'busy');
      poll(true);
    } catch (err) {
      $('narr-start').disabled = false;
      note(err.message, 'err');
    }
  }

  function showReport(report, payload) {
    const mix = report.mix || {};
    const blocks = report.blocks || [];
    const cloned = Boolean(report.clone_ref);
    const limited = Boolean(report.limited);
    const total = Number(report.blocks_total || blocks.length) || blocks.length;
    const rows = [
      ['Hossz', fmtDur(mix.seconds)],
      ['Hangosság', `${Number(mix.lufs || 0).toFixed(2)} LUFS`],
      ['Csúcs', `${Number(mix.peak_dbfs || 0).toFixed(2)} dBFS`],
      ['Blokkok', limited ? `${blocks.length}/${total}` : String(blocks.length)],
      ['Hang', cloned ? 'klónhang' : (report.voice || '')],
    ];
    $('narr-stats').innerHTML = rows.map(([label, value]) => (
      `<div class="narr-stat"><b>${esc(value)}</b><span>${esc(label)}</span></div>`
    )).join('');
    if (payload.audio_url) {
      $('narr-audio').src = payload.audio_url;
    }
    const dl = $('narr-download');
    if (payload.output_url) {
      dl.href = payload.output_url + '&download=1';
      dl.hidden = false;
    } else {
      dl.hidden = true;
    }
    const al = $('narr-audio-link');
    if (payload.audio_url) {
      al.href = payload.audio_url + '&download=1';
      al.hidden = false;
    } else {
      al.hidden = true;
    }
    const limitNote = $('narr-limit-note');
    const stateEl = $('narr-result-state');
    if (limited) {
      const cut = (report.trimmed && report.trimmed_to)
        ? `, a videó ${fmtDur(mix.seconds)}-ig vágva` : '';
      if (limitNote) {
        limitNote.innerHTML = '<span class="warn">Próbafutás: '
          + `${blocks.length}/${total} blokk${cut} — a teljes változathoz `
          + 'indítsd „Csak próba” nélkül.</span>';
        limitNote.hidden = false;
      }
      if (stateEl) stateEl.textContent = 'próba';
    } else {
      if (limitNote) {
        limitNote.hidden = true;
        limitNote.textContent = '';
      }
      if (stateEl) stateEl.textContent = 'kész';
    }
    $('narr-blocks').innerHTML = blocks.map((b) => (
      `<div class="sync-librow"><div class="sync-libmain"><strong>#${b.i}</strong> `
      + `<span class="sync-configinfo">${fmtDur(b.start)}–${fmtDur(b.end)}</span>`
      + `<p>${esc(b.text)}</p></div></div>`
    )).join('');
    $('narr-result').hidden = false;
  }

  async function poll(force) {
    if (state.timer) clearTimeout(state.timer);
    try {
      const data = await Auris.api('/api/narration/status');
      if (data.log_tail) {
        const log = $('narr-log');
        log.textContent = data.log_tail;
        log.scrollTop = log.scrollHeight;
      }
      if (data.report) showReport(data.report, data);
      if (data.state === 'running') {
        state.running = true;
        badge('fut', 'busy');
        $('narr-stop').hidden = false;
        $('narr-start').disabled = true;
        $('narr-run-info').textContent = data.started_at ? `indult: ${data.started_at}` : '';
        state.timer = setTimeout(poll, 2000);
        return;
      }
      state.running = false;
      $('narr-stop').hidden = true;
      $('narr-start').disabled = false;
      if (data.state === 'done') {
        badge('kész', 'ok');
        $('narr-run-info').textContent = data.finished_at ? `kész: ${data.finished_at}` : '';
        if (!data.report) note('A futás lezárult, de a riport nem található.', 'err');
      } else if (data.state === 'failed') {
        badge('hiba', 'err');
        note('A narráció hibával állt le — nézd meg a naplót.', 'err');
      } else if (data.state === 'stopped') {
        badge('leállítva');
      }
      if (force && data.state === 'idle') badge('készen áll');
    } catch (err) {
      note(err.message, 'err');
    }
  }

  async function stop() {
    try {
      await Auris.api('/api/narration/stop', { method: 'POST', body: {} });
      badge('leállítva');
      $('narr-stop').hidden = true;
      $('narr-start').disabled = false;
    } catch (err) {
      note(err.message, 'err');
    }
  }

  async function loadVoices() {
    try {
      const data = await Auris.api('/api/narration/voices');
      state.voices = data.voices || [];
      state.cloneAvailable = Boolean(data.clone_available);
      if (!state.cloneAvailable) {
        $('narr-clone').disabled = true;
        $('narr-clone-hint').textContent =
          'Ezen a gépen nincs XTTS klónmodell, ezért csak kész hang választható.';
      }
      if (data.clone_ref && !$('narr-ref').value) $('narr-ref').value = data.clone_ref;
      renderVoices();
    } catch (err) {
      note(err.message, 'err');
    }
  }

  function init() {
    $('narr-pick').addEventListener('click', () => pick('video', 'narr-path'));
    $('narr-srt-pick').addEventListener('click', () => pick('srt', 'narr-srt'));
    $('narr-ref-pick').addEventListener('click', () => pick('audio', 'narr-ref'));
    $('narr-upload-btn').addEventListener('click', () => $('narr-upload').click());
    $('narr-srt-upload-btn').addEventListener('click', () => $('narr-srt-upload').click());
    $('narr-ref-upload-btn').addEventListener('click', () => $('narr-ref-upload').click());
    $('narr-upload').addEventListener('change', () => upload('video', 'narr-upload', 'narr-path'));
    $('narr-srt-upload').addEventListener('change', () => upload('srt', 'narr-srt-upload', 'narr-srt'));
    $('narr-ref-upload').addEventListener('change', () => upload('audio', 'narr-ref-upload', 'narr-ref'));
    $('narr-clone').addEventListener('change', syncClone);
    $('narr-limit-on').addEventListener('change', () => {
      $('narr-limit-wrap').hidden = !$('narr-limit-on').checked;
      $('narr-limit-wrap').style.display =
        $('narr-limit-on').checked ? 'flex' : 'none';
    });
    $('narr-start').addEventListener('click', start);
    $('narr-stop').addEventListener('click', stop);
    $('narr-url').addEventListener('input', () => {
      if ($('narr-url').value.trim()) note('YouTube URL alapján dolgozik; a videófájl mezőt nem használja.');
    });
    $('narr-path').addEventListener('input', () => {
      if ($('narr-path').value.trim()) $('narr-url').value = '';
    });
    $('narr-clonebox').style.display = 'none';
    $('narr-limit-wrap').style.display = 'none';
    loadVoices();
    poll(true);
  }

  document.addEventListener('DOMContentLoaded', init);
})();
