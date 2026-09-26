// The two v16 figures, using its recorded data and endpoint-preserving smoothing.
// Native SVG keeps the plots sharp; no chart library, fonts, or animation runtime.
const NS = 'http://www.w3.org/2000/svg';
const COLORS = ['#1d2b2e', '#ef7758', '#d3a13f', '#008c77', '#739aac', '#8981bc'];
const LABELS = ['Fixed λ', 'α = 0', 'α = 0.25', 'α = 0.5', 'α = 0.75', 'α = 1'];
const ORDER = [1, 2, 3, 4, 5, 0]; // Keep fixed λ visible where starts overlap.
const lerp = (a, b, t) => a + (b - a) * t;
const point = p => `${p[0].toFixed(2)},${p[1].toFixed(2)}`;
const pathD = points => points.length ? `M${points.map(point).join('L')}` : '';
const pct = value => `${value < 0 ? '−' : value > 0 ? '+' : ''}${Math.abs(value).toFixed(1)}%`;

function svgNode(tag, attrs, parent) {
  const node = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attrs || {})) node.setAttribute(key, value);
  if (parent) parent.append(node);
  return node;
}

function label(parent, value, x, y, anchor = 'start', className = '') {
  const node = svgNode('text', { x, y, 'text-anchor': anchor, class: className }, parent);
  node.textContent = value;
  return node;
}

function line(parent, x1, y1, x2, y2, attrs = {}) {
  return svgNode('line', { x1, y1, x2, y2, stroke: '#e9edeb', ...attrs }, parent);
}

function marker(parent, effort, color, hollow = false) {
  const d = ['M4.4,0 A4.4,4.4 0 1,0 -4.4,0 A4.4,4.4 0 1,0 4.4,0',
    'M-4,-4 H4 V4 H-4 Z', 'M0,-5.2 L4.8,3.5 H-4.8 Z'][effort];
  return svgNode('path', { d, fill: hollow ? 'white' : color,
    stroke: hollow ? color : 'white', 'stroke-width': hollow ? 1.3 : .8 }, parent);
}

function setupChart(svg, clipId, compactRelative = false) {
  const width = svg.clientWidth;
  const height = compactRelative
    ? Math.min(500, Math.max(220, (width - 54) * 124 / 188 + 76))
    : Math.min(500, Math.max(300, width * .84));
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.style.height = `${height}px`;
  const description = svg.querySelector('desc');
  svg.replaceChildren(description);
  const defs = svgNode('defs', {}, svg);
  const clip = svgNode('clipPath', { id: clipId }, defs);
  const bounds = { left: 42, right: width - 12, top: 32, bottom: height - 44 };
  const clipRect = svgNode('rect', { x: bounds.left - 6, y: bounds.top - 6,
    width: bounds.right - bounds.left + 12, height: bounds.bottom - bounds.top + 12 }, clip);
  const axes = svgNode('g', { 'aria-hidden': 'true' }, svg);
  const marks = svgNode('g', { 'aria-hidden': 'true', 'clip-path': `url(#${clipId})` }, svg);
  return { width, height, defs, bounds, axes, marks, clipRect };
}

// Identical to smoothThread in v16: centered average of ±5 saved checkpoints,
// followed by a linear correction that preserves both recorded endpoints.
function smoothThread(raw) {
  const n = raw.length;
  const smoothed = raw.map((_, k) => {
    const lo = Math.max(0, k - 5), hi = Math.min(n - 1, k + 5);
    let x = 0, y = 0;
    for (let j = lo; j <= hi; j++) { x += raw[j][0]; y += raw[j][1]; }
    return [x / (hi - lo + 1), y / (hi - lo + 1)];
  });
  const first = raw[0].map((v, j) => v - smoothed[0][j]);
  const last = raw[n - 1].map((v, j) => v - smoothed[n - 1][j]);
  return smoothed.map((p, k) => p.map((v, j) => v + lerp(first[j], last[j], k / (n - 1))));
}

function distanceSquared(p, a, b) {
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const length = dx * dx + dy * dy;
  const t = length ? Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length)) : 0;
  return (p[0] - a[0] - t * dx) ** 2 + (p[1] - a[1] - t * dy) ** 2;
}

function mount(root, data) {
  const training = root.querySelector('[data-training]');
  const pathsSvg = root.querySelector('[data-paths]');
  const interaction = root.querySelector('[data-paths-interaction]');
  const readout = root.querySelector('[data-path-readout]');
  const play = root.querySelector('[data-play]');
  const slider = root.querySelector('[data-step]');
  const stepLabel = root.querySelector('[data-step-label]');
  const finalStep = data.steps.at(-1);
  let step = finalStep, playing = false, animation = 0, lastTime = 0;
  let curves = [], connectors = [], selected = null, pinned = false, highlight, endpoint;
  let pathRecords = [], cells = new Map(), plotBounds, origin;
  let pointerFrame = 0, pendingPointer, downPointer;
  const CELL = 24, HIT_RADIUS = 14;
  const smoothed = data.medium_threads.map(method => method.map(smoothThread));

  function drawTraining() {
    const chart = setupChart(training, 'swe-film-training-clip');
    const { left: L, right: R, top: T, bottom: B } = chart.bounds;
    const X = x => L + x / 16.5 * (R - L);
    const Y = y => B - (y - 18) / 60 * (B - T);
    for (let x = 0; x <= 16; x += chart.width < 420 ? 4 : 2) {
      line(chart.axes, X(x), T, X(x), B);
      label(chart.axes, x, X(x), B + 22, 'middle', 'swe-film-tick');
    }
    for (let y = 20; y <= 70; y += 10) {
      line(chart.axes, L, Y(y), R, Y(y));
      label(chart.axes, `${y}%`, L - 9, Y(y) + 4, 'end', 'swe-film-tick');
    }
    label(chart.axes, 'Success ↑', L, 17, 'start', 'swe-film-axis-label');
    label(chart.axes, 'Mean cost (attempts per task) →', R, chart.height - 3, 'end', 'swe-film-axis-label');
    const initial = data.mean_abs[0].map(effort => [X(effort[0][0]), Y(effort[0][1])]);
    svgNode('path', { d: pathD(initial), fill: 'none', stroke: '#9ba6a2',
      'stroke-width': 1.3, 'stroke-dasharray': '4 5' }, chart.marks);
    initial.forEach((p, e) => marker(chart.marks, e, '#879390', true).setAttribute('transform', `translate(${point(p)})`));
    curves = [];
    connectors = [];
    for (const method of ORDER) {
      const group = svgNode('g', {}, chart.marks);
      connectors[method] = svgNode('path', { fill: 'none', stroke: COLORS[method],
        'stroke-width': .9, opacity: .35 }, group);
      for (let effort = 0; effort < 3; effort++) {
        curves.push({ method, effort,
          points: data.mean_abs[method][effort].map(p => [X(p[0]), Y(p[1])]),
          path: svgNode('path', { fill: 'none', stroke: COLORS[method], 'stroke-width': method === 0 ? 2.1 : 1.9,
            'stroke-linejoin': 'round', 'stroke-linecap': 'round', 'data-mean-method': method, 'data-effort': effort }, group),
          head: marker(group, effort, COLORS[method]) });
      }
    }
    renderStep(step);
  }

  function renderStep(next) {
    step = Math.max(0, Math.min(finalStep, next));
    // Interpolate in actual training steps: checkpoint 0→1 is shorter than 10→20.
    let i = 0;
    while (i + 1 < data.steps.length && data.steps[i + 1] <= step) i++;
    const j = Math.min(i + 1, data.steps.length - 1);
    const fraction = i === j ? 0 : (step - data.steps[i]) / (data.steps[j] - data.steps[i]);
    const heads = Array.from({ length: 6 }, () => []);
    for (const curve of curves) {
      const a = curve.points[i], b = curve.points[j];
      const head = [lerp(a[0], b[0], fraction), lerp(a[1], b[1], fraction)];
      const trail = curve.points.slice(0, i + 1);
      if (fraction) trail.push(head);
      curve.path.setAttribute('d', pathD(trail));
      curve.head.setAttribute('transform', `translate(${point(head)})`);
      heads[curve.method][curve.effort] = head;
    }
    connectors.forEach((node, m) => node.setAttribute('d', pathD(heads[m])));
    const shown = Math.round(step);
    slider.value = shown;
    stepLabel.textContent = shown.toLocaleString('en-US');
    slider.setAttribute('aria-valuetext', `Training step ${shown.toLocaleString('en-US')} of 1,000`);
  }

  function updatePlayButton() {
    root.querySelector('[data-play-label]').textContent = playing ? 'Pause' : 'Play';
    root.querySelector('[data-play-icon]').textContent = playing ? 'Ⅱ' : '▶';
    play.setAttribute('aria-label', playing ? 'Pause training' : step >= finalStep ? 'Play training from step zero' : 'Play training');
  }

  function pause() {
    playing = false;
    cancelAnimationFrame(animation);
    updatePlayButton();
  }

  function tick(time) {
    if (!playing) return;
    renderStep(step + Math.min(time - lastTime, 100) * finalStep / 10000);
    lastTime = time;
    if (step >= finalStep) pause();
    else animation = requestAnimationFrame(tick);
  }

  play.addEventListener('click', () => {
    if (playing) { pause(); return; }
    if (step >= finalStep) renderStep(0);
    playing = true;
    lastTime = performance.now();
    updatePlayButton();
    animation = requestAnimationFrame(tick);
  });
  slider.addEventListener('input', () => { pause(); renderStep(Number(slider.value)); updatePlayButton(); });
  document.addEventListener('visibilitychange', () => { if (document.hidden) pause(); });
  if ('IntersectionObserver' in window) {
    new IntersectionObserver(entries => { if (!entries[0].isIntersecting) pause(); }).observe(training);
  }

  function drawPaths() {
    const chart = setupChart(pathsSvg, 'swe-film-paths-clip', window.matchMedia('(max-width: 900px)').matches);
    const { left: L, right: R, top: T, bottom: B } = chart.bounds;
    // v16's expanded trajectory view: x = [-84, 104], y = [-12, 112].
    // Equal scales preserve the direction of improvement, just as in the film.
    const scale = Math.min((R - L) / 188, (B - T) / 124);
    const left = L + (R - L - 188 * scale) / 2;
    const top = T + (B - T - 124 * scale) / 2;
    origin = [left + 84 * scale, top + 112 * scale];
    const X = x => origin[0] + x * scale, Y = y => origin[1] - y * scale;
    plotBounds = { left, right: left + 188 * scale, top, bottom: top + 124 * scale };
    for (const [key, value] of Object.entries({ x: left, y: top, width: 188 * scale, height: 124 * scale })) chart.clipRect.setAttribute(key, value);
    line(chart.axes, X(-84), Y(0), X(104), Y(0), { stroke: '#cdd5d1' });
    line(chart.axes, X(0), Y(-12), X(0), Y(112), { stroke: '#cdd5d1' });
    const increment = chart.width < 420 ? 40 : 20;
    for (let x = -80; x <= 80; x += increment) {
      if (x === 0) continue;
      line(chart.axes, X(x), Y(0) - 4, X(x), Y(0) + 4, { stroke: '#cdd5d1' });
      label(chart.axes, `${x > 0 ? '+' : '−'}${Math.abs(x)}%`, X(x), Y(0) + 19, 'middle', 'swe-film-tick');
    }
    for (let y = 20; y <= 100; y += 20) {
      line(chart.axes, X(0) - 4, Y(y), X(0) + 4, Y(y), { stroke: '#cdd5d1' });
      label(chart.axes, `+${y}%`, X(0) - 8, Y(y) + 4, 'end', 'swe-film-tick');
    }
    label(chart.axes, 'Success change ↑', chart.width < 300 ? R : X(0) + 7, top - 12,
      chart.width < 300 ? 'end' : 'start', 'swe-film-axis-label');
    label(chart.axes, 'Cost change →', X(104), Math.max(Y(0) + 42, B + 25), 'end', 'swe-film-axis-label');
    pathRecords = [];
    cells = new Map();
    for (const method of ORDER) {
      const id = `swe-film-gradient-${method}`;
      const gradient = svgNode('radialGradient', { id, gradientUnits: 'userSpaceOnUse',
        cx: origin[0], cy: origin[1], r: 110 * scale / 3.6 }, chart.defs);
      [[0, 0], [.25, .12], [1, 1]].forEach(([offset, opacity]) => svgNode('stop', {
        offset, 'stop-color': COLORS[method], 'stop-opacity': opacity }, gradient));
      smoothed[method].forEach((raw, problem) => {
        const points = raw.map(p => [X(p[0]), Y(p[1])]);
        const record = { method, problem, points, d: pathD(points) };
        record.node = svgNode('path', { d: record.d, fill: 'none', stroke: `url(#${id})`,
          'stroke-width': method === 0 ? .8 : 1, opacity: method === 0 ? .16 : .5,
          'stroke-linecap': 'round', 'stroke-linejoin': 'round',
          'data-path-method': method, 'data-problem': problem }, chart.marks);
        pathRecords.push(record);
        // Index line segments by screen cell so the closest visible line wins,
        // including in dense bundles. DOM paint order does not choose the hit.
        for (let i = 1; i < points.length; i++) {
          const a = points[i - 1], b = points[i];
          const minX = Math.max(left, Math.min(a[0], b[0]) - HIT_RADIUS);
          const maxX = Math.min(plotBounds.right, Math.max(a[0], b[0]) + HIT_RADIUS);
          const minY = Math.max(top, Math.min(a[1], b[1]) - HIT_RADIUS);
          const maxY = Math.min(plotBounds.bottom, Math.max(a[1], b[1]) + HIT_RADIUS);
          if (minX > maxX || minY > maxY) continue;
          const segment = { record, a, b };
          for (let cx = Math.floor(minX / CELL); cx <= Math.floor(maxX / CELL); cx++) {
            for (let cy = Math.floor(minY / CELL); cy <= Math.floor(maxY / CELL); cy++) {
              const key = `${cx}:${cy}`;
              if (!cells.has(key)) cells.set(key, []);
              cells.get(key).push(segment);
            }
          }
        }
      });
    }
    highlight = svgNode('path', { fill: 'none', 'stroke-width': 2.4,
      'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'data-highlight': '', visibility: 'hidden' }, chart.marks);
    endpoint = svgNode('circle', { r: 3.5, stroke: 'white', 'stroke-width': 1, visibility: 'hidden' }, chart.marks);
    svgNode('circle', { cx: origin[0], cy: origin[1], r: 4, fill: COLORS[0], stroke: 'white', 'stroke-width': 1.5 }, chart.marks);
    if (selected) selectPath(pathRecords.find(p => p.method === selected.method && p.problem === selected.problem));
  }

  function selectPath(record) {
    selected = record;
    if (!record) {
      highlight.setAttribute('visibility', 'hidden');
      endpoint.setAttribute('visibility', 'hidden');
      const hint = document.createElement('span');
      hint.className = 'swe-film-hint';
      hint.textContent = 'Hover or tap a line to inspect a problem.';
      readout.replaceChildren(hint);
      return;
    }
    highlight.setAttribute('d', record.d);
    highlight.setAttribute('stroke', COLORS[record.method]);
    highlight.setAttribute('visibility', 'visible');
    highlight.dataset.method = record.method;
    highlight.dataset.problem = record.problem;
    const last = record.points.at(-1);
    endpoint.setAttribute('cx', last[0]);
    endpoint.setAttribute('cy', last[1]);
    endpoint.setAttribute('fill', COLORS[record.method]);
    endpoint.setAttribute('visibility', 'visible');
    const title = document.createElement('strong');
    title.textContent = `${LABELS[record.method]} · Problem ${String(record.problem + 1).padStart(2, '0')}`;
    title.title = data.tasks[record.problem];
    const values = document.createElement('span');
    const final = data.medium_threads[record.method][record.problem].at(-1);
    values.textContent = `Cost ${pct(final[0])} · Success ${pct(final[1])}`;
    readout.style.setProperty('--method-color', COLORS[record.method]);
    readout.replaceChildren(title, values);
  }

  function localPoint(event) {
    const rect = pathsSvg.getBoundingClientRect();
    const box = pathsSvg.viewBox.baseVal;
    return [(event.clientX - rect.left) * box.width / rect.width,
      (event.clientY - rect.top) * box.height / rect.height];
  }

  function closest(event) {
    const p = localPoint(event);
    if (p[0] < plotBounds.left || p[0] > plotBounds.right || p[1] < plotBounds.top || p[1] > plotBounds.bottom) return null;
    // The shared origin cannot distinguish any of the 600 paths.
    if ((p[0] - origin[0]) ** 2 + (p[1] - origin[1]) ** 2 < 64) return null;
    const segments = cells.get(`${Math.floor(p[0] / CELL)}:${Math.floor(p[1] / CELL)}`) || [];
    let best = null, distance = event.pointerType === 'touch' ? HIT_RADIUS ** 2 : 7 ** 2;
    for (const segment of segments) {
      const d = distanceSquared(p, segment.a, segment.b);
      if (d < distance) { best = segment.record; distance = d; }
    }
    return best;
  }

  interaction.addEventListener('pointermove', event => {
    if (pinned || event.pointerType === 'touch') return;
    pendingPointer = event;
    if (pointerFrame) return;
    pointerFrame = requestAnimationFrame(() => {
      pointerFrame = 0;
      const record = closest(pendingPointer);
      if (record !== selected) selectPath(record);
    });
  });
  interaction.addEventListener('pointerleave', () => {
    cancelAnimationFrame(pointerFrame);
    pointerFrame = 0;
    if (!pinned && selected) selectPath(null);
  });
  interaction.addEventListener('pointerdown', event => {
    downPointer = { x: event.clientX, y: event.clientY, time: performance.now() };
  });
  interaction.addEventListener('pointercancel', () => { downPointer = null; });
  interaction.addEventListener('pointerup', event => {
    if (!downPointer) return;
    const tapped = Math.hypot(event.clientX - downPointer.x, event.clientY - downPointer.y) < 10 && performance.now() - downPointer.time < 600;
    downPointer = null;
    if (!tapped) return;
    cancelAnimationFrame(pointerFrame);
    pointerFrame = 0;
    const record = closest(event);
    if (pinned && record === selected) { pinned = false; selectPath(null); }
    else { pinned = Boolean(record); selectPath(record); }
  });
  interaction.addEventListener('keydown', event => {
    if (event.key === 'Escape') { pinned = false; selectPath(null); return; }
    if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    let method = selected?.method ?? 3, problem = selected?.problem ?? 0;
    if (event.key === 'ArrowLeft') problem = (problem + data.tasks.length - 1) % data.tasks.length;
    if (event.key === 'ArrowRight') problem = (problem + 1) % data.tasks.length;
    if (event.key === 'ArrowUp') method = (method + 1) % LABELS.length;
    if (event.key === 'ArrowDown') method = (method + LABELS.length - 1) % LABELS.length;
    if (event.key === 'Home') problem = 0;
    if (event.key === 'End') problem = data.tasks.length - 1;
    pinned = true;
    selectPath(pathRecords.find(p => p.method === method && p.problem === problem));
  });

  let previousWidth = 0;
  function resize() {
    const width = training.clientWidth;
    if (width < 1 || Math.abs(width - previousWidth) < 1) return;
    previousWidth = width;
    drawTraining();
    drawPaths();
    root.dataset.ready = 'true';
  }
  resize();
  new ResizeObserver(resize).observe(training.parentElement);
}

for (const root of document.querySelectorAll('[data-swe-film-figures]')) {
  const status = root.querySelector('[data-loading]');
  const figures = root.querySelector('[data-figures]');
  let loading = false;
  async function load() {
    if (loading) return;
    loading = true;
    status.hidden = false;
    status.textContent = 'Loading the figures…';
    try {
      const response = await fetch(root.dataset.source);
      if (!response.ok) throw new Error(`Figure data returned ${response.status}`);
      const data = await response.json();
      figures.hidden = false;
      mount(root, data);
      status.hidden = true;
    } catch (error) {
      loading = false;
      figures.hidden = true;
      status.textContent = 'The figures couldn’t load.';
      const retry = document.createElement('button');
      retry.type = 'button';
      retry.textContent = 'Try again';
      retry.addEventListener('click', load, { once: true });
      status.append(retry);
      console.error(error);
    }
  }
  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver(entries => {
      if (!entries.some(entry => entry.isIntersecting)) return;
      observer.disconnect();
      load();
    }, { rootMargin: '500px' });
    observer.observe(root);
  } else load();
}
