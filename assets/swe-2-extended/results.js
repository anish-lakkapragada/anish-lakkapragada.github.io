// Recorded results only. No client-side training or interpolation between alphas.
const root = document.querySelector('#swe-results');
const $ = (selector) => root.querySelector(selector);
const $$ = (selector) => [...root.querySelectorAll(selector)];
const ALPHAS = [0, 0.25, 0.5, 0.75, 1];
// Dark enough for small labels and white button text as well as large marks.
const COLORS = ['#00796c', '#627700', '#3861e8', '#8740c2', '#bd4221'];
const FIXED = '#34333e';
const INITIAL = '#9a9ba8';
const EFFORTS = ['low', 'medium', 'high'];
const NAMES = ['Save cost', 'Favor cost savings', 'Balanced gains', 'Favor success', 'Improve success'];
const DESCRIPTIONS = [
  'Target cost savings while preserving initial success. Finite training can leave a small success gain or loss.',
  'Target three parts relative cost reduction for every one part relative success gain.',
  'Target equal relative improvements in cost and success.',
  'Target three parts relative success gain for every one part relative cost reduction.',
  'Target success gains while preserving the initial cost. Individual runs can finish above or below that budget.',
];
const state = { alpha: 2, effort: 1, problem: 44, seed: 0, step: 51, dots: false };
let data;
let replay;
let replayRequest = 0;
let playback = null;
const replayCache = new Map();
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const fmt = (v, digits = 1) => Number(v).toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits });
const signed = (v, digits = 1) => `${v < -0.5 * 10 ** -digits ? '−' : '+'}${fmt(Math.abs(v), digits)}`;
const alpha = () => ALPHAS[state.alpha];
const method = () => state.alpha + 1;
const color = () => COLORS[state.alpha];
const problem = () => data.problems[state.problem];
const selected = () => problem().results[method()];
const text = (selector, value) => { $(selector).textContent = value; };
const balance = (p) => (1 - alpha()) * p.results[method()].gain[state.effort] - alpha() * p.results[method()].saving[state.effort];
const scrollTo = (element) => element.scrollIntoView({ behavior: reducedMotion.matches ? 'instant' : 'smooth', block: 'start' });

function node(tag, attrs = {}, content) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value);
  if (content !== undefined) el.textContent = content;
  return el;
}

function limits(values, includeZero = true) {
  const min = includeZero ? Math.min(0, ...values) : Math.min(...values);
  const max = includeZero ? Math.max(0, ...values) : Math.max(...values);
  const span = max - min || Math.max(Math.abs(max) * 0.1, 0.01);
  return [min - span * 0.08, max + span * 0.16];
}

function ticks(min, max, count = 4) {
  const raw = (max - min) / count;
  const power = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].find((v) => v * power >= raw) * power;
  const values = [];
  for (let n = Math.ceil(min / step) * step; n <= max + step * 0.001; n += step) values.push(Math.abs(n) < step * 0.001 ? 0 : n);
  return values;
}

function chart(name, { xDomain, yDomain, xLabel, yLabel, height = 300 }) {
  const svg = $(`[data-chart="${name}"]`);
  const w = Math.max(200, svg.getBoundingClientRect().width);
  const h = height;
  const margin = { l: 45, r: 18, t: 33, b: 44 };
  svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
  svg.setAttribute('height', h);
  svg.replaceChildren();
  const title = node('title', {}, `${yLabel} by ${xLabel}`);
  svg.append(title);
  const x = (v) => margin.l + (v - xDomain[0]) / (xDomain[1] - xDomain[0]) * (w - margin.l - margin.r);
  const y = (v) => h - margin.b - (v - yDomain[0]) / (yDomain[1] - yDomain[0]) * (h - margin.t - margin.b);
  const add = (tag, attrs, content) => { const n = node(tag, attrs, content); svg.append(n); return n; };
  const label = (px, py, content, attrs = {}) => add('text', { x: px, y: py, fill: '#737381', 'font-size': 10, ...attrs }, content);
  for (const t of ticks(...yDomain)) {
    add('line', { x1: margin.l, x2: w - margin.r, y1: y(t), y2: y(t), stroke: t === 0 ? '#c3c3d1' : '#e7e7ef', 'stroke-dasharray': t === 0 ? '3 3' : 'none' });
    const digits = yDomain[1] < 0.1 ? 3 : yDomain[1] < 1 ? 2 : Number.isInteger(t) ? 0 : 1;
    label(margin.l - 8, y(t) + 3, fmt(t, digits), { 'text-anchor': 'end' });
  }
  for (const t of ticks(...xDomain, w < 350 ? 3 : 5)) {
    if (t === 0) add('line', { x1: x(t), x2: x(t), y1: margin.t, y2: h - margin.b, stroke: '#c3c3d1', 'stroke-dasharray': '3 3' });
    label(x(t), h - margin.b + 18, fmt(t, Number.isInteger(t) ? 0 : 1), { 'text-anchor': 'middle' });
  }
  label(margin.l, 14, yLabel, { fill: '#535460', 'font-size': 10.5 });
  label((margin.l + w - margin.r) / 2, h - 4, xLabel, { fill: '#535460', 'font-size': 10.5, 'text-anchor': 'middle' });
  const path = (points, attrs = {}) => add('path', { d: points.map(([a,b], i) => `${i ? 'L' : 'M'}${x(a).toFixed(2)},${y(b).toFixed(2)}`).join(' '), fill: 'none', stroke: color(), 'stroke-width': 2.5, 'stroke-linejoin': 'round', 'stroke-linecap': 'round', ...attrs });
  const dot = (a, b, attrs = {}) => add('circle', { cx: x(a), cy: y(b), r: 5, fill: color(), stroke: '#fff', 'stroke-width': 1.5, ...attrs });
  return { svg, x, y, w, h, margin, add, label, path, dot };
}

function syncControls() {
  root.style.setProperty('--swe-accent', color());
  root.style.setProperty('--swe-tint', `${color()}10`);
  $$('#swe-alpha').forEach((el) => { el.value = state.alpha; });
  $('#swe-alpha').setAttribute('aria-valuetext', `Alpha ${alpha()}: ${NAMES[state.alpha]}`);
  text('[data-alpha-label]', `α = ${alpha()}`);
  $$('[data-alpha-index]').forEach((button) => button.setAttribute('aria-pressed', String(+button.dataset.alphaIndex === state.alpha)));
  $$('[data-alpha-select]').forEach((el) => { el.value = state.alpha; });
  $$('[data-effort]').forEach((el) => { el.value = state.effort; });
  $$('[data-problem]').forEach((el) => { el.value = state.problem; });
  const seeds = $('[data-seed]');
  if (seeds.dataset.problem !== problem().id) {
    seeds.replaceChildren(...problem().seeds.map((seed, i) => new Option(String(seed), String(i))));
    seeds.dataset.problem = problem().id;
  }
  seeds.value = state.seed;
}

function renderDirection() {
  const e = state.effort;
  const selectedCurve = data.curves[method()];
  const end = selectedCurve.cost.length - 1;
  const saving = selectedCurve.saving[end][e];
  const gain = selectedCurve.gain[end][e];
  text('[data-saving]', `${fmt(Math.abs(saving))}%`);
  text('[data-saving-label]', saving >= 0 ? 'less cost' : 'more cost');
  text('[data-gain]', `${signed(gain)}%`);
  text('[data-direction-name]', NAMES[state.alpha]);
  text('[data-direction-description]', DESCRIPTIONS[state.alpha]);
  text('[data-absolute-cost]', `${fmt(selectedCurve.cost[end][e], 2)} attempts`);
  text('[data-absolute-success]', `${fmt(selectedCurve.success[end][e], 2)}%`);
  const fixed = data.curves[0];
  const fixedSaving = fixed.saving[end][e];
  text('[data-fixed-comparison]', `${fmt(Math.abs(fixedSaving))}% ${fixedSaving >= 0 ? 'less' : 'more'} cost; ${signed(fixed.gain[end][e])}% relative success. Mean cost ${fmt(fixed.cost[end][e], 2)}, success ${fmt(fixed.success[end][e], 2)}%.`);
  const allX = data.curves.flatMap((c) => c.saving.map((row) => row[e]));
  const allY = data.curves.flatMap((c) => c.gain.map((row) => row[e]));
  if (state.dots) for (const p of data.problems) { allX.push(p.results[method()].saving[e]); allY.push(p.results[method()].gain[e]); }
  const xDomain = limits(allX);
  const yDomain = limits(allY);
  const c = chart('direction', { xDomain, yDomain, xLabel: 'Cost reduction (%) →', yLabel: 'Relative success gain (%) ↑', height: window.innerWidth < 700 ? 310 : 370 });
  c.svg.setAttribute('role', state.dots ? 'group' : 'img');
  const rayLength = Math.min(alpha() < 1 ? xDomain[1] / (1 - alpha()) : Infinity, alpha() > 0 ? yDomain[1] / alpha() : Infinity) * .8;
  const ray = [(1-alpha())*rayLength, alpha()*rayLength];
  c.path([[0,0], ray], { stroke: '#a2a2b1', 'stroke-width': 1.5, 'stroke-dasharray': '5 5' });
  const angle = Math.atan2(c.y(ray[1])-c.y(0), c.x(ray[0])-c.x(0));
  c.add('path', { d: `M${c.x(ray[0])-8*Math.cos(angle-.45)},${c.y(ray[1])-8*Math.sin(angle-.45)} L${c.x(ray[0])},${c.y(ray[1])} L${c.x(ray[0])-8*Math.cos(angle+.45)},${c.y(ray[1])-8*Math.sin(angle+.45)}`, fill:'none', stroke: '#a2a2b1', 'stroke-width': 1.5 });
  for (let m = 1; m < data.curves.length; m++) {
    if (m === method()) continue;
    const series = data.curves[m];
    c.path(series.saving.map((row,i) => [row[e], series.gain[i][e]]), { stroke: COLORS[m-1], opacity: .2, 'stroke-width': 1.5 });
    c.dot(series.saving[end][e], series.gain[end][e], { fill: COLORS[m-1], opacity: .45, r: 3 });
  }
  if (state.dots) data.problems.forEach((p,i) => {
    const r = p.results[method()];
    const dot = c.dot(r.saving[e], r.gain[e], { r: 4, opacity: .45, class: 'swe-point', tabindex: '0', role:'button', 'aria-label': `Inspect problem ${i+1}: ${fmt(r.saving[e])}% cost reduction, ${signed(r.gain[e])}% success gain` });
    dot.append(node('title', {}, `Problem ${i+1} · p_good ${fmt(p.pg,3)}, ratio ${fmt(p.ratio,3)}`));
    const inspect = () => { setProblem(i); scrollTo($('.swe-atlas')); };
    dot.addEventListener('click', inspect);
    dot.addEventListener('keydown', (ev) => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); inspect(); } });
  });
  c.path(fixed.saving.map((row,i) => [row[e], fixed.gain[i][e]]), { stroke: FIXED, 'stroke-width': 2.3 });
  c.path(selectedCurve.saving.map((row,i) => [row[e], selectedCurve.gain[i][e]]), { 'stroke-width': 3.5 });
  c.dot(0, 0, { fill: '#fff', stroke: INITIAL, r: 4.5, 'stroke-width': 2 });
  c.dot(fixed.saving[end][e], fixed.gain[end][e], { fill: FIXED, r: 5.5 });
  c.dot(saving, gain, { r: 7, 'stroke-width': 2.5 });
  c.label(c.x(saving) + 9, c.y(gain) - 10, `α = ${alpha()}`, { fill: color(), 'font-size': 11, 'font-weight': 700 });
  c.label(c.x(fixed.saving[end][e]) + 8, c.y(fixed.gain[end][e]) - 10, 'Fixed λ', { fill: FIXED, 'font-size': 10.5, 'font-weight': 700 });
  c.label(c.x(0) + 8, c.y(0) + 15, 'Start', { 'font-size': 10 });
}

function currentReplay() {
  return replay && replay.id === problem().id ? replay.seeds[state.seed] : null;
}

function renderReplay() {
  const current = currentReplay();
  if (!current) return;
  const run = current.runs[method()];
  const fixed = current.runs[0];
  const e = state.effort;
  const t = Math.min(state.step, replay.steps.length - 1);
  state.step = t;
  $('#swe-step').max = replay.steps.length - 1;
  $('#swe-step').value = t;
  $('#swe-step').setAttribute('aria-valuetext', `Training step ${replay.steps[t]}`);
  text('[data-step-label]', replay.steps[t].toLocaleString('en-US'));
  const xDomain = limits([...run.c.map((v) => v[e]), ...fixed.c.map((v) => v[e])], false);
  const yDomain = limits([...run.s.map((v) => v[e]), ...fixed.s.map((v) => v[e])], false);
  const c = chart('replay', { xDomain, yDomain, xLabel:'Mean cost (attempts)', yLabel:'Success rate (%)', height: 260 });
  for (const [r, ink] of [[fixed, FIXED], [run, color()]]) {
    c.path(r.c.map((v,i) => [v[e],r.s[i][e]]), { stroke: ink, opacity: .12, 'stroke-width': 2 });
    c.path(r.c.slice(0,t+1).map((v,i) => [v[e],r.s[i][e]]), { stroke: ink, 'stroke-width': 2.8 });
    c.dot(r.c[t][e], r.s[t][e], { fill: ink, r: 6 });
  }
  c.dot(run.c[0][e], run.s[0][e], { fill: '#fff', stroke: INITIAL, r: 4 });
  const penaltyDomain = limits([...run.l.map((v) => v[e]), fixed.l[0][e]], false);
  const p = chart('penalty', { xDomain:[0,1000], yDomain:penaltyDomain, xLabel:'Training step', yLabel:'λ', height:260 });
  p.path([[0,fixed.l[0][e]],[1000,fixed.l[0][e]]], { stroke: FIXED, 'stroke-width': 1.8, 'stroke-dasharray': '4 4' });
  p.path(run.l.map((v,i) => [replay.steps[i],v[e]]), { opacity: .15 });
  p.path(run.l.slice(0,t+1).map((v,i) => [replay.steps[i],v[e]]), { 'stroke-width': 2.8 });
  p.add('line', { x1:p.x(replay.steps[t]), x2:p.x(replay.steps[t]), y1:p.margin.t, y2:p.h-p.margin.b, stroke:color(), opacity:.35, 'stroke-dasharray':'3 3' });
  p.dot(replay.steps[t],run.l[t][e],{r:5});
  text('[data-q]', `${fmt(100*run.q[t])}%`);
  text('[data-candidates]', fmt(run.c[t][e],2));
  text('[data-replay-success]', `${fmt(run.s[t][e])}%`);
  const u = run.u[t][e];
  const v = run.v[t][e];
  const rate = replay.rate[t];
  const delta = rate*((1-alpha())*u-alpha()*v);
  text('[data-rate]', fmt(rate,4));
  text('[data-one-alpha]', fmt(1-alpha(),2));
  text('[data-equation-alpha]', fmt(alpha(),2));
  text('[data-u]', signed(u,3));
  text('[data-v]', signed(v,3));
  text('[data-delta]', signed(delta,5));
  if (t === 0) {
    text('[data-correction]', 'Shared initialization. No batch has been sampled and no controller update has occurred.');
  } else {
    const phrase = delta > 0 ? 'Success gains are ahead of the requested balance. Increase the cost penalty.' : delta < 0 ? 'Cost savings are ahead of the requested balance. Decrease the cost penalty.' : 'The sampled gains match the requested balance. Keep the penalty unchanged.';
    text('[data-correction]', `${phrase} λ: ${fmt(run.used[t][e],5)} → ${fmt(run.l[t][e],5)}.`);
  }
}

async function loadReplay() {
  const id = problem().id;
  const request = ++replayRequest;
  stopPlayback();
  $('.swe-replay-content').setAttribute('aria-busy','true');
  $('.swe-replay-content').inert = true;
  text('.swe-replay-status','Loading this problem’s recorded runs…');
  try {
    if (!replayCache.has(id)) {
      const url = new URL(`replays/${id}.json`, new URL(root.dataset.source, location.href));
      replayCache.set(id, fetch(url).then((response) => {
        if (!response.ok) throw new Error(`Replay request failed: ${response.status}`);
        return response.json();
      }).catch((error) => { replayCache.delete(id); throw error; }));
    }
    const loaded = await replayCache.get(id);
    if (request !== replayRequest) return;
    replay = loaded;
    text('.swe-replay-status','');
    $('.swe-replay-content').removeAttribute('aria-busy');
    $('.swe-replay-content').inert = false;
    renderReplay();
  } catch (error) {
    if (request !== replayRequest) return;
    const status = $('.swe-replay-status');
    status.textContent = 'This replay could not be loaded. ';
    const retry = document.createElement('button');
    retry.type = 'button'; retry.className = 'swe-button swe-outline'; retry.textContent = 'Try again';
    retry.addEventListener('click', loadReplay);
    status.append(retry);
    console.error(error);
  }
}

function stopPlayback() {
  clearInterval(playback);
  playback = null;
  text('[data-play]','Play');
  $('[data-play]').setAttribute('aria-label','Play training replay');
  $('[data-correction]').setAttribute('aria-live','polite');
}

function gridColor(value) {
  const target = value < 0 ? [56,97,232] : [219,84,46];
  const base = [246,245,250];
  const intensity = Math.min(1,Math.abs(value)/5);
  return `rgb(${base.map((v,i) => Math.round(v+(target[i]-v)*intensity)).join(',')})`;
}

function renderGrid() {
  const e = state.effort;
  $$('[data-grid-problem]').forEach((button) => {
    const i = +button.dataset.gridProblem;
    const p = data.problems[i];
    const gap = balance(p);
    button.style.backgroundColor = gridColor(gap);
    button.setAttribute('aria-pressed',String(i===state.problem));
    button.tabIndex = i===state.problem ? 0 : -1;
    button.setAttribute('aria-label',`Problem ${i+1}, p good ${fmt(p.pg,3)}, tool ratio ${fmt(p.ratio,3)}. Balance gap ${signed(gap,2)}. Cost reduction ${fmt(p.results[method()].saving[e])}%, success gain ${signed(p.results[method()].gain[e])}%.`);
    button.title = `Problem ${i+1} · balance gap ${signed(gap,2)}`;
  });
}

function renderProblem() {
  const p = problem();
  const r = selected();
  const fixed = p.results[0];
  const e = state.effort;
  text('[data-problem-title]', `Problem ${state.problem+1} / 100`);
  text('[data-problem-params]', `p_good = ${fmt(p.pg,3)} · p_bad = ${fmt(p.pb,3)} · ${EFFORTS[e]} effort`);
  text('[data-balance]', `Balance gap ${signed(balance(p),2)}`);
  text('[data-problem-saving]', `${signed(r.saving[e])}%`);
  text('[data-problem-gain]', `${signed(r.gain[e])}%`);
  const maxCost = Math.max(p.baseCost[e], ...r.seeds.map((s) => s.cost[e]), ...fixed.seeds.map((s) => s.cost[e]))*1.15;
  const success = (cost,q) => 100*(q*p.pg*cost/(1+p.pg*cost)+(1-q)*p.pb*cost/(1+p.pb*cost));
  const maxSuccess = Math.min(100, Math.max(success(maxCost,r.q), success(maxCost,fixed.q))*1.12);
  const c = chart('problem', { xDomain:[0,maxCost], yDomain:[0,maxSuccess], xLabel:'Mean cost (attempts)', yLabel:'Success rate (%)', height:270 });
  // The trained curves nearly coincide. Draw the fixed curve as a fine dashed
  // overlay so both remain visible without displacing any measured coordinates.
  for (const [q, ink, dash] of [[.5,INITIAL,'5 4'],[r.q,color(),'none'],[fixed.q,FIXED,'3 5']]) {
    c.path(Array.from({length:90},(_,i) => { const cost = i*maxCost/89; return [cost,success(cost,q)]; }), { stroke:ink, 'stroke-width': ink === FIXED ? 1.4 : 2, 'stroke-dasharray':dash, opacity: .85 });
  }
  c.dot(p.baseCost[e],p.baseSuccess[e],{fill:'#fff',stroke:INITIAL,r:5,'stroke-width':2});
  for (const [result,ink] of [[fixed,FIXED],[r,color()]]) {
    c.dot(result.cost[e], result.success[e], {fill:ink,r:6.5});
    for (const [i,s] of result.seeds.entries()) {
      const point = c.dot(s.cost[e],s.success[e],{fill:'#fff',stroke:ink,r:3.5,'stroke-width':1.5});
      point.append(node('title',{},`Seed ${p.seeds[i]} · ${fmt(s.cost[e],3)} attempts, ${fmt(s.success[e],3)}% success`));
    }
  }
  const rows = [];
  for (let i=0;i<3;i++) for (const [r,label] of [[fixed,'Fixed λ'],[selected(),`α = ${alpha()}`]]) {
    const tr = document.createElement('tr');
    for (const value of [p.seeds[i],label,fmt(r.seeds[i].cost[e],3),`${fmt(r.seeds[i].success[e],3)}%`]) {
      const td = document.createElement('td'); td.textContent=value; tr.append(td);
    }
    rows.push(tr);
  }
  $('[data-seed-table]').replaceChildren(...rows);
}

function render() {
  syncControls();
  renderDirection();
  renderReplay();
  renderGrid();
  renderProblem();
}

function setProblem(i) {
  state.problem=i;
  syncControls();
  renderGrid();
  renderProblem();
  loadReplay();
}

function setupControls() {
  const ticks = $('[data-alpha-controls]');
  ALPHAS.forEach((value,i) => {
    const b = document.createElement('button');
    b.type='button'; b.textContent=String(value); b.dataset.alphaIndex=i;
    b.setAttribute('aria-label',`Alpha ${value}: ${NAMES[i]}`);
    b.addEventListener('click',() => { state.alpha=i; render(); });
    ticks.append(b);
  });
  $('#swe-alpha').addEventListener('input',(ev) => { state.alpha=+ev.target.value; render(); });
  $$('[data-alpha-select]').forEach((el) => {
    el.replaceChildren(...ALPHAS.map((a,i) => new Option(String(a),String(i))));
    el.addEventListener('change',() => { state.alpha=+el.value; render(); });
  });
  $$('[data-effort]').forEach((el) => el.addEventListener('change',() => { state.effort=+el.value; render(); }));
  $$('[data-problem]').forEach((el) => {
    el.replaceChildren(...data.problems.map((p,i) => new Option(`${String(i+1).padStart(2,'0')} · p_good ${fmt(p.pg,3)} · ratio ${fmt(p.ratio,2)}`,String(i))));
    el.addEventListener('change',() => setProblem(+el.value));
  });
  $('[data-seed]').addEventListener('change',(ev) => { state.seed=+ev.target.value; stopPlayback(); renderReplay(); });
  $('[data-show-problems]').addEventListener('click',(ev) => {
    state.dots=!state.dots;
    ev.currentTarget.setAttribute('aria-pressed',String(state.dots));
    ev.currentTarget.textContent=state.dots ? 'Hide individual problems' : 'Show all 100 problems';
    $('[data-dot-help]').hidden=!state.dots;
    renderDirection();
  });
  $('#swe-step').addEventListener('input',(ev) => { stopPlayback(); state.step=+ev.target.value; renderReplay(); });
  $('[data-play]').addEventListener('click',() => {
    if (playback) { stopPlayback(); return; }
    if (!currentReplay()) return;
    if (state.step >= replay.steps.length-1) state.step=0;
    text('[data-play]','Pause');
    $('[data-play]').setAttribute('aria-label','Pause training replay');
    $('[data-correction]').setAttribute('aria-live','off');
    renderReplay();
    playback=setInterval(() => {
      state.step++;
      renderReplay();
      if (state.step>=replay.steps.length-1) stopPlayback();
    },reducedMotion.matches ? 700 : 150);
  });
  document.addEventListener('visibilitychange',() => { if (document.hidden) stopPlayback(); });
  const grid = $('.swe-grid');
  // Display large p_good at the top; ratio increases to the right.
  for (let row=9;row>=0;row--) for (let col=0;col<10;col++) {
    const i=data.problems.findIndex((p) => p.row===row && p.column===col);
    const b=document.createElement('button');
    b.type='button'; b.dataset.gridProblem=i;
    b.addEventListener('click',() => setProblem(i));
    b.addEventListener('keydown',(ev) => {
      const moves={ArrowLeft:[0,-1],ArrowRight:[0,1],ArrowUp:[1,0],ArrowDown:[-1,0]};
      if (!moves[ev.key]) return;
      ev.preventDefault();
      const [dr,dc]=moves[ev.key];
      const next=data.problems.findIndex((p) => p.row===Math.max(0,Math.min(9,row+dr)) && p.column===Math.max(0,Math.min(9,col+dc)));
      setProblem(next);
      $(`[data-grid-problem="${next}"]`).focus();
    });
    grid.append(b);
  }
  $$('[data-preset]').forEach((b) => b.addEventListener('click',() => {
    const metric=b.dataset.preset;
    const scores=data.problems.map((p) => metric==='saving' ? p.results[method()].saving[state.effort] : metric==='miss' ? Math.abs(balance(p)) : -Math.hypot(p.results[method()].saving[state.effort],p.results[method()].gain[state.effort]));
    setProblem(scores.indexOf(Math.max(...scores)));
  }));
  $('[data-replay-this]').addEventListener('click',() => { scrollTo($('#swe-replay-title')); });
  let frame;
  let width = 0;
  new ResizeObserver(([entry]) => {
    if (Math.abs(entry.contentRect.width-width)<1) return;
    width=entry.contentRect.width;
    cancelAnimationFrame(frame);
    frame=requestAnimationFrame(() => { renderDirection(); renderReplay(); renderProblem(); });
  }).observe(root);
}

async function init() {
  $('.swe-load').hidden=false;
  try {
    const response=await fetch(root.dataset.source);
    if (!response.ok) throw new Error(`Overview request failed: ${response.status}`);
    data=await response.json();
    $('.swe-interactives').hidden=false;
    $('.swe-load').hidden=true;
    setupControls();
    render();
    // Delay detailed histories until the reader approaches the replay.
    const observer=new IntersectionObserver((entries) => {
      if (entries.some((entry) => entry.isIntersecting)) {
        observer.disconnect();
        if (!currentReplay()) loadReplay();
      }
    },{rootMargin:'300px'});
    observer.observe($('#swe-replay-title'));
  } catch (error) {
    $('.swe-interactives').hidden=true;
    $('.swe-load').hidden=false;
    $('.swe-load').textContent='The interactive results could not be loaded. Reload to try again, or download the figure data below. ';
    const a=document.createElement('a'); a.href=root.dataset.source; a.textContent='Download figure data';
    $('.swe-load').append(a);
    console.error(error);
  }
}

if (root) init();
