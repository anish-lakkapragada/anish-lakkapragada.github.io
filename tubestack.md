---
layout: tubestack
title: tubestack
permalink: /tubestack/
sitemap: false
---

<!--
  Download URLs are resolved at page load by the inline <script> at the
  bottom: it hits the GitHub API for the latest release and rewrites the
  download buttons to the current .dmg / .exe asset URLs by extension match,
  so this page never needs editing on future releases.
  Fallback href points users at the releases page if JS / the API fails.
-->

<main class="ts-wrap">

  <h1 class="ts-title">tubestack</h1>

  <nav class="ts-social" aria-label="social links">
    <a href="https://github.com/anish-lakkapragada/tubestack"
       target="_blank" rel="noreferrer" aria-label="github repository">
      <svg viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M12 .5C5.65.5.5 5.65.5 12c0 5.08 3.29 9.39 7.86 10.91.58.11.79-.25.79-.56 0-.27-.01-1-.02-1.96-3.2.7-3.87-1.54-3.87-1.54-.52-1.32-1.27-1.67-1.27-1.67-1.04-.71.08-.69.08-.69 1.15.08 1.76 1.18 1.76 1.18 1.02 1.75 2.68 1.24 3.34.95.1-.74.4-1.24.73-1.53-2.55-.29-5.24-1.28-5.24-5.69 0-1.26.45-2.28 1.18-3.09-.12-.29-.51-1.46.11-3.04 0 0 .97-.31 3.18 1.18a11.04 11.04 0 0 1 5.79 0c2.21-1.49 3.18-1.18 3.18-1.18.62 1.58.23 2.75.11 3.04.74.81 1.18 1.83 1.18 3.09 0 4.42-2.69 5.39-5.25 5.68.41.36.78 1.06.78 2.14 0 1.55-.01 2.79-.01 3.17 0 .31.21.68.8.56C20.21 21.39 23.5 17.08 23.5 12 23.5 5.65 18.35.5 12 .5z"/></svg>
      <span>repo</span>
    </a>
    <a href="https://x.com/_anishlk" target="_blank" rel="noreferrer" aria-label="receive updates on x">
      <span>receive updates on</span>
      <svg viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24h-6.654l-5.214-6.817-5.965 6.817H1.685l7.73-8.835L1.254 2.25H8.08l4.713 6.231 5.45-6.231zm-1.161 17.52h1.833L7.084 4.126H5.117l11.966 15.644z"/></svg>
    </a>
  </nav>

  <p class="ts-lede">
    a quiet, local desktop app for exploring youtube without the visual pollution. free for mac &amp; windows.
  </p>

  <div class="ts-cta-row">
    <div class="ts-dropdown" id="ts-mac-dropdown">
      <button class="ts-btn ts-btn-split" type="button"
              aria-haspopup="true" aria-expanded="false"
              aria-controls="ts-mac-menu">
        <svg viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M17.05 20.28c-.98.95-2.05.8-3.08.35-1.09-.46-2.09-.48-3.24 0-1.44.62-2.2.44-3.06-.35C2.79 15.25 3.51 7.59 9.05 7.31c1.35.07 2.29.74 3.08.8 1.18-.24 2.31-.93 3.57-.84 1.51.12 2.65.72 3.4 1.8-3.12 1.87-2.38 5.98.48 7.13-.57 1.5-1.31 2.99-2.54 4.09zM12.03 7.25c-.15-2.23 1.66-4.07 3.74-4.25.29 2.58-2.34 4.5-3.74 4.25z"/></svg>
        <span>download for mac</span>
        <svg class="ts-chevron" viewBox="0 0 12 12" aria-hidden="true"><path fill="currentColor" d="M2 4l4 4 4-4z"/></svg>
      </button>
      <div class="ts-dropdown-menu" id="ts-mac-menu" role="menu">
        <a class="ts-dropdown-item" role="menuitem"
           href="https://github.com/anish-lakkapragada/tubestack/releases/latest"
           data-asset-pattern="arm64\.dmg$">
          <span class="ts-dropdown-title">apple silicon</span>
          <span class="ts-dropdown-sub">M1 / M2 / M3 / M4</span>
        </a>
        <a class="ts-dropdown-item" role="menuitem"
           href="https://github.com/anish-lakkapragada/tubestack/releases/latest"
           data-asset-pattern="x64\.dmg$">
          <span class="ts-dropdown-title">intel</span>
          <span class="ts-dropdown-sub">x86_64</span>
        </a>
      </div>
    </div>
    <a class="ts-btn" href="/tubestack/windows/">
      <svg viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M3 5.5L10.5 4.5v7H3v-6zM11.5 4.4L21 3v8.5h-9.5V4.4zM3 12.5h7.5v7L3 18.5v-6zM11.5 12.5H21V21l-9.5-1.4V12.5z"/></svg>
      download for windows
    </a>
  </div>

  <div class="ts-video" aria-label="tubestack screen recording">
    <video controls preload="auto" playsinline
           poster="/assets/Tubestack-desktop-poster.jpg">
      <source src="/assets/Tubestack-desktop.mp4" type="video/mp4">
    </video>
  </div>

  <section class="ts-body">
    <p>
      tubestack is a minimal desktop application for reading youtube content (videos & shorts) in a more clean, organized way. it is designed to cultivate self-reflection and intentful exploration in one&#39;s interests.
    </p>
    <p>
      i&#39;ve been using tubestack locally for over three months and would be honored if you gave it a try. tubestack is 100% free and local.
    </p>
  </section>

</main>

<script>
  (function () {
    fetch('https://api.github.com/repos/anish-lakkapragada/tubestack/releases/latest', {
      headers: { 'Accept': 'application/vnd.github+json' }
    })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (data) {
        if (!data || !data.assets) return;
        document.querySelectorAll('a[data-asset-pattern]').forEach(function (a) {
          var rx = new RegExp(a.getAttribute('data-asset-pattern'), 'i');
          var match = data.assets.find(function (asset) { return rx.test(asset.name); });
          if (match) a.href = match.browser_download_url;
        });
      })
      .catch(function () { /* leave fallback href in place */ });

    // Mac dropdown: open/close + outside-click + Esc to close
    var dd = document.getElementById('ts-mac-dropdown');
    if (!dd) return;
    var btn = dd.querySelector('button');
    function close() {
      dd.classList.remove('is-open');
      btn.setAttribute('aria-expanded', 'false');
    }
    function open() {
      dd.classList.add('is-open');
      btn.setAttribute('aria-expanded', 'true');
    }
    btn.addEventListener('click', function (e) {
      e.stopPropagation();
      dd.classList.contains('is-open') ? close() : open();
    });
    document.addEventListener('click', function (e) {
      if (!dd.contains(e.target)) close();
    });
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') close();
    });
    // close after picking an option (the link still navigates)
    dd.querySelectorAll('.ts-dropdown-item').forEach(function (a) {
      a.addEventListener('click', close);
    });
  })();
</script>
