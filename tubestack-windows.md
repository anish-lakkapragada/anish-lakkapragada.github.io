---
layout: tubestack
title: tubestack — for windows
permalink: /tubestack/windows/
sitemap: false
---

<main class="ts-wrap">

  <h1 class="ts-title">tubestack (windows).</h1>

  <nav class="ts-social" aria-label="social links">
    <a href="https://github.com/anish-lakkapragada/anish-lakkapragada.github.io"
       target="_blank" rel="noreferrer" aria-label="github repository">
      <svg viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M12 .5C5.65.5.5 5.65.5 12c0 5.08 3.29 9.39 7.86 10.91.58.11.79-.25.79-.56 0-.27-.01-1-.02-1.96-3.2.7-3.87-1.54-3.87-1.54-.52-1.32-1.27-1.67-1.27-1.67-1.04-.71.08-.69.08-.69 1.15.08 1.76 1.18 1.76 1.18 1.02 1.75 2.68 1.24 3.34.95.1-.74.4-1.24.73-1.53-2.55-.29-5.24-1.28-5.24-5.69 0-1.26.45-2.28 1.18-3.09-.12-.29-.51-1.46.11-3.04 0 0 .97-.31 3.18 1.18a11.04 11.04 0 0 1 5.79 0c2.21-1.49 3.18-1.18 3.18-1.18.62 1.58.23 2.75.11 3.04.74.81 1.18 1.83 1.18 3.09 0 4.42-2.69 5.39-5.25 5.68.41.36.78 1.06.78 2.14 0 1.55-.01 2.79-.01 3.17 0 .31.21.68.8.56C20.21 21.39 23.5 17.08 23.5 12 23.5 5.65 18.35.5 12 .5z"/></svg>
      <span>github</span>
    </a>
    <a href="https://x.com/_anishlk" target="_blank" rel="noreferrer" aria-label="receive updates on x">
      <span>receive updates on</span>
      <svg viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24h-6.654l-5.214-6.817-5.965 6.817H1.685l7.73-8.835L1.254 2.25H8.08l4.713 6.231 5.45-6.231zm-1.161 17.52h1.833L7.084 4.126H5.117l11.966 15.644z"/></svg>
    </a>
  </nav>

  <p class="ts-lede">
    one quick note before you download.
  </p>

  <section class="ts-body">
    <p>
      windows will almost certainly show a smartscreen warning. something like &ldquo;windows protected your pc&rdquo;. this is normal for new, independent applications.
    </p>
    <p>
      to run tubestack: click <strong>more info</strong>, then <strong>run anyway</strong>. that's all :)
    </p>
  </section>

  <div class="ts-cta-row" style="grid-template-columns: 1fr;">
    <a id="ts-download-win" class="ts-btn"
       href="https://github.com/anish-lakkapragada/tubestack/releases/latest"
       data-asset-pattern="windows.*\.exe$">
      <svg viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M3 5.5L10.5 4.5v7H3v-6zM11.5 4.4L21 3v8.5h-9.5V4.4zM3 12.5h7.5v7L3 18.5v-6zM11.5 12.5H21V21l-9.5-1.4V12.5z"/></svg>
      download .exe
    </a>
  </div>

  <p class="ts-switch">on a mac instead? <a href="/tubestack/">go here →</a></p>

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
  })();
</script>
