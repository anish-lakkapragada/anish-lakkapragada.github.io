---
layout: bookshelf
title: bookshelf
permalink: /bookshelf/
sitemap: false
icon: fas fa-book-open
order: 3
---

<div class="bookshelf">
  <p class="bookshelf-intro"><span id="typed"></span><span class="caret" id="caret">|</span></p>
  <div id="book-grid"></div>
</div>

<script>
  (function () {
    var MESSAGE = "often, i feel down about being unable to solve problems. but reading reminds me learning can happen in other ways. these books have taught me a lot.";

    function ready(fn) {
      if (document.readyState !== "loading") fn();
      else document.addEventListener("DOMContentLoaded", fn);
    }

    ready(function () {
      var shelf = document.querySelector(".bookshelf");
      var typedEl = document.getElementById("typed");
      var caretEl = document.getElementById("caret");
      var root = document.getElementById("book-grid");
      if (!shelf || !typedEl || !root) return;

      // 1) fade the page in
      requestAnimationFrame(function () {
        shelf.classList.add("loaded");
      });

      var reduceMotion =
        window.matchMedia &&
        window.matchMedia("(prefers-reduced-motion: reduce)").matches;

      // 2) type the intro message, then render the books
      if (reduceMotion) {
        typedEl.textContent = MESSAGE;
        if (caretEl) caretEl.classList.add("done");
        loadBooks();
      } else {
        var i = 0;
        // hold + blink the cursor for 1.5s right before "but" (next sentence)
        var pauseAt = MESSAGE.indexOf("but reading");
        (function typeChar() {
          if (i <= MESSAGE.length) {
            typedEl.textContent = MESSAGE.slice(0, i);
            var pause = i === pauseAt; // text now ends just before "but"
            i++;
            setTimeout(typeChar, pause ? 750 : 45);
          } else {
            if (caretEl) caretEl.classList.add("done");
            loadBooks();
          }
        })();
      }

      // 3) fetch books.json and build the category sections
      function loadBooks() {
        fetch("{{ '/assets/books.json' | relative_url }}")
          .then(function (r) {
            if (!r.ok) throw new Error("HTTP " + r.status);
            return r.json();
          })
          .then(renderCategories)
          .catch(function (err) {
            root.innerHTML =
              '<p style="color:#5b4827;">Could not load books.</p>';
            console.error("bookshelf: failed to load books.json", err);
          });
      }

      function renderCategories(data) {
        // supports { categories: [{ category, books: [...] }] };
        // tolerates a bare array of categories or a flat array of books.
        var categories = (data && data.categories) || data || [];

        var observer = makeObserver();
        var frag = document.createDocumentFragment();

        categories.forEach(function (group) {
          // a flat list of books (no grouping) -> render under one untitled section
          var name = group && group.category;
          var books = (group && group.books) || (group && group.title ? [group] : []);
          if (!books.length) return;

          var section = document.createElement("section");
          section.className = "book-section";

          if (name) {
            var totalPages = books.reduce(function (sum, b) {
              return sum + (b.pageCount || 0);
            }, 0);
            var heading = document.createElement("h2");
            heading.className = "book-section-title";
            heading.textContent =
              name + " (" + totalPages.toLocaleString() + " pages)";
            section.appendChild(heading);
          }

          var grid = document.createElement("div");
          grid.className = "book-grid";

          // collapse a series (books sharing a `series`) into one stacked cell,
          // placed where the series' first (highest-ranked) member sits.
          var items = [];
          var seriesSlot = {};
          books.forEach(function (book) {
            if (book.series) {
              if (seriesSlot[book.series] == null) {
                seriesSlot[book.series] = items.length;
                items.push({ type: "series", name: book.series, books: [book] });
              } else {
                items[seriesSlot[book.series]].books.push(book);
              }
            } else {
              items.push({ type: "single", book: book });
            }
          });

          items.forEach(function (item, idx) {
            var card;
            if (item.type === "series") {
              item.books.sort(function (a, b) {
                return (a.seriesIndex || 0) - (b.seriesIndex || 0); // book 1 on top
              });
              card = buildStack(item.name, item.books);
            } else {
              card = buildCard(item.book);
            }
            // small per-item stagger so each section cascades in
            card.style.transitionDelay = (idx % 8) * 45 + "ms";
            grid.appendChild(card);
          });

          section.appendChild(grid);
          frag.appendChild(section);
        });

        root.appendChild(frag);

        // reveal each category section one by one as it enters the viewport
        var sections = root.querySelectorAll(".book-section");
        sections.forEach(function (section) {
          if (observer) observer.observe(section);
          else section.classList.add("is-visible");
        });
      }

      function buildCard(book) {
        var authors = (book.authors || []).join(", ");
        var pages = book.pageCount ? book.pageCount + " pages" : "";
        var sub = [authors, pages].filter(Boolean).join(" · ");

        var card = document.createElement("div");
        card.className = "book-card";

        var frame = document.createElement("div");
        frame.className = "book-cover-frame";

        var coverUrl = book.cover && book.cover.url;
        if (coverUrl) {
          var img = document.createElement("img");
          img.loading = "lazy";
          img.alt = book.title || "Book cover";
          img.src = coverUrl;
          img.addEventListener("error", function () {
            renderPlaceholder(frame, book.title);
          });
          frame.appendChild(img);
        } else {
          renderPlaceholder(frame, book.title);
        }

        var meta = document.createElement("div");
        meta.className = "book-meta";
        var titleLine = document.createElement("div");
        titleLine.className = "book-title";
        titleLine.textContent = book.title || "";
        var subLine = document.createElement("div");
        subLine.className = "book-sub";
        subLine.textContent = sub;
        meta.appendChild(titleLine);
        meta.appendChild(subLine);

        card.appendChild(frame);
        card.appendChild(meta);
        return card;
      }

      // a series rendered as a stacked deck of covers (book 1 on top)
      function buildStack(seriesName, books) {
        var totalPages = books.reduce(function (s, b) {
          return s + (b.pageCount || 0);
        }, 0);

        var card = document.createElement("div");
        card.className = "book-card book-card--stack";

        var stack = document.createElement("div");
        stack.className = "book-stack";
        // center the hover fan around the middle of the deck
        stack.style.setProperty("--half", (books.length - 1) / 2);

        books.forEach(function (book, i) {
          var frame = document.createElement("div");
          frame.className = "book-cover-frame stack-layer";
          frame.style.setProperty("--i", i);
          frame.style.zIndex = books.length - i; // book 1 sits on top

          var coverUrl = book.cover && book.cover.url;
          if (coverUrl) {
            var img = document.createElement("img");
            img.loading = "lazy";
            img.alt = book.title || "Book cover";
            img.src = coverUrl;
            img.addEventListener("error", function () {
              renderPlaceholder(frame, book.title);
            });
            frame.appendChild(img);
          } else {
            renderPlaceholder(frame, book.title);
          }
          stack.appendChild(frame);
        });

        var badge = document.createElement("span");
        badge.className = "stack-count";
        badge.textContent = "×" + books.length;
        stack.appendChild(badge);

        var meta = document.createElement("div");
        meta.className = "book-meta";
        var titleLine = document.createElement("div");
        titleLine.className = "book-title";
        titleLine.textContent = seriesName;
        var subLine = document.createElement("div");
        subLine.className = "book-sub";
        subLine.textContent =
          books.length + " books · " + totalPages.toLocaleString() + " pages";
        meta.appendChild(titleLine);
        meta.appendChild(subLine);

        card.appendChild(stack);
        card.appendChild(meta);
        return card;
      }

      function renderPlaceholder(frame, title) {
        frame.classList.add("is-placeholder");
        frame.innerHTML = "";
        var t = document.createElement("span");
        t.className = "placeholder-title";
        t.textContent = title || "Untitled";
        frame.appendChild(t);
      }

      function makeObserver() {
        if (!("IntersectionObserver" in window)) return null;
        return new IntersectionObserver(
          function (entries, obs) {
            entries.forEach(function (entry) {
              if (entry.isIntersecting) {
                entry.target.classList.add("is-visible");
                obs.unobserve(entry.target);
              }
            });
          },
          { threshold: 0.1, rootMargin: "0px 0px -8% 0px" }
        );
      }
    });
  })();
</script>
