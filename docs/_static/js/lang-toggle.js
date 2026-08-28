/**
 * Language toggle for the Coda docs (EN <-> 中文).
 *
 * The two languages are two independent Sphinx trees published as:
 *   local http.server   /index.html             and  /zh/index.html
 *   GitHub Pages        /LoongSage/index.html     and  /LoongSage/zh/index.html
 *
 * So the position of the "zh" segment differs between environments. This
 * script therefore never assumes an index: it asks Sphinx how far up the root
 * of the current tree is (the data-content_root attribute Sphinx writes on
 * <html>) and only checks whether that root ends with a "zh" segment.
 */
(function () {
  "use strict";

  var LANG_SEGMENT = "zh";
  var LABELS = { en: "EN", zh: "中文" };
  var HINTS = { en: "Switch to English", zh: "切换到中文" };
  var PROBE_TIMEOUT_MS = 1500;

  // Root of the language tree the current page belongs to, e.g. "/zh/".
  function langRootPath() {
    var root = document.documentElement.getAttribute("data-content_root") || "./";
    return new URL(root, window.location.href).pathname;
  }

  function segments(path) {
    return path.split("/").filter(function (s) {
      return s.length > 0;
    });
  }

  function detect() {
    var segs = segments(langRootPath());
    return segs.length && segs[segs.length - 1] === LANG_SEGMENT ? "zh" : "en";
  }

  // Prefix shared by both languages: "/" locally, "/LoongSage/" on GitHub Pages.
  function siteRootPath(current, rootPath) {
    return current === "en"
      ? rootPath
      : rootPath.replace(new RegExp(LANG_SEGMENT + "/$"), "");
  }

  function targets(target) {
    var rootPath = langRootPath();
    var siteRoot = siteRootPath(detect(), rootPath);
    var rel = window.location.pathname.slice(rootPath.length);
    var base = target === "zh" ? siteRoot + LANG_SEGMENT + "/" : siteRoot;
    // The in-page anchor is dropped on purpose: heading slugs differ between
    // languages, so keeping it would often land on a non-existent anchor.
    return { page: base + rel, home: base };
  }

  // The trees are meant to stay 1:1, but a page may exist in one language
  // only. Probe first and fall back to that language's home page rather than
  // dropping the reader on a 404.
  function go(target) {
    var urls = targets(target);
    if (!window.fetch || !window.AbortController || window.location.protocol === "file:") {
      window.location.assign(urls.page);
      return;
    }
    var controller = new AbortController();
    var timer = window.setTimeout(function () {
      controller.abort();
    }, PROBE_TIMEOUT_MS);
    window
      .fetch(urls.page, { method: "HEAD", signal: controller.signal })
      .then(function (res) {
        window.clearTimeout(timer);
        window.location.assign(res.ok ? urls.page : urls.home);
      })
      .catch(function () {
        window.clearTimeout(timer);
        window.location.assign(urls.page);
      });
  }

  function buildWidget(current) {
    var wrap = document.createElement("div");
    wrap.className = "coda-lang-toggle";
    wrap.setAttribute("role", "group");
    wrap.setAttribute("aria-label", current === "zh" ? "语言" : "Language");

    ["en", "zh"].forEach(function (lang) {
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "coda-lang-toggle__btn";
      btn.textContent = LABELS[lang];
      btn.title = HINTS[lang];
      btn.setAttribute("aria-label", HINTS[lang]);
      if (lang === current) {
        btn.classList.add("is-active");
        btn.setAttribute("aria-current", "true");
        btn.disabled = true;
      } else {
        btn.addEventListener("click", function () {
          go(lang);
        });
      }
      wrap.appendChild(btn);
    });
    return wrap;
  }

  // Theme header is assembled progressively, and its class names change
  // between theme releases — try the known hosts, keep watching for a short
  // while, then fall back to a floating button so the toggle never vanishes.
  var HOSTS = [
    ".article-header-buttons",
    ".header-article-items__end",
    ".bd-header-article .header-article-items__end",
  ];

  function insert(widget) {
    for (var i = 0; i < HOSTS.length; i++) {
      var host = document.querySelector(HOSTS[i]);
      if (host) {
        host.appendChild(widget); // last item of the header button group
        return true;
      }
    }
    return false;
  }

  function mount() {
    if (document.querySelector(".coda-lang-toggle")) return;
    var widget = buildWidget(detect());
    if (insert(widget)) return;

    var observer = new MutationObserver(function () {
      if (insert(widget)) observer.disconnect();
    });
    observer.observe(document.body, { childList: true, subtree: true });
    window.setTimeout(function () {
      observer.disconnect();
      if (!widget.isConnected) {
        widget.classList.add("coda-lang-toggle--floating");
        document.body.appendChild(widget);
      }
    }, 5000);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", mount);
  } else {
    mount();
  }
})();
