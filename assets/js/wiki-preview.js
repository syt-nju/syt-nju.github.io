(function(root, factory) {
  var api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  } else {
    root.WikiPreview = api;
  }
}(this, function() {
  "use strict";

  var HOST_LABELS = {
    "arxiv.org": "arXiv",
    "thinkingmachines.ai": "Thinking Machines",
    "tinker-docs.thinkingmachines.ai": "Tinker Docs",
    "zhuanlan.zhihu.com": "知乎专栏",
    "saraswatmks.github.io": "Floating Bytes",
    "nrehiew.github.io": "nrehiew"
  };

  function trailingSlash(pathname) {
    if (!pathname || pathname === "/") {
      return pathname || "/";
    }
    return pathname.charAt(pathname.length - 1) === "/" ? pathname : pathname + "/";
  }

  function parseHref(href, origin) {
    var base = origin || "https://example.invalid";
    var url;
    try {
      url = new URL(href, base);
    } catch (error) {
      return null;
    }
    if (url.protocol !== "http:" && url.protocol !== "https:") {
      return null;
    }
    var baseUrl;
    try {
      baseUrl = new URL(base);
    } catch (error) {
      baseUrl = null;
    }
    var sameOrigin = Boolean(baseUrl) && url.origin === baseUrl.origin;
    return {
      href: url.href,
      host: url.hostname.replace(/^www\./, ""),
      path: trailingSlash(url.pathname),
      hash: decodeURIComponent((url.hash || "").replace(/^#/, "")),
      sameOrigin: sameOrigin
    };
  }

  function isWikiPath(path) {
    return path === "/wiki/" || path.indexOf("/wiki/") === 0;
  }

  function findEntry(entries, path) {
    var needle = trailingSlash(path);
    for (var i = 0; i < entries.length; i += 1) {
      var entryPath = trailingSlash(String(entries[i].url || "").split("#")[0]);
      if (entryPath === needle) {
        return entries[i];
      }
    }
    return null;
  }

  function hostLabel(host) {
    return HOST_LABELS[host] || host;
  }

  function previewForLink(entries, href, origin) {
    var parsed = parseHref(href, origin);
    if (!parsed) {
      return null;
    }
    if (parsed.sameOrigin && isWikiPath(parsed.path)) {
      if (parsed.path === "/wiki/search.json/") {
        return null;
      }
      var entry = findEntry(entries, parsed.path);
      if (!entry) {
        return {
          kind: "知识库",
          title: parsed.path,
          summary: "",
          section: parsed.hash
        };
      }
      return {
        kind: "知识库",
        title: entry.title,
        summary: entry.summary || "",
        section: parsed.hash
      };
    }
    if (parsed.sameOrigin) {
      return null;
    }
    return {
      kind: "原文",
      title: hostLabel(parsed.host),
      summary: parsed.host + parsed.path.replace(/\/$/, ""),
      section: ""
    };
  }

  function prefersFineHover() {
    return window.matchMedia && window.matchMedia("(hover: hover) and (pointer: fine)").matches;
  }

  function placeCard(card, anchor) {
    var rect = anchor.getBoundingClientRect();
    var width = card.offsetWidth;
    var height = card.offsetHeight;
    var left = rect.left + window.scrollX;
    var top = rect.bottom + window.scrollY + 8;
    var maxLeft = window.scrollX + document.documentElement.clientWidth - width - 12;
    left = Math.max(window.scrollX + 12, Math.min(left, maxLeft));
    if (rect.bottom + height + 16 > window.innerHeight && rect.top > height + 16) {
      top = rect.top + window.scrollY - height - 8;
    }
    card.style.left = left + "px";
    card.style.top = top + "px";
  }

  function renderCard(card, preview) {
    card.textContent = "";
    var kind = document.createElement("span");
    var title = document.createElement("strong");
    kind.className = "wiki-link-preview__kind";
    kind.textContent = preview.kind;
    title.textContent = preview.title;
    card.appendChild(kind);
    card.appendChild(title);
    if (preview.section) {
      var section = document.createElement("em");
      section.textContent = preview.section.replace(/-/g, " ");
      card.appendChild(section);
    }
    if (preview.summary) {
      var summary = document.createElement("span");
      summary.textContent = preview.summary;
      card.appendChild(summary);
    }
  }

  function shouldSkip(anchor) {
    if (!anchor || anchor.target === "_download") {
      return true;
    }
    if (anchor.closest(".wiki-sidebar, .wiki-pagination, .wiki-search, .wiki-card-grid")) {
      return true;
    }
    if (anchor.classList.contains("wiki-card") || anchor.classList.contains("wiki-search-result")) {
      return true;
    }
    return false;
  }

  function init(options) {
    var settings = options || {};
    var root = settings.root || document;
    var article = root.querySelector(".wiki-article");
    var shell = root.querySelector(".wiki-shell");
    if (!article || !shell) {
      return;
    }

    var indexUrl = settings.indexUrl || shell.getAttribute("data-wiki-index");
    if (!indexUrl) {
      return;
    }

    var card = root.createElement("div");
    card.className = "wiki-link-preview";
    card.hidden = true;
    card.setAttribute("role", "tooltip");
    root.body.appendChild(card);

    var hideTimer = 0;
    var showTimer = 0;
    var active = null;
    var entries = [];

    function hide() {
      window.clearTimeout(showTimer);
      card.hidden = true;
      if (active) {
        active.removeAttribute("aria-describedby");
      }
      active = null;
    }

    function show(anchor, preview) {
      active = anchor;
      card.id = "wiki-link-preview";
      renderCard(card, preview);
      card.hidden = false;
      placeCard(card, anchor);
      anchor.setAttribute("aria-describedby", "wiki-link-preview");
    }

    function previewFrom(anchor) {
      return previewForLink(entries, anchor.getAttribute("href") || "", settings.origin || window.location.origin);
    }

    function scheduleShow(anchor) {
      window.clearTimeout(hideTimer);
      window.clearTimeout(showTimer);
      showTimer = window.setTimeout(function() {
        var preview = previewFrom(anchor);
        if (preview) {
          show(anchor, preview);
        }
      }, 160);
    }

    function scheduleHide() {
      window.clearTimeout(showTimer);
      hideTimer = window.setTimeout(hide, 80);
    }

    fetch(indexUrl).then(function(response) {
      if (!response.ok) {
        throw new Error("Wiki preview index failed with " + response.status);
      }
      return response.json();
    }).then(function(loaded) {
      entries = loaded;
    }).catch(function(error) {
      console.error("Unable to load Wiki preview index:", error);
    });

    var scopes = [article.querySelector(".page__content"), article.querySelector(".wiki-article__sources")];
    scopes.forEach(function(scope) {
      if (!scope) {
        return;
      }
      scope.addEventListener("mouseover", function(event) {
        if (!prefersFineHover()) {
          return;
        }
        var anchor = event.target.closest("a[href]");
        if (!anchor || shouldSkip(anchor) || !scope.contains(anchor)) {
          return;
        }
        scheduleShow(anchor);
      });
      scope.addEventListener("mouseout", function(event) {
        var anchor = event.target.closest("a[href]");
        if (!anchor || !scope.contains(anchor)) {
          return;
        }
        if (event.relatedTarget && anchor.contains(event.relatedTarget)) {
          return;
        }
        scheduleHide();
      });
      scope.addEventListener("focusin", function(event) {
        var anchor = event.target.closest("a[href]");
        if (!anchor || shouldSkip(anchor) || !scope.contains(anchor)) {
          return;
        }
        var preview = previewFrom(anchor);
        if (preview) {
          window.clearTimeout(hideTimer);
          show(anchor, preview);
        }
      });
      scope.addEventListener("focusout", function() {
        scheduleHide();
      });
    });

    window.addEventListener("scroll", hide, true);
    window.addEventListener("resize", hide);
  }

  if (typeof document !== "undefined") {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", function() { init(); });
    } else {
      init();
    }
  }

  return {
    findEntry: findEntry,
    hostLabel: hostLabel,
    parseHref: parseHref,
    previewForLink: previewForLink
  };
}));
