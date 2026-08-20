(function() {
  var nav = document.querySelector(".wiki-sidebar");
  var article = document.querySelector(".wiki-article .page__content");
  if (!nav || !article) {
    return;
  }

  var links = Array.prototype.slice.call(nav.querySelectorAll(".wiki-sidebar__toc a[href*='#']"));
  if (!links.length) {
    return;
  }

  function headingId(link) {
    var href = link.getAttribute("href") || "";
    var hash = href.split("#")[1] || "";
    try {
      return decodeURIComponent(hash);
    } catch (error) {
      return hash;
    }
  }

  function escapeSelector(id) {
    if (window.CSS && typeof CSS.escape === "function") {
      return CSS.escape(id);
    }
    return id.replace(/([^a-zA-Z0-9_-])/g, "\\$1");
  }

  var headings = links
    .map(function(link) {
      return article.querySelector("#" + escapeSelector(headingId(link)));
    })
    .filter(Boolean);

  function setActive(id) {
    links.forEach(function(link) {
      var active = headingId(link) === id;
      link.classList.toggle("is-active", active);
      if (active) {
        link.setAttribute("aria-current", "location");
      } else {
        link.removeAttribute("aria-current");
      }
    });
  }

  function syncFromHash() {
    var id = decodeURIComponent((location.hash || "").replace(/^#/, ""));
    if (id) {
      setActive(id);
    }
  }

  links.forEach(function(link) {
    link.addEventListener("click", function() {
      setActive(headingId(link));
    });
  });

  if (!("IntersectionObserver" in window) || !headings.length) {
    syncFromHash();
    return;
  }

  var visible = {};
  var observer = new IntersectionObserver(
    function(entries) {
      entries.forEach(function(entry) {
        visible[entry.target.id] = entry.isIntersecting && entry.intersectionRatio > 0;
      });
      var current = headings.find(function(heading) {
        return visible[heading.id];
      });
      if (current) {
        setActive(current.id);
      }
    },
    {
      rootMargin: "0px 0px -65% 0px",
      threshold: [0, 1]
    }
  );

  headings.forEach(function(heading) {
    observer.observe(heading);
  });
  syncFromHash();
})();
