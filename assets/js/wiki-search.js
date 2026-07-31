(function(root, factory) {
  var api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  } else {
    root.WikiSearch = api;
  }
}(this, function() {
  "use strict";

  function normalize(value) {
    return String(value || "").toLowerCase();
  }

  function tokensFor(query) {
    return normalize(query).trim().split(/\s+/).filter(Boolean);
  }

  function search(entries, query) {
    var tokens = tokensFor(query);
    if (!tokens.length) {
      return [];
    }

    return entries.map(function(entry) {
      var title = normalize(entry.title);
      var summary = normalize(entry.summary);
      var content = normalize(entry.content);
      var haystack = title + " " + summary + " " + content;

      if (!tokens.every(function(token) { return haystack.indexOf(token) !== -1; })) {
        return null;
      }

      var score = tokens.reduce(function(total, token) {
        return total +
          (title.indexOf(token) !== -1 ? 3 : 0) +
          (summary.indexOf(token) !== -1 ? 2 : 0) +
          (content.indexOf(token) !== -1 ? 1 : 0);
      }, 0);

      return { entry: entry, score: score };
    }).filter(Boolean).sort(function(left, right) {
      return right.score - left.score || left.entry.title.localeCompare(right.entry.title);
    }).map(function(result) {
      return result.entry;
    });
  }

  function createSnippet(entry, query) {
    var tokens = tokensFor(query);
    var candidates = [entry.summary, entry.content].map(function(value) {
      var text = String(value || "").replace(/\s+/g, " ").trim();
      var matches = tokens.reduce(function(total, token) {
        return total + (normalize(text).indexOf(token) !== -1 ? 1 : 0);
      }, 0);
      return { text: text, matches: matches };
    }).sort(function(left, right) {
      return right.matches - left.matches;
    });
    var text = candidates[0].text;
    var firstMatch = -1;

    tokens.forEach(function(token) {
      var index = normalize(text).indexOf(token);
      if (index !== -1 && (firstMatch === -1 || index < firstMatch)) {
        firstMatch = index;
      }
    });

    var start = Math.max(0, firstMatch === -1 ? 0 : firstMatch - 70);
    var end = Math.min(text.length, start + 180);
    return (start > 0 ? "…" : "") + text.slice(start, end) + (end < text.length ? "…" : "");
  }

  function escapeRegExp(value) {
    return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  function appendHighlighted(element, text, query) {
    var tokens = tokensFor(query).sort(function(a, b) { return b.length - a.length; });
    if (!tokens.length) {
      element.textContent = text;
      return;
    }

    var pattern = new RegExp("(" + tokens.map(escapeRegExp).join("|") + ")", "gi");
    String(text).split(pattern).forEach(function(part) {
      if (tokens.indexOf(normalize(part)) !== -1) {
        var mark = document.createElement("mark");
        mark.textContent = part;
        element.appendChild(mark);
      } else {
        element.appendChild(document.createTextNode(part));
      }
    });
  }

  function renderResults(container, status, entries, query) {
    container.textContent = "";
    var allResults = search(entries, query);
    var results = allResults.slice(0, 20);
    if (allResults.length > results.length) {
      status.textContent = "Showing " + results.length + " of " + allResults.length + " results";
    } else {
      status.textContent = results.length ? results.length + " result" + (results.length === 1 ? "" : "s") : "No matching articles";
    }

    results.forEach(function(entry) {
      var link = document.createElement("a");
      var title = document.createElement("strong");
      var snippet = document.createElement("span");
      link.className = "wiki-search-result";
      link.href = entry.url;
      appendHighlighted(title, entry.title, query);
      appendHighlighted(snippet, createSnippet(entry, query), query);
      link.appendChild(title);
      link.appendChild(snippet);
      container.appendChild(link);
    });
  }

  function init() {
    var searchBox = document.querySelector(".wiki-search");
    var input = document.getElementById("wiki-search-input");
    var status = document.getElementById("wiki-search-status");
    var results = document.getElementById("wiki-search-results");
    if (!searchBox || !input || !status || !results) {
      return;
    }

    fetch(searchBox.getAttribute("data-index-url")).then(function(response) {
      if (!response.ok) {
        throw new Error("Search index request failed with " + response.status);
      }
      return response.json();
    }).then(function(entries) {
      input.addEventListener("input", function() {
        var query = input.value.trim();
        if (!query) {
          status.textContent = "";
          results.textContent = "";
          return;
        }
        renderResults(results, status, entries, query);
      });
    }).catch(function(error) {
      console.error("Unable to load Wiki search index:", error);
      status.textContent = "Search is temporarily unavailable.";
    });
  }

  if (typeof document !== "undefined") {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", init);
    } else {
      init();
    }
  }

  return {
    createSnippet: createSnippet,
    search: search
  };
}));
