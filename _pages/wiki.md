---
title: "Wiki"
permalink: /wiki/
layout: wiki
author_profile: false
summary: "A living knowledge base compiled from source material and maintained as reusable notes."
---

<div class="wiki-search" role="search" data-index-url="{{ '/wiki/search.json' | relative_url }}">
  <label for="wiki-search-input">Search the Wiki</label>
  <input id="wiki-search-input" type="search" placeholder="Search titles, summaries, and articles…" autocomplete="off">
  <p id="wiki-search-status" class="wiki-search__status" aria-live="polite"></p>
  <div id="wiki-search-results" class="wiki-search__results"></div>
</div>

{% assign topics = site.data.wiki_topics | sort: "order" %}
{% for topic in topics %}
  {% assign articles = site.wiki | where: "topic", topic.slug | sort: "order" %}
  <section class="wiki-topic">
    <h2>{{ topic.title | escape }}</h2>
    <p class="wiki-topic__description">{{ topic.description | escape }}</p>
    {% if articles == empty %}
      <p>No articles yet.</p>
    {% else %}
      {% for article in articles %}
        <a class="wiki-card" href="{{ article.url | relative_url }}">
          <strong>{{ article.title | escape }}</strong>
          <span>{{ article.summary | escape }}</span>
        </a>
      {% endfor %}
    {% endif %}
  </section>
{% endfor %}
