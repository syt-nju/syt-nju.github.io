---
title: "Reading List"
permalink: /reading-list/
author_profile: true
---

A running collection of original blogs and articles I found worth keeping. Each card links straight to the source.

{% assign entries = site.data.reading_list.entries | sort: "date" | reverse %}

{% if entries == nil or entries == empty %}

*No entries yet.*

{% else %}

<div class="reading-list">
{% for item in entries %}
  {% assign host = item.url | split: "://" | last | split: "/" | first | remove_first: "www." %}
  <a class="reading-card" href="{{ item.url }}" target="_blank" rel="noopener noreferrer">
    <img class="reading-card__icon" src="https://www.google.com/s2/favicons?domain={{ host }}&sz=64" alt="" loading="lazy" width="32" height="32">
    <span class="reading-card__body">
      <span class="reading-card__title">{{ item.title }}</span>
      <span class="reading-card__host">{{ host }}</span>
    </span>
  </a>
{% endfor %}
</div>

{% endif %}
