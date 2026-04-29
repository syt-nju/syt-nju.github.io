---
title: "Visualized Reading"
permalink: /visual-reading/
author_profile: true
---

{% include base_path %}

I turn papers and blogs I am actively reading into single-file, self-contained interactive HTML pages. Each entry below opens a standalone visualization; it is my way of forcing myself to restate what I just read in a form that someone else could also navigate.

The exact Cursor agent command I use to produce these pages lives at [blog-to-html](https://github.com/syt-nju/my_cursor/blob/main/.agent/commands/blog-to-html.md).

{% assign entries = site.data.visual_reading.entries | sort: "date" | reverse %}

{% if entries == nil or entries == empty %}

*No entries yet. New visualizations will show up here.*

{% else %}

{% for item in entries %}

### [{{ item.title }}]({{ base_path }}/files/visual-reading/{{ item.slug }}/)

<p class="page__meta">
  <time datetime="{{ item.date }}">{{ item.date | date: "%b %-d, %Y" }}</time>
  {% if item.source_type %} · <span>{{ item.source_type }}</span>{% endif %}
  {% if item.source_url %} · <a href="{{ item.source_url }}">original</a>{% endif %}
</p>

{% endfor %}

{% endif %}
