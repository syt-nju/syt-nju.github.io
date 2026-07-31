---
title: "知识库"
permalink: /wiki/
layout: wiki
author_profile: false
lang: zh-CN
summary: "从原始资料中持续整理、可复用的个人知识库。"
---

<div class="wiki-toolbar">
  <div class="wiki-search" role="search" data-index-url="{{ '/wiki/search.json' | relative_url }}">
    <label for="wiki-search-input">搜索知识库</label>
    <div class="wiki-search__input-wrap">
      <span aria-hidden="true">⌕</span>
      <input id="wiki-search-input" type="search" placeholder="搜索标题、摘要和正文…" autocomplete="off">
    </div>
    <p id="wiki-search-status" class="wiki-search__status" aria-live="polite"></p>
    <div id="wiki-search-results" class="wiki-search__results"></div>
  </div>
  <div class="wiki-language-filter" role="group" aria-label="按文章语言筛选">
    <span>语言</span>
    <button type="button" data-wiki-lang="all" aria-pressed="true">全部</button>
    <button type="button" data-wiki-lang="zh-CN" aria-pressed="false">中文</button>
    <button type="button" data-wiki-lang="en" aria-pressed="false">English</button>
  </div>
</div>

<p id="wiki-filter-empty" class="wiki-filter-empty" hidden>当前语言下暂无文章。</p>

{% assign topics = site.data.wiki_topics | sort: "order" %}
{% for topic in topics %}
  {% assign articles = site.wiki | where: "topic", topic.slug | sort: "order" %}
  <section class="wiki-topic" data-wiki-topic>
    <header class="wiki-topic__header">
      <div>
        <p class="wiki-topic__eyebrow">主题</p>
        <h2>{{ topic.title | escape }}</h2>
      </div>
      <span class="wiki-topic__count" data-topic-count>{{ articles.size }} 篇</span>
    </header>
    <p class="wiki-topic__description">{{ topic.description | escape }}</p>
    {% if articles == empty %}
      <p>暂无文章。</p>
    {% else %}
      <div class="wiki-card-grid">
        {% for article in articles %}
          <a class="wiki-card" href="{{ article.url | relative_url }}" data-wiki-lang="{{ article.lang }}">
            <div class="wiki-card__meta">
              <span class="wiki-language-badge">{% if article.lang == "zh-CN" %}中文{% else %}English{% endif %}</span>
              <time datetime="{{ article.updated | date: '%Y-%m-%d' }}">{{ article.updated | date: "%Y-%m-%d" }}</time>
            </div>
            <h3>{{ article.title | escape }}</h3>
            <p>{{ article.summary | escape }}</p>
            <span class="wiki-card__action">阅读文章 <span aria-hidden="true">→</span></span>
          </a>
        {% endfor %}
      </div>
    {% endif %}
  </section>
{% endfor %}
