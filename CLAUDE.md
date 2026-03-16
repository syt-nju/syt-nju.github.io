# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是基于 [academicpages](https://github.com/academicpages/academicpages.github.io) 模板构建的 Jekyll 学术团队网站（SocratesClub），使用 Minimal Mistakes 主题，部署在 GitHub Pages 上。

## 常用命令

```bash
# 安装 Ruby 依赖
bundle install

# 本地预览（访问 http://localhost:4000）
bundle exec jekyll serve

# 构建静态站点（输出到 _site/）
bundle exec jekyll build

# 优化 JavaScript 资源（需要 Node.js）
npm run build:js
```

> 注意：修改 `_config.yml` 后需要重启 `jekyll serve`，该文件不会被热更新。

## 内容架构

内容通过 Jekyll **集合（collections）** 和 YAML front matter 管理，各集合对应目录如下：

| 集合 | 目录 | URL 规则 |
|---|---|---|
| 学术发表 | `_publications/` | `/publications/:path/` |
| 演讲报告 | `_talks/` | `/talks/:path/` |
| 教学内容 | `_teaching/` | `/teaching/:path/` |
| 项目作品 | `_portfolio/` | `/portfolio/:path/` |
| 博客文章 | `_posts/` | `/:categories/:title/` |
| 静态页面 | `_pages/` | （每页 front matter 单独设置 permalink） |

## 关键配置

- **`_config.yml`**：全站配置，包括站点标题、URL（`https://socratesgroup.github.io/`）、作者信息、分析统计、社交链接、集合与插件配置。
- **`_data/`**：结构化数据文件（导航菜单等）。
- **`_layouts/`** 和 **`_includes/`**：HTML 模板，组件级修改优先编辑 includes。
- **`assets/js/_main.js`**：JS 源文件（被 Jekyll 构建排除），编译产物为 `assets/js/main.min.js`。
- **`markdown_generator/`**：Python 脚本，可从 TSV/CSV 批量生成集合条目。

## Front Matter 默认值

`_config.yml` 的 `defaults` 节已设置默认值：文章默认开启 `author_profile`、`read_time`、`comments`、`share`；演讲条目使用 `talk` 布局（而非 `single`）。

## 新增内容方式

- **发表/演讲/教学条目**：在对应集合目录下新建 `.md` 文件，填写 `title`、`date`、`venue`、`paperurl`、`citation` 等 front matter。
- **静态页面**：在 `_pages/` 下新建，front matter 中设置 `permalink`（`_config.yml` 已将 `_pages/` 加入 include）。
- **图片**：放入 `images/`；PDF 等可下载文件放入 `files/`。
