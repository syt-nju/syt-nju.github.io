const assert = require("assert");
const WikiSearch = require("../assets/js/wiki-search.js");

const entries = [
  {
    title: "深度搜索概览",
    summary: "表观结构复杂度不等于实现搜索难度。",
    content: "实现难度由最便宜的 identifying route 决定。",
    lang: "zh-CN",
    url: "/wiki/deep-search/overview/"
  },
  {
    title: "Skill Library",
    summary: "技能库是冻结 LLM 的外部程序性记忆。",
    content: "Voyager uses executable code skills in a vector library.",
    lang: "zh-CN",
    url: "/wiki/harness-evolution/skill-library/"
  }
];

assert.deepStrictEqual(WikiSearch.search(entries, ""), []);
assert.strictEqual(WikiSearch.search(entries, "identifying")[0].title, "深度搜索概览");
assert.strictEqual(WikiSearch.search(entries, "Voyager")[0].title, "Skill Library");
assert.deepStrictEqual(WikiSearch.search(entries, "missing"), []);
assert.strictEqual(WikiSearch.filterByLanguage(entries, "en").length, 0);
assert.strictEqual(WikiSearch.filterByLanguage(entries, "zh-CN").length, 2);
assert.strictEqual(WikiSearch.filterByLanguage(entries, "all").length, 2);

const snippet = WikiSearch.createSnippet(entries[0], "identifying");
assert.ok(snippet.toLowerCase().includes("identifying"));

const summarySnippet = WikiSearch.createSnippet(entries[1], "程序性记忆");
assert.ok(summarySnippet.includes("程序性记忆"));

console.log("wiki-search tests passed");
