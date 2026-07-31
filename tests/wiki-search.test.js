const assert = require("assert");
const WikiSearch = require("../assets/js/wiki-search.js");

const entries = [
  {
    title: "Online RFT vs RLVR",
    summary: "Why negative samples matter.",
    content: "RLVR preserves exploration through negative feedback.",
    url: "/wiki/agent-training/online-rft-vs-rlvr/"
  },
  {
    title: "GSPO objective",
    summary: "Sequence-level clipping for MoE training.",
    content: "Routing replay changes token-level importance ratios.",
    url: "/wiki/agent-training/gspo-objective/"
  }
];

assert.deepStrictEqual(WikiSearch.search(entries, ""), []);
assert.strictEqual(WikiSearch.search(entries, "negative")[0].title, "Online RFT vs RLVR");
assert.strictEqual(WikiSearch.search(entries, "routing replay")[0].title, "GSPO objective");
assert.deepStrictEqual(WikiSearch.search(entries, "missing"), []);

const snippet = WikiSearch.createSnippet(entries[0], "exploration");
assert.ok(snippet.toLowerCase().includes("exploration"));

const summarySnippet = WikiSearch.createSnippet(entries[1], "sequence-level");
assert.ok(summarySnippet.toLowerCase().includes("sequence-level"));

console.log("wiki-search tests passed");
