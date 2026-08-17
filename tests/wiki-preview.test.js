const assert = require("assert");
const WikiPreview = require("../assets/js/wiki-preview.js");

const origin = "https://syt-nju.github.io";
const entries = [
  {
    title: "SFT、RL 与 OPD",
    summary: "OPD 是学生自采样加上 dense 老师监督。",
    url: "/wiki/on-policy-distillation/sft-rl-opd/"
  }
];

const parsed = WikiPreview.parseHref("/wiki/on-policy-distillation/sft-rl-opd/#exposure-bias", origin);
assert.strictEqual(parsed.path, "/wiki/on-policy-distillation/sft-rl-opd/");
assert.strictEqual(parsed.hash, "exposure-bias");
assert.strictEqual(parsed.sameOrigin, true);

const internal = WikiPreview.previewForLink(
  entries,
  "/wiki/on-policy-distillation/sft-rl-opd/#exposure-bias",
  origin
);
assert.strictEqual(internal.kind, "知识库");
assert.strictEqual(internal.title, "SFT、RL 与 OPD");
assert.strictEqual(internal.summary, "OPD 是学生自采样加上 dense 老师监督。");
assert.strictEqual(internal.section, "exposure-bias");

const source = WikiPreview.previewForLink(
  entries,
  "https://thinkingmachines.ai/blog/on-policy-distillation/",
  origin
);
assert.strictEqual(source.kind, "原文");
assert.strictEqual(source.title, "Thinking Machines");
assert.ok(source.summary.indexOf("thinkingmachines.ai") !== -1);

assert.strictEqual(WikiPreview.hostLabel("arxiv.org"), "arXiv");
assert.strictEqual(
  WikiPreview.findEntry(entries, "/wiki/on-policy-distillation/sft-rl-opd").title,
  "SFT、RL 与 OPD"
);
assert.strictEqual(WikiPreview.previewForLink(entries, "mailto:hi@example.com", origin), null);

console.log("wiki-preview tests passed");
