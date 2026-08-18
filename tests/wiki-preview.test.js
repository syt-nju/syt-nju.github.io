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

function makeNode(tagName, text, next) {
  return {
    tagName: tagName,
    innerText: text,
    textContent: text,
    nextElementSibling: next || null,
    parentElement: null
  };
}

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
assert.strictEqual(internal.hash, "exposure-bias");
assert.strictEqual(
  WikiPreview.previewForLink(entries, "/wiki/on-policy-distillation/sft-rl-opd/", origin),
  null
);

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

assert.strictEqual(
  WikiPreview.clipExcerpt("短句。"),
  "短句。"
);
assert.strictEqual(
  WikiPreview.clipExcerpt("第一句。第二句还很长。", 8),
  "第一句。"
);

const afterTable = makeNode("P", "两个 OPD 学生几乎一样，遗忘都轻于 SFT teacher。");
const table = makeNode("TABLE", "SFT teacher 0.775", afterTable);
const setup = makeNode("P", "最小代码编辑任务上，nrehiew 先分别 SFT / RL 出两个老师。", table);
const heading = makeNode("H2", "老师可以 SFT overtrain，学生仍少忘", setup);
const nextHeading = makeNode("H2", "算力证据");
afterTable.nextElementSibling = nextHeading;

const excerpt = WikiPreview.collectSectionText(heading, 480);
assert.ok(excerpt.indexOf("最小代码编辑") !== -1);
assert.ok(excerpt.indexOf("两个 OPD 学生几乎一样") !== -1);
assert.ok(excerpt.indexOf("0.775") === -1);
assert.ok(excerpt.indexOf("算力证据") === -1);

const extracted = WikiPreview.extractFromDocument(
  {
    getElementById: function(id) {
      return id === "sft-teacher-opd" ? heading : null;
    }
  },
  "sft-teacher-opd"
);
assert.strictEqual(extracted.title, "老师可以 SFT overtrain，学生仍少忘");
assert.ok(extracted.summary.indexOf("两个 OPD 学生几乎一样") !== -1);
assert.strictEqual(WikiPreview.extractFromDocument({ getElementById: function() { return null; } }, "missing"), null);

const sparseCreditText = "RL 在学生自己的 rollout 上学习，因此能直接惩罚自己会犯的错，状态是对的。代价是每条轨迹通常只有对错这一类稀疏信号。";
const sparseCreditHeading = makeNode("H2", "Sparse credit", makeNode("P", sparseCreditText, makeNode("H2", "OPD：学生走，老师评")));
const exposureBiasText = "SFT / 普通蒸馏在老师或数据的轨迹上做 next-token 学习。训练时每一步的前缀几乎都是正确的那条路。";
const exposureBiasHeading = makeNode("H2", "Exposure bias", makeNode("P", exposureBiasText, sparseCreditHeading));
const sectionDoc = {
  getElementById: function(id) {
    if (id === "exposure-bias") {
      return exposureBiasHeading;
    }
    if (id === "sparse-credit") {
      return sparseCreditHeading;
    }
    return null;
  }
};
assert.strictEqual(
  WikiPreview.extractFromDocument(sectionDoc, "exposure-bias").summary,
  exposureBiasText
);
assert.strictEqual(
  WikiPreview.extractFromDocument(sectionDoc, "sparse-credit").summary,
  sparseCreditText
);

console.log("wiki-preview tests passed");
