let language = "en",
  manifest,
  activeCase,
  busy = false,
  lastScores = null,
  lastError = null;
const $ = (id) => document.getElementById(id);
const t = (en, zh) => (language === "zh" ? zh : en);
const labels = {
  ACK: ["Actinic keratosis", "日光性角化病"],
  BCC: ["Basal cell carcinoma", "基底细胞癌"],
  MEL: ["Melanoma", "黑色素瘤"],
  NEV: ["Nevus", "痣"],
  SCC: ["Squamous cell carcinoma", "鳞状细胞癌"],
  SEK: ["Seborrheic keratosis", "脂溢性角化病"],
};
const fields = {
  age: ["Age", "年龄"],
  fitspatrick: ["Fitzpatrick skin type", "Fitzpatrick 肤色分型"],
  diameter_1: ["Lesion width (mm)", "病变横径（毫米）"],
  diameter_2: ["Lesion height (mm)", "病变纵径（毫米）"],
  gender: ["Recorded sex", "记录的性别"],
  region: ["Body site", "身体部位"],
  skin_cancer_history: [
    "Personal/family skin-cancer history",
    "本人或家人皮肤癌史",
  ],
  cancer_history: ["Personal/family cancer history", "本人或家人癌症史"],
  itch: ["Itching", "瘙痒"],
  grew: ["Recent growth", "近期长大"],
  hurt: ["Pain", "疼痛"],
  changed: ["Recent change", "近期变化"],
  bleed: ["Bleeding", "出血"],
  elevation: ["Raised surface", "表面隆起"],
};
const sites = {
  ARM: "手臂",
  BACK: "背部",
  CHEST: "胸部",
  ABDOMEN: "腹部",
  FACE: "面部",
  THIGH: "大腿",
  FOREARM: "前臂",
  HAND: "手",
  SCALP: "头皮",
  NECK: "颈部",
  EAR: "耳",
  FOOT: "脚",
  LIP: "唇",
  NOSE: "鼻部",
};
const methods = {
  clinical_linear: [
    "Clinical record only",
    "只看临床资料",
    "Linear baseline",
    "线性基线",
  ],
  image_linear: [
    "Photo only",
    "只看照片",
    "Frozen ResNet18 + linear head",
    "冻结 ResNet18＋线性预测层",
  ],
  concat_linear: [
    "Photo + clinical record",
    "照片＋临床资料",
    "Feature concatenation + linear head",
    "特征拼接＋线性预测层",
  ],
  metablock: [
    "MetaBlock fusion",
    "MetaBlock 融合",
    "Clinical information gates image features",
    "临床信息调节图像特征",
  ],
  healnet: [
    "HEALNet fusion",
    "HEALNet 融合",
    "Reduced multimodal attention model",
    "缩小的多模态注意力模型",
  ],
};
const labelName = (key) => labels[key]?.[language === "zh" ? 1 : 0] || key;
function valueText(key, value) {
  if (value === null || value === "UNK") return t("Not recorded", "未记录");
  if (value === "True" || value === "TRUE") return t("Yes", "是");
  if (value === "False" || value === "FALSE") return t("No", "否");
  if (language === "zh" && key === "region") return sites[value] || value;
  if (value === "MALE") return t("Male", "男");
  if (value === "FEMALE") return t("Female", "女");
  return value;
}
function fieldList(target, keys) {
  $(target).replaceChildren(
    ...keys.map((key) => {
      const row = document.createElement("div"),
        dt = document.createElement("dt"),
        dd = document.createElement("dd");
      dt.textContent = fields[key][language === "zh" ? 1 : 0];
      dd.textContent = valueText(key, activeCase.fields[key]);
      row.append(dt, dd);
      return row;
    }),
  );
}
function showCase() {
  activeCase = manifest.cases[Number($("case-select").value) || 0];
  lastScores = null;
  lastError = null;
  $("prediction").hidden = true;
  $("case-image").src = activeCase.image;
  $("case-image").alt =
    t("Actual public clinical photograph for ", "公开临床照片：") +
    activeCase.id;
  $("record-id").textContent = activeCase.id;
  fieldList("clinical-fields", ["age", "region", "diameter_1", "itch", "grew"]);
  fieldList("all-fields", Object.keys(fields));
  $("run-status").textContent = t(
    "First run downloads the image encoder (about 45 MB). Computation stays in this browser.",
    "首次运行下载约 45 MB 的图像编码器。计算在当前浏览器内进行。",
  );
}
function showResults(scores) {
  const cards = $("prediction-cards"),
    advanced = $("advanced-prediction-cards");
  cards.replaceChildren();
  advanced.replaceChildren();
  for (const name of [
    "image_linear",
    "clinical_linear",
    "concat_linear",
    "metablock",
    "healnet",
  ]) {
    const row = methods[name],
      prob = scores[name],
      order = prob.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]);
    const card = document.createElement("article");
    card.className =
      "prediction" + (name === "concat_linear" ? " combined" : "");
    const heading = document.createElement("h4");
    heading.textContent = row[language === "zh" ? 1 : 0];
    const method = document.createElement("p");
    method.className = "method";
    method.textContent = row[language === "zh" ? 3 : 2];
    const top = document.createElement("p");
    top.className = "top";
    top.textContent = labelName(manifest.classes[order[0][1]]);
    card.append(heading, method, top);
    for (const [p, i] of order) {
      const line = document.createElement("div");
      line.className = "prob-row";
      const cap = document.createElement("div");
      cap.className = "prob-label";
      const label = document.createElement("span");
      label.textContent = labelName(manifest.classes[i]);
      const number = document.createElement("span");
      number.textContent = (p * 100).toFixed(1) + "%";
      cap.append(label, number);
      const track = document.createElement("div");
      track.className = "bar-track";
      const bar = document.createElement("div");
      bar.className = "bar-fill";
      bar.style.width = p * 100 + "%";
      track.append(bar);
      line.append(cap, track);
      card.append(line);
    }
    (["metablock", "healnet"].includes(name) ? advanced : cards).append(card);
  }
  $("record-label").textContent = labelName(activeCase.published_label);
  $("example-split").textContent = t(
    "Held-out example · selected by ID, not prediction quality",
    "保留测试样本 · 按编号选取，不挑预测正确的例子",
  );
  $("prediction").hidden = false;
}
function percent(v) {
  return (v * 100).toFixed(1) + "%";
}
function renderMetrics() {
  const results = manifest.test,
    u = results.uncertainty;
  const delta =
    results.mean_macro_f1.concat_linear - results.mean_macro_f1.image_linear;
  const interval = u.primary_difference_interval;
  const claim =
    interval[0] > 0
      ? t("The paired interval is above zero.", "配对重采样区间高于零。")
      : interval[1] < 0
        ? t("The paired interval is below zero.", "配对重采样区间低于零。")
        : t(
            "The interval includes zero, so this experiment does not establish a clear improvement.",
            "区间包含零，本实验不能证明明确改善。",
          );
  $("results-summary").textContent = t(
    `Adding clinical fields to the same linear model changed macro-F1 by ${(delta * 100).toFixed(1)} percentage points (95% interval ${(interval[0] * 100).toFixed(1)} to ${(interval[1] * 100).toFixed(1)}). ${claim}`,
    `在同类线性模型中加入临床字段，Macro-F1 改变 ${(delta * 100).toFixed(1)} 个百分点（95% 区间：${(interval[0] * 100).toFixed(1)} 至 ${(interval[1] * 100).toFixed(1)}）。${claim}`,
  );
  $("metrics").replaceChildren(
    ...Object.keys(methods).map((name) => {
      const tr = document.createElement("tr");
      for (const text of [
        methods[name][language === "zh" ? 1 : 0],
        percent(results.mean_macro_f1[name]),
        u.macro_f1_intervals[name].map(percent).join(" – "),
      ]) {
        const td = document.createElement("td");
        td.textContent = text;
        tr.append(td);
      }
      return tr;
    }),
  );
  $("training-details").textContent = t(
    `Fit: ${manifest.counts.fit.images} images; inner tuning: ${manifest.counts.tune.images}; validation: ${manifest.counts.validation.images}; held-out test: ${manifest.counts.test.images}. Shared patient IDs, ambiguous lesion IDs and exact image duplicates stay together. Gold dotted lines mark the restored checkpoint.`,
    `拟合 ${manifest.counts.fit.images} 张，内部调参 ${manifest.counts.tune.images} 张，验证 ${manifest.counts.validation.images} 张，保留测试 ${manifest.counts.test.images} 张。共享患者编号、含糊的病变编号及精确重复照片不跨分区。金色虚线为恢复的检查点。`,
  );
  $("model-evidence").textContent = JSON.stringify(
    {
      version: manifest.version,
      source: manifest.source_url,
      display_seed: manifest.seed,
      selection_sha256: manifest.selection_sha256,
      clinical_validation: false,
    },
    null,
    2,
  );
  if (manifest.source_repo) $("github").href = manifest.source_repo;
}
function translate() {
  document.documentElement.lang = language === "zh" ? "zh-CN" : "en";
  document.querySelectorAll("[data-en]").forEach((node) => {
    if (node.tagName === "H1") node.innerHTML = node.dataset[language];
    else node.textContent = node.dataset[language];
  });
  $("language").textContent = language === "en" ? "中文讲解" : "English";
  if (manifest) {
    renderMetrics();
    $("load-status").textContent = t(
      "Real paired research records · PAD-UFES-20 · CC BY 4.0",
      "真实配对研究记录 · PAD-UFES-20 · CC BY 4.0",
    );
    if (activeCase) {
      fieldList("clinical-fields", [
        "age",
        "region",
        "diameter_1",
        "itch",
        "grew",
      ]);
      fieldList("all-fields", Object.keys(fields));
      if (lastScores) showResults(lastScores);
      if (!busy && lastError) {
        $("run-status").textContent =
          t(
            "Model execution failed. Reload this page and try again. Details: ",
            "模型运行失败。请刷新页面后重试。详情：",
          ) + lastError;
      } else if (!busy && lastScores && window.__lastInference) {
        const seconds = (window.__lastInference.elapsed_ms / 1000).toFixed(1);
        $("run-status").textContent = t(
          `Computed in ${seconds} seconds · actual ONNX model inference · seed 17`,
          `用时 ${seconds} 秒 · 实际 ONNX 模型计算 · 种子 17`,
        );
      } else if (!busy) {
        $("run-status").textContent = t(
          "First run downloads the image encoder (about 45 MB). Computation stays in this browser.",
          "首次运行下载约 45 MB 的图像编码器。计算在当前浏览器内进行。",
        );
      }
    }
  }
}
$("language").onclick = () => {
  language = language === "en" ? "zh" : "en";
  translate();
};
const sessions = new Map();
async function session(name, file) {
  if (!sessions.has(name))
    sessions.set(
      name,
      await ort.InferenceSession.create("assets/" + file, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      }),
    );
  return sessions.get(name);
}
function pixels() {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 224;
  canvas.getContext("2d").drawImage($("case-image"), 0, 0, 224, 224);
  const rgba = canvas.getContext("2d").getImageData(0, 0, 224, 224).data;
  const result = new Float32Array(3 * 224 * 224),
    mean = [0.485, 0.456, 0.406],
    std = [0.229, 0.224, 0.225];
  for (let c = 0; c < 3; c++)
    for (let i = 0; i < 224 * 224; i++)
      result[c * 224 * 224 + i] = (rgba[i * 4 + c] / 255 - mean[c]) / std[c];
  return new ort.Tensor("float32", result, [1, 3, 224, 224]);
}
const softmax = (values) => {
  const max = Math.max(...values),
    exp = Array.from(values, (v) => Math.exp(v - max)),
    sum = exp.reduce((a, b) => a + b, 0);
  return exp.map((v) => v / sum);
};
$("run").onclick = async () => {
  if (busy) return;
  busy = true;
  lastScores = null;
  lastError = null;
  window.__lastInference = null;
  $("run").disabled = true;
  $("case-select").disabled = true;
  $("prediction").hidden = true;
  const started = performance.now();
  try {
    $("run-status").textContent = t(
      "Loading model weights. The first run may take a moment…",
      "正在加载模型权重，首次运行需要一些时间……",
    );
    await $("case-image").decode();
    const encoder = await session("encoder", manifest.encoder.file);
    $("run-status").textContent = t(
      "Reading the photo with ResNet18…",
      "正在用 ResNet18 提取照片特征……",
    );
    const image = await encoder.run({ pixels: pixels() });
    const scores = {};
    for (const [name, model] of Object.entries(manifest.models)) {
      $("run-status").textContent =
        t("Computing: ", "正在计算：") +
        methods[name][language === "zh" ? 1 : 0];
      const head = await session(name, model.file),
        all = {
          image_features: image.image_features,
          clinical: new ort.Tensor(
            "float32",
            new Float32Array(activeCase.clinical),
            [1, activeCase.clinical.length],
          ),
        },
        feeds = {};
      head.inputNames.forEach((key) => (feeds[key] = all[key]));
      const out = await head.run(feeds);
      scores[name] = softmax(out.logits.data);
    }
    lastScores = scores;
    showResults(scores);
    window.__lastInference = {
      case_id: activeCase.id,
      scores,
      elapsed_ms: performance.now() - started,
    };
    $("run-status").textContent = t(
      `Computed in ${((performance.now() - started) / 1000).toFixed(1)} seconds · actual ONNX model inference · seed 17`,
      `用时 ${((performance.now() - started) / 1000).toFixed(1)} 秒 · 实际 ONNX 模型计算 · 种子 17`,
    );
  } catch (error) {
    console.error(error);
    lastError = String(error.message || error);
    $("run-status").textContent =
      t(
        "Model execution failed. Reload this page and try again. Details: ",
        "模型运行失败。请刷新页面后重试。详情：",
      ) + String(error.message || error);
  } finally {
    busy = false;
    $("run").disabled = false;
    $("case-select").disabled = false;
  }
};
translate();
try {
  if (!window.ort) throw new Error("ONNX runtime is unavailable");
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = new URL("runtime/", location.href).href;
  const response = await fetch("assets/manifest.json");
  if (!response.ok)
    throw new Error("Public model release is not available yet");
  manifest = await response.json();
  $("case-select").replaceChildren(
    ...manifest.cases.map((c, i) => {
      const o = document.createElement("option");
      o.value = i;
      o.textContent = c.id.replace(".png", "");
      return o;
    }),
  );
  $("case-select").onchange = showCase;
  showCase();
  renderMetrics();
  $("load-status").textContent = t(
    "Real paired research records · PAD-UFES-20 · CC BY 4.0",
    "真实配对研究记录 · PAD-UFES-20 · CC BY 4.0",
  );
  $("run").disabled = false;
  window.__manifest = manifest;
} catch (error) {
  $("load-status").textContent =
    t("Unable to load the demo: ", "演示暂时无法加载：") + error.message;
}
