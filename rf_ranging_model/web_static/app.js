"use strict";

const form = document.getElementById("scenarioForm");
const runButton = document.getElementById("runButton");
const message = document.getElementById("message");
const distance = document.getElementById("distance");
const distanceOutput = document.getElementById("distanceOutput");

const presets = {
  balanced: { distance_m: 8, direct_path_gain: 1, reflection_count: 2, reflection_gain: .55, reflection_delay_ns: 6, reflection_phase_rad: 2.55, snr_db: 28, phase_noise_deg: .5, cfo_hz: 20, uwb_frame_count: 4, base_group_delay_ns: .18, group_delay_ripple_ns: .08, amplitude_ripple_db: .5, calibration_mode: "frequency_profile", calibration_delay_ns: .18, calibration_ripple_ns: .08, calibration_amplitude_ripple_db: .5, seed: 7 },
  los: { distance_m: 8, direct_path_gain: 1, reflection_count: 0, reflection_gain: 0, reflection_delay_ns: 6, reflection_phase_rad: 0, snr_db: 60, phase_noise_deg: 0, cfo_hz: 0, uwb_frame_count: 4, base_group_delay_ns: 0, group_delay_ripple_ns: 0, amplitude_ripple_db: 0, calibration_mode: "none", calibration_delay_ns: 0, calibration_ripple_ns: 0, calibration_amplitude_ripple_db: 0, seed: 7 },
  multipath: { distance_m: 12, direct_path_gain: 1, reflection_count: 4, reflection_gain: 1.05, reflection_delay_ns: 3, reflection_phase_rad: 2.8, snr_db: 25, phase_noise_deg: 1.2, cfo_hz: 30, uwb_frame_count: 4, base_group_delay_ns: .35, group_delay_ripple_ns: .12, amplitude_ripple_db: .5, calibration_mode: "frequency_profile", calibration_delay_ns: .35, calibration_ripple_ns: .12, calibration_amplitude_ripple_db: .5, seed: 19 },
  noisy: { distance_m: 18, direct_path_gain: 1, reflection_count: 2, reflection_gain: .65, reflection_delay_ns: 8, reflection_phase_rad: -1.7, snr_db: 5, phase_noise_deg: 3, cfo_hz: 100, uwb_frame_count: 4, base_group_delay_ns: .4, group_delay_ripple_ns: .2, amplitude_ripple_db: .5, calibration_mode: "frequency_profile", calibration_delay_ns: .4, calibration_ripple_ns: .2, calibration_amplitude_ripple_db: .5, seed: 31 },
};

const examples = {
  los: {
    name: "理想 LOS",
    expected: "BLE 与 UWB 都应贴近 8 m 真值，路径图中只有一条 LOS。",
    values: presets.los,
  },
  ble_multipath: {
    name: "BLE 复杂多径",
    expected: "BLE 原始单径估计会被反射拉偏，Matrix Pencil/BIC 候选应明显更接近 12 m。",
    values: presets.multipath,
  },
  uwb_late_single: {
    name: "UWB 单帧反例",
    expected: "弱首径 0.1、强晚径 1.0 且晚 10 ns；单帧可能把约 11 m 的晚径误当作高置信距离。",
    values: { distance_m: 8, direct_path_gain: .1, reflection_count: 1, reflection_gain: 1, reflection_delay_ns: 10, reflection_phase_rad: 0, snr_db: 5, phase_noise_deg: 0, cfo_hz: 0, uwb_frame_count: 1, base_group_delay_ns: 0, group_delay_ripple_ns: 0, amplitude_ripple_db: 0, calibration_mode: "none", calibration_delay_ns: 0, calibration_ripple_ns: 0, calibration_amplitude_ripple_db: 0, seed: 0 },
  },
  uwb_late_four: {
    name: "UWB 四帧恢复",
    expected: "保持相同弱首径/强晚径信道，仅改为四帧相位对齐；距离应回到约 8 m，并保留 LOW_CONFIDENCE。",
    values: { distance_m: 8, direct_path_gain: .1, reflection_count: 1, reflection_gain: 1, reflection_delay_ns: 10, reflection_phase_rad: 0, snr_db: 5, phase_noise_deg: 0, cfo_hz: 0, uwb_frame_count: 4, base_group_delay_ns: 0, group_delay_ripple_ns: 0, amplitude_ripple_db: 0, calibration_mode: "none", calibration_delay_ns: 0, calibration_ripple_ns: 0, calibration_amplitude_ripple_db: 0, seed: 0 },
  },
  low_coherence: {
    name: "低帧相干度",
    expected: "四帧 coherence 应低于 0.75，状态机必须降为 LOW_CONFIDENCE，不能保持高置信 PASS。",
    values: { distance_m: 8, direct_path_gain: 1, reflection_count: 0, reflection_gain: 0, reflection_delay_ns: 10, reflection_phase_rad: 0, snr_db: -10, phase_noise_deg: 0, cfo_hz: 0, uwb_frame_count: 4, base_group_delay_ns: 0, group_delay_ripple_ns: 0, amplitude_ripple_db: 0, calibration_mode: "none", calibration_delay_ns: 0, calibration_ripple_ns: 0, calibration_amplitude_ripple_db: 0, seed: 41 },
  },
  bad_calibration: {
    name: "错误校准保护",
    expected: "0.5 m 场景错误减去 20 ns：BLE 应 REJECTED，UWB 应 NO_PATH，而不是显示负距离。",
    values: { distance_m: .5, direct_path_gain: 1, reflection_count: 0, reflection_gain: 0, reflection_delay_ns: 10, reflection_phase_rad: 0, snr_db: 120, phase_noise_deg: 0, cfo_hz: 0, uwb_frame_count: 4, base_group_delay_ns: 0, group_delay_ripple_ns: 0, amplitude_ripple_db: 0, calibration_mode: "scalar", calibration_delay_ns: 20, calibration_ripple_ns: 0, calibration_amplitude_ripple_db: 0, seed: 7 },
  },
};

let activeExample = null;

function setMessage(text = "") {
  message.textContent = text;
  message.classList.toggle("hidden", !text);
}

function applyValues(values) {
  for (const [key, value] of Object.entries(values)) {
    const input = form.elements.namedItem(key);
    if (!input) continue;
    if (input.type === "checkbox") input.checked = Boolean(value);
    else input.value = value;
  }
  distanceOutput.value = `${Number(form.elements.distance_m.value).toFixed(1)} m`;
}

function setActiveExample(name) {
  activeExample = name;
  document.querySelectorAll(".example-card").forEach(button => {
    const selected = button.dataset.example === name;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
  const example = name ? examples[name] : null;
  document.getElementById("exampleName").textContent = example?.name || "自由实验";
  document.getElementById("exampleExpected").textContent = example?.expected || "可从左侧修改参数，或选择上方案例。";
  document.getElementById("exampleObserved").textContent = example ? "参数已载入，正在运行…" : "等待运行。";
}

function setPreset(name) {
  applyValues(presets[name]);
  setActiveExample(null);
  document.querySelectorAll(".preset").forEach(button => button.classList.toggle("active", button.dataset.preset === name));
}

function setExample(name, runNow = true) {
  const example = examples[name];
  if (!example) return;
  applyValues(example.values);
  setActiveExample(name);
  document.querySelectorAll(".preset").forEach(button => button.classList.remove("active"));
  if (runNow) runExperiment();
}

function readPayload() {
  const data = new FormData(form);
  const payload = {};
  for (const [key, value] of data.entries()) {
    payload[key] = key === "calibration_mode" ? value : Number(value);
  }
  return payload;
}

function formatDistance(value) {
  return Number.isFinite(value) ? `${value.toFixed(4)} m` : "不可用";
}

function formatError(value) {
  if (!Number.isFinite(value)) return "未产生有效距离";
  const sign = value >= 0 ? "+" : "";
  return `误差 ${sign}${value.toFixed(4)} m`;
}

function badge(element, status) {
  const normalized = String(status || "UNKNOWN").toUpperCase();
  element.textContent = normalized;
  element.className = "badge " + (normalized.includes("PASS") || normalized === "AVAILABLE" ? "pass" : normalized.includes("FAIL") || normalized.includes("NO_PATH") || normalized.includes("REJECT") ? "fail" : "partial");
}

function svgElement(name, attributes = {}) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
  return element;
}

function drawLineChart(svg, x, y, options = {}) {
  svg.replaceChildren();
  if (!x?.length || !y?.length) return;
  const width = 720, height = 280, left = 52, right = 16, top = 16, bottom = 36;
  const plotWidth = width - left - right, plotHeight = height - top - bottom;
  const xMin = Math.min(...x), xMax = Math.max(...x);
  const allY = options.secondaryY?.length ? y.concat(options.secondaryY) : y;
  let yMin = Math.min(...allY), yMax = Math.max(...allY);
  if (Math.abs(yMax - yMin) < 1e-12) { yMin -= 1; yMax += 1; }
  const sx = value => left + (value - xMin) / (xMax - xMin || 1) * plotWidth;
  const sy = value => top + (yMax - value) / (yMax - yMin) * plotHeight;
  for (let i = 0; i <= 4; i++) {
    const py = top + i * plotHeight / 4;
    svg.append(svgElement("line", { x1: left, x2: width - right, y1: py, y2: py, class: "grid" }));
    const label = svgElement("text", { x: left - 8, y: py + 4, "text-anchor": "end" });
    label.textContent = (yMax - i * (yMax - yMin) / 4).toFixed(options.yDigits ?? 1);
    svg.append(label);
  }
  for (let i = 0; i <= 4; i++) {
    const px = left + i * plotWidth / 4;
    svg.append(svgElement("line", { x1: px, x2: px, y1: top, y2: height - bottom, class: "grid" }));
    const label = svgElement("text", { x: px, y: height - 13, "text-anchor": "middle" });
    label.textContent = (xMin + i * (xMax - xMin) / 4).toFixed(options.xDigits ?? 0);
    svg.append(label);
  }
  const points = x.map((value, index) => `${sx(value).toFixed(2)},${sy(y[index]).toFixed(2)}`).join(" ");
  if (options.secondaryY?.length === x.length) {
    const secondaryPoints = x.map((value, index) => `${sx(value).toFixed(2)},${sy(options.secondaryY[index]).toFixed(2)}`).join(" ");
    svg.append(svgElement("polyline", { points: secondaryPoints, class: "line", stroke: options.secondaryColor || "#4c6d7d", "stroke-dasharray": "4 4", opacity: ".75" }));
  }
  svg.append(svgElement("polyline", { points, class: "line", stroke: options.color || "#56d6e7" }));
  if (options.markers) {
    for (const marker of options.markers) {
      if (!Number.isFinite(marker.x) || marker.x < xMin || marker.x > xMax) continue;
      const px = sx(marker.x);
      svg.append(svgElement("line", { x1: px, x2: px, y1: top, y2: height - bottom, class: "marker", stroke: marker.color }));
      const label = svgElement("text", { x: Math.min(px + 5, width - 110), y: top + 13, fill: marker.color });
      label.textContent = marker.label;
      svg.append(label);
    }
  }
  const xLabel = svgElement("text", { x: width - right, y: height - 3, "text-anchor": "end" });
  xLabel.textContent = options.xLabel || "";
  svg.append(xLabel);
}

const reasonLabels = {
  LOW_FRAME_COHERENCE: "帧间相干度不足",
  CALIBRATION_EXCEEDS_DETECTED_DELAY: "校准量超过已检测首径",
  calibrated_distance_out_of_range: "校准后距离越界",
  ambiguous_delay_peaks: "多径峰值存在歧义",
  phase_residual_too_high: "相位残差过高",
  unresolved_multipath_components: "多径分量不可分辨",
  multipath_reconstruction_residual_too_high: "多径重构残差过高",
  multipath_early_path_too_weak: "最早路径过弱",
  sparse_multipath_challenger_disagrees: "候选估计器不一致",
  NO_SIGNIFICANT_CONNECTED_LOBE: "未找到显著连接主瓣",
};

function friendlyReason(reason) {
  if (!reason) return "";
  return String(reason).split(";").map(item => reasonLabels[item] || item.replaceAll("_", " ")).join("；");
}

function appendSvgText(svg, text, attributes = {}) {
  const node = svgElement("text", attributes);
  node.textContent = text;
  svg.append(node);
  return node;
}

function drawDistanceComparison(svg, data) {
  svg.replaceChildren();
  const ble = data.summary.ble;
  const uwb = data.summary.uwb;
  const rows = [
    { label: "BLE 旧单径", value: ble.legacy_single_path_distance_m, color: "#56d6e7" },
    { label: "BLE 鲁棒候选", value: ble.candidate_distance_m, color: "#73a7ff" },
    { label: "BLE 最终可用", value: ble.calibrated_distance_m, color: "#6ce3a4", unavailable: ble.status },
    { label: "UWB 原始", value: uwb.raw_distance_m, color: "#ffad66" },
    { label: "UWB 最终可用", value: uwb.calibrated_distance_m, color: "#bd8cff", unavailable: uwb.status },
  ];
  const truth = Number(data.scenario.distance_m);
  const finiteValues = [truth, ...rows.map(row => row.value)].filter(Number.isFinite);
  const rawMin = Math.min(...finiteValues);
  const rawMax = Math.max(...finiteValues);
  const spread = Math.max(rawMax - rawMin, .25);
  const xMin = rawMin - Math.max(.15, spread * .12);
  const xMax = rawMax + Math.max(.15, spread * .12);
  const width = 720, height = 330, left = 142, right = 36, top = 44, bottom = 42;
  const plotWidth = width - left - right;
  const sx = value => left + (value - xMin) / (xMax - xMin || 1) * plotWidth;

  for (let index = 0; index <= 5; index++) {
    const value = xMin + index * (xMax - xMin) / 5;
    const x = sx(value);
    svg.append(svgElement("line", { x1: x, x2: x, y1: top - 8, y2: height - bottom, class: "grid" }));
    appendSvgText(svg, value.toFixed(spread < 2 ? 2 : 1), { x, y: height - 16, "text-anchor": "middle" });
  }

  const truthX = sx(truth);
  svg.append(svgElement("line", { x1: truthX, x2: truthX, y1: top - 17, y2: height - bottom, class: "truth-marker" }));
  appendSvgText(svg, `真值 ${truth.toFixed(2)} m`, { x: Math.min(truthX + 6, width - 94), y: 18, class: "truth-label" });

  rows.forEach((row, index) => {
    const y = top + 28 + index * 43;
    appendSvgText(svg, row.label, { x: left - 12, y: y + 4, "text-anchor": "end", class: "row-label" });
    svg.append(svgElement("line", { x1: left, x2: width - right, y1: y, y2: y, class: "row-track" }));
    if (Number.isFinite(row.value)) {
      const x = sx(row.value);
      svg.append(svgElement("circle", { cx: x, cy: y, r: 6.5, fill: row.color, class: "distance-dot" }));
      appendSvgText(svg, `${row.value.toFixed(4)} m`, { x: Math.min(x + 10, width - 86), y: y - 9, fill: row.color, class: "value-label" });
    } else {
      appendSvgText(svg, row.unavailable || "不可用", { x: width - right, y: y + 4, "text-anchor": "end", class: "unavailable-label" });
    }
  });

  const usableErrors = [ble.error_m, uwb.error_m].filter(Number.isFinite).map(Math.abs);
  document.getElementById("distanceSpread").textContent = usableErrors.length ? `最大可用误差 ${Math.max(...usableErrors).toFixed(4)} m` : "最终距离已拒绝";
}

function drawPathChart(svg, data) {
  svg.replaceChildren();
  const truthPaths = Array.isArray(data.paths) ? data.paths : [];
  const estimatedPaths = Array.isArray(data.ble_estimated_paths) ? data.ble_estimated_paths : [];
  if (!truthPaths.length) return;
  const losDelay = truthPaths[0].delay_ns;
  const truth = truthPaths.map(path => ({ ...path, excess: path.delay_ns - losDelay }));
  const estimates = estimatedPaths.map(path => ({ ...path, excess: path.delay_ns - losDelay }));
  const excessValues = [0, 10, ...truth.map(path => path.excess), ...estimates.map(path => path.excess)].filter(Number.isFinite);
  let xMin = Math.min(...excessValues);
  let xMax = Math.max(...excessValues);
  const xPadding = Math.max(1, (xMax - xMin) * .08);
  xMin -= xPadding;
  xMax += xPadding;
  const magnitudes = [...truth.map(path => path.magnitude), ...estimates.map(path => path.magnitude)].filter(Number.isFinite);
  const maxMagnitude = Math.max(...magnitudes, 1e-9);
  const width = 720, height = 330, left = 58, right = 28, top = 43, bottom = 60;
  const plotWidth = width - left - right, plotHeight = height - top - bottom;
  const baseline = height - bottom;
  const sx = value => left + (value - xMin) / (xMax - xMin || 1) * plotWidth;
  const sy = magnitude => baseline - magnitude / maxMagnitude * (plotHeight - 20);

  for (let index = 0; index <= 5; index++) {
    const value = xMin + index * (xMax - xMin) / 5;
    const x = sx(value);
    svg.append(svgElement("line", { x1: x, x2: x, y1: top, y2: baseline, class: "grid" }));
    appendSvgText(svg, value.toFixed(1), { x, y: height - 30, "text-anchor": "middle" });
  }
  svg.append(svgElement("line", { x1: left, x2: width - right, y1: baseline, y2: baseline, class: "axis-line" }));
  appendSvgText(svg, "超额延迟 / ns", { x: width - right, y: height - 7, "text-anchor": "end" });
  appendSvgText(svg, "实柱：仿真真值   空心圆：BLE 子空间估计", { x: left, y: 17, class: "path-legend" });

  const barWidth = Math.max(9, Math.min(34, plotWidth / Math.max(truth.length * 3, 8)));
  truth.forEach((path, index) => {
    const x = sx(path.excess);
    const y = sy(path.magnitude);
    const color = index === 0 ? "#6ce3a4" : "#ffad66";
    svg.append(svgElement("rect", { x: x - barWidth / 2, y, width: barWidth, height: baseline - y, rx: 4, fill: color, opacity: index === 0 ? .9 : .72 }));
    appendSvgText(svg, index === 0 ? "LOS" : `R${index}`, { x, y: Math.max(y - 8, top + 9), "text-anchor": "middle", fill: color, class: "path-label" });
    appendSvgText(svg, path.magnitude.toFixed(2), { x, y: baseline + 16, "text-anchor": "middle", class: "magnitude-label" });
  });

  estimates.forEach((path, index) => {
    if (!Number.isFinite(path.excess) || !Number.isFinite(path.magnitude)) return;
    const x = sx(path.excess);
    const y = sy(path.magnitude);
    svg.append(svgElement("line", { x1: x, x2: x, y1: y, y2: baseline, class: "estimate-stem" }));
    svg.append(svgElement("circle", { cx: x, cy: y, r: path.selected ? 7 : 5.5, class: path.selected ? "estimate-dot selected" : "estimate-dot" }));
    appendSvgText(svg, path.selected ? "BLE 选中" : `E${index + 1}`, { x: Math.min(x + 7, width - 76), y: Math.max(y - 8, top + 10), class: path.selected ? "selected-label" : "estimate-label" });
  });

  const ratio = truth.length > 1 ? truth[0].magnitude / Math.max(...truth.slice(1).map(path => path.magnitude)) : null;
  document.getElementById("pathTimelineNote").textContent = Number.isFinite(ratio) ? `首径/最强晚径 ${ratio.toFixed(2)}:1` : "仅 LOS 路径";
}

function resultSummary(data) {
  const ble = data.summary.ble;
  const uwb = data.summary.uwb;
  const bleText = ble.available ? `BLE 误差 ${ble.error_m >= 0 ? "+" : ""}${ble.error_m.toFixed(3)} m` : `BLE ${ble.status}`;
  const uwbText = uwb.available ? `UWB 误差 ${uwb.error_m >= 0 ? "+" : ""}${uwb.error_m.toFixed(3)} m（${uwb.status}）` : `UWB ${uwb.status}`;
  const coherence = Number.isFinite(data.scenario.uwb_frame_coherence) ? `coherence ${data.scenario.uwb_frame_coherence.toFixed(3)}` : "";
  return [bleText, uwbText, coherence].filter(Boolean).join(" · ");
}

function updateExampleResult(data) {
  const summary = resultSummary(data);
  document.getElementById("exampleObserved").textContent = summary;
  if (!activeExample) return;
  const button = document.querySelector(`.example-card[data-example="${activeExample}"]`);
  if (!button) return;
  button.querySelector(".example-result").textContent = summary;
  const risky = !data.summary.ble.available || !data.summary.uwb.available || data.summary.uwb.status !== "PASS" || Math.abs(data.summary.uwb.error_m || 0) > 1;
  button.classList.toggle("has-warning", risky);
  button.classList.toggle("has-result", !risky);
}

function updateResults(data) {
  const ble = data.summary.ble, uwb = data.summary.uwb;
  document.getElementById("bleDistance").textContent = formatDistance(ble.calibrated_distance_m);
  document.getElementById("bleError").textContent = friendlyReason(ble.rejection_reason) || formatError(ble.error_m);
  document.getElementById("bleRaw").textContent = formatDistance(ble.raw_distance_m);
  document.getElementById("bleCandidate").textContent = formatDistance(ble.candidate_distance_m);
  document.getElementById("bleConfidence").textContent = Number.isFinite(ble.confidence) ? `${(ble.confidence * 100).toFixed(1)}%` : "—";
  document.getElementById("bleRatio").textContent = Number.isFinite(ble.ambiguity_peak_ratio) ? ble.ambiguity_peak_ratio.toFixed(2) : "—";
  document.getElementById("bleResidual").textContent = Number.isFinite(ble.phase_residual_rms_deg) ? `${ble.phase_residual_rms_deg.toFixed(2)}°` : "—";
  document.getElementById("bleMode").textContent = ble.estimator_mode === "multipath_matrix_pencil" ? "多径子空间" : "单径相位";
  document.getElementById("bleOrder").textContent = Number.isInteger(ble.model_order) ? String(ble.model_order) : "—";
  badge(document.getElementById("bleBadge"), ble.status);

  document.getElementById("uwbDistance").textContent = formatDistance(uwb.calibrated_distance_m);
  document.getElementById("uwbError").textContent = friendlyReason(uwb.rejection_reason) || formatError(uwb.error_m);
  document.getElementById("uwbConfidence").textContent = Number.isFinite(uwb.confidence) ? `${(uwb.confidence * 100).toFixed(1)}%` : "—";
  document.getElementById("uwbRaw").textContent = formatDistance(uwb.raw_distance_m);
  document.getElementById("uwbFrames").textContent = String(data.scenario.uwb_frame_count ?? "—");
  document.getElementById("uwbCoherence").textContent = Number.isFinite(data.scenario.uwb_frame_coherence) ? data.scenario.uwb_frame_coherence.toFixed(3) : "—";
  badge(document.getElementById("uwbBadge"), uwb.status);

  const risk = data.summary.nlos_probability;
  document.getElementById("nlosProbability").textContent = `${(risk * 100).toFixed(1)}%`;
  document.getElementById("riskFill").style.width = `${Math.max(0, Math.min(100, risk * 100))}%`;
  document.getElementById("pathCount").textContent = `${data.scenario.path_count} 条路径`;

  const calibration = data.summary.rfic_calibration;
  const modeLabels = { frequency_profile: "频率 Profile", scalar: "标量群时延", none: "未校准" };
  document.getElementById("calibrationMode").textContent = modeLabels[calibration.mode] || calibration.mode;
  document.getElementById("bleProfile").textContent = calibration.ble_profile?.profile_id || "—";
  document.getElementById("uwbProfile").textContent = calibration.uwb_profile?.profile_id || "—";
  document.getElementById("calibrationNotice").textContent = calibration.demo_only ? "本页 Profile 是显式合成扫频演示；提交实测结论前必须替换为 VNA/环回曲线。" : calibration.mode === "scalar" ? "仅消除常数群时延，频率波纹仍保留。" : "当前结果未应用 RFIC 校准。";

  drawLineChart(document.getElementById("bleChart"), data.series.ble_frequency_mhz, data.series.ble_calibrated_magnitude_db, { color: "#73a7ff", secondaryY: data.series.ble_raw_magnitude_db, secondaryColor: "#56d6e7", xLabel: "MHz", yDigits: 1 });
  drawLineChart(document.getElementById("uwbChart"), data.series.cir_delay_ns, data.series.cir_normalized_magnitude, {
    color: "#ffad66", xLabel: "ns", xDigits: 1, yDigits: 2,
    markers: [
      { x: data.series.true_delay_ns, color: "#6ce3a4", label: "真实 LOS" },
      { x: data.series.detected_delay_ns, color: "#ff7c83", label: "检测首径" },
    ],
  });
  drawDistanceComparison(document.getElementById("distanceChart"), data);
  drawPathChart(document.getElementById("pathChart"), data);
  updateExampleResult(data);
}

async function runExperiment(event) {
  event?.preventDefault();
  setMessage();
  runButton.disabled = true;
  runButton.firstElementChild.textContent = "计算中…";
  try {
    const response = await fetch("/api/run", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(readPayload()) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "实验运行失败");
    updateResults(data);
  } catch (error) {
    setMessage(error.message || String(error));
  } finally {
    runButton.disabled = false;
    runButton.firstElementChild.textContent = "运行实验";
  }
}

async function loadAcceptance() {
  try {
    const response = await fetch("/api/acceptance");
    const data = await response.json();
    if (!data.available) throw new Error(data.message || "没有验收结果");
    const badges = document.getElementById("acceptanceBadges");
    badges.replaceChildren();
    for (const [label, status] of [["软件", data.software_gate_status], ["标准", data.standards_status], ["总体", data.overall_status]]) {
      const item = document.createElement("span"); item.className = "badge neutral"; item.textContent = `${label} ${status}`; badge(item, status); item.textContent = `${label} ${status}`; badges.append(item);
    }
    const counts = data.status_counts || {};
    const freshness = data.fresh ? "当前版本" : `已过期：${data.message || "请重新验收"}`;
    document.getElementById("acceptanceSummary").textContent = `${freshness} · 生成于 ${data.generated_at || "未知"} · PASS ${counts.PASS || 0} · PARTIAL ${counts.PARTIAL || 0} · FAIL ${counts.FAIL || 0} · NOT TESTED ${counts["NOT TESTED"] || 0}`;
    const container = document.getElementById("acceptanceChecks");
    container.replaceChildren();
    for (const check of data.checks) {
      const item = document.createElement("div");
      const statusClass = check.status === "PASS" ? "pass" : check.status === "FAIL" ? "fail" : "partial";
      item.className = `check-item ${statusClass}`;
      item.title = check.criterion || "";
      item.innerHTML = `<span class="check-dot"></span><b></b><span></span>`;
      item.querySelector("b").textContent = `${check.check_id} · ${check.title}`;
      item.lastElementChild.textContent = check.status;
      container.append(item);
    }
  } catch (error) {
    document.getElementById("acceptanceSummary").textContent = error.message || String(error);
  }
}

async function initialize() {
  distance.addEventListener("input", () => distanceOutput.value = `${Number(distance.value).toFixed(1)} m`);
  form.addEventListener("input", () => {
    if (activeExample) setActiveExample(null);
  });
  form.addEventListener("submit", runExperiment);
  document.getElementById("resetButton").addEventListener("click", () => { setPreset("balanced"); runExperiment(); });
  document.querySelectorAll(".preset").forEach(button => button.addEventListener("click", () => { setPreset(button.dataset.preset); runExperiment(); }));
  document.querySelectorAll(".example-card").forEach(button => button.addEventListener("click", () => setExample(button.dataset.example)));
  setExample("los", false);
  try {
    const response = await fetch("/api/health");
    if (!response.ok) throw new Error();
    document.querySelector(".pulse").classList.add("online");
    document.getElementById("serviceStatus").textContent = "本地引擎在线";
  } catch (_) {
    document.getElementById("serviceStatus").textContent = "本地引擎离线";
  }
  await Promise.all([loadAcceptance(), runExperiment()]);
}

initialize();
