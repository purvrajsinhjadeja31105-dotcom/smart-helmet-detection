// Smart Helmet Detection - single-page UI.
// Talks to the FastAPI backend served from the same origin (/api/v1/...).
"use strict";

const API = "/api/v1";
const $ = (id) => document.getElementById(id);
const IMAGE_EXT = ["jpg", "jpeg", "png", "bmp", "webp"];
const VIDEO_EXT = ["mp4", "avi", "mov", "mkv", "webm", "m4v"];

let selectedFile = null;
let selectedKind = null;     // "image" | "video"
let originalUrl = null;      // object URL of the selected file (shown as "Original")

// ------------------------------------------------------------------ helpers
const show = (el, on = true) => el.classList.toggle("hidden", !on);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fileUrl = (runId, name) => `${API}/files/${encodeURIComponent(runId)}/${encodeURIComponent(name)}`;
const pct = (x) => `${Math.round((x || 0) * 100)}%`;
const fmtSize = (b) => (b > 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1e3))} KB`);
const fmtTime = (s) => {
  const m = Math.floor(s / 60), r = (s % 60).toFixed(1);
  return m ? `${m}:${r.padStart(4, "0")}` : `${r} s`;
};

function kindOf(file) {
  const ext = (file.name.split(".").pop() || "").toLowerCase();
  if (file.type.startsWith("image/") || IMAGE_EXT.includes(ext)) return "image";
  if (file.type.startsWith("video/") || VIDEO_EXT.includes(ext)) return "video";
  return null;
}

function showError(msg) {
  $("error").textContent = msg;
  show($("error"), true);
  show($("progress-card"), false);
  $("analyse-btn").disabled = false;
}

function setProgress(fraction, label, note = "") {
  $("progress-fill").style.width = `${Math.min(100, Math.max(0, fraction * 100))}%`;
  $("progress-label").textContent = label;
  $("progress-note").textContent = note;
}

// ------------------------------------------------------------------ system status
async function checkHealth() {
  try {
    const r = await fetch(`${API}/health`);
    const h = await r.json();
    if (h.status === "ok") {
      $("status").textContent = `Ready · ${h.device}`;
      $("status").className = "status status-ok";
    } else {
      $("status").textContent = "Models not loaded";
      $("status").className = "status status-bad";
      showError(`The AI models could not be loaded: ${h.error || "unknown error"}`);
    }
  } catch {
    $("status").textContent = "Server not reachable";
    $("status").className = "status status-bad";
  }
}

// ------------------------------------------------------------------ file selection
function selectFile(file) {
  const kind = kindOf(file);
  show($("error"), false);
  if (!kind) {
    showError("Please choose an image (JPG, PNG, WEBP, BMP) or a video (MP4, AVI, MOV, MKV, WEBM).");
    return;
  }
  selectedFile = file;
  selectedKind = kind;
  if (originalUrl) URL.revokeObjectURL(originalUrl);
  originalUrl = URL.createObjectURL(file);

  $("file-name").textContent = file.name;
  $("file-meta").textContent = `${kind === "image" ? "Image" : "Video"} · ${fmtSize(file.size)}`;
  show($("preview-img"), kind === "image");
  show($("preview-video"), kind === "video");
  show($("all-plates-row"), kind === "image");
  if (kind === "image") $("preview-img").src = originalUrl;
  else $("preview-video").src = originalUrl;

  show($("dropzone"), false);
  show($("selected"), true);
  show($("results"), false);
}

function resetUpload() {
  selectedFile = null;
  selectedKind = null;
  $("file-input").value = "";
  $("preview-img").removeAttribute("src");
  $("preview-video").removeAttribute("src");
  show($("selected"), false);
  show($("dropzone"), true);
  show($("results"), false);
  show($("progress-card"), false);
  show($("error"), false);
  $("analyse-btn").disabled = false;
  window.scrollTo({ top: 0, behavior: "smooth" });
}

// ------------------------------------------------------------------ upload (XHR for upload progress)
function upload(url, file, removeRain, allPlates = false) {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append("file", file);
    form.append("remove_rain", removeRain ? "true" : "false");
    form.append("read_all_plates", allPlates ? "true" : "false");
    const xhr = new XMLHttpRequest();
    xhr.open("POST", url);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) setProgress(0.3 * (e.loaded / e.total), `Uploading… ${pct(e.loaded / e.total)}`);
    };
    xhr.upload.onload = () => setProgress(0.3, "Analysing…",
      selectedKind === "image" ? "Finding motorcycles, helmets and number plates (a few seconds on CPU)" : "");
    xhr.onload = () => {
      let body = null;
      try { body = JSON.parse(xhr.responseText); } catch { /* not JSON */ }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body);
      else reject(new Error(body?.detail ? String(body.detail) : `Server error ${xhr.status}`));
    };
    xhr.onerror = () => reject(new Error("Could not reach the server. Is it still running?"));
    xhr.send(form);
  });
}

async function pollJob(jobId) {
  const started = Date.now();
  for (;;) {
    await new Promise((r) => setTimeout(r, 1500));
    const r = await fetch(`${API}/jobs/${encodeURIComponent(jobId)}`);
    if (!r.ok) throw new Error("Lost track of the video job.");
    const job = await r.json();
    const secs = Math.round((Date.now() - started) / 1000);
    if (job.status === "done") return job.run_id;
    if (job.status === "failed") throw new Error(job.error || "Video processing failed.");
    if (job.status === "queued") {
      setProgress(0.3, "Waiting for the previous job to finish…", `${secs} s`);
    } else {
      setProgress(0.3 + 0.7 * job.progress, `Analysing video… ${pct(job.progress)}`,
        `Tracking motorcycles frame by frame · ${secs} s elapsed`);
    }
  }
}

async function analyse() {
  if (!selectedFile) return;
  $("analyse-btn").disabled = true;
  show($("error"), false);
  show($("results"), false);
  show($("progress-card"), true);
  setProgress(0, "Uploading…");
  const removeRain = $("remove-rain").checked;
  try {
    let report;
    if (selectedKind === "image") {
      report = await upload(`${API}/detect/image`, selectedFile, removeRain, $("all-plates").checked);
    } else {
      const job = await upload(`${API}/detect/video`, selectedFile, removeRain);
      const runId = await pollJob(job.job_id);
      const r = await fetch(`${API}/results/${encodeURIComponent(runId)}`);
      if (!r.ok) throw new Error("The video finished, but its report could not be loaded.");
      report = await r.json();
    }
    setProgress(1, "Done");
    renderResults(report);
  } catch (e) {
    showError(e.message || String(e));
  } finally {
    $("analyse-btn").disabled = false;
  }
}

// ------------------------------------------------------------------ results
function renderResults(rep) {
  const s = rep.summary || {};
  const runId = rep.run_id;
  const isVideo = rep.type === "video";

  $("s-moto").textContent = s.motorcycles ?? 0;
  $("s-viol").textContent = s.violations ?? 0;
  $("s-safe").textContent = s.safe ?? 0;
  $("s-unk").textContent = isVideo ? "–" : (s.unknown ?? 0);
  const seconds = isVideo ? (s.processing_s ?? rep.processing_ms / 1000) : rep.processing_ms / 1000;
  $("s-time").textContent = fmtTime(seconds);

  const v = $("verdict");
  if ((s.violations ?? 0) > 0) {
    v.className = "verdict verdict-bad";
    v.textContent = `${s.violations} helmet violation${s.violations > 1 ? "s" : ""} found` +
      (s.plates_read ? ` · ${s.plates_read} number plate${s.plates_read > 1 ? "s" : ""} read` : "");
  } else if ((s.motorcycles ?? 0) > 0) {
    v.className = "verdict verdict-good";
    v.textContent = "No violations: all detected riders wear helmets (or no head was visible)";
  } else {
    v.className = "verdict verdict-none";
    v.textContent = "No motorcycles were detected in this file";
  }

  const enh = rep.enhancement?.applied || [];
  const asked = rep.options?.requested_enhancements || [];
  $("enh-note").textContent = isVideo
    ? `Video: ${rep.video?.duration_s ?? "?"} s, ${rep.video?.analysed_frames ?? "?"} frames analysed` +
      (asked.length ? ` · requested: ${asked.join(", ")}` : "") + ` · device: ${rep.device}`
    : (enh.length ? `Image enhancement applied: ${enh.join(", ").replace(/_/g, " ")}` : "Image quality was good: no enhancement needed") +
      ` · device: ${rep.device}`;
  if (s.riders_without_bike_box) {
    $("enh-note").textContent += ` · ${s.riders_without_bike_box} motorcycle${s.riders_without_bike_box > 1 ? "s" : ""}` +
      " found through the rider's head (the bike itself was not clearly detected)";
  }

  // media
  show($("image-view"), !isVideo);
  show($("video-view"), isVideo);
  const mediaName = rep.files?.annotated;
  if (isVideo) {
    $("proc-video").src = fileUrl(runId, mediaName);
  } else {
    $("orig-img").src = originalUrl;
    $("proc-img").src = fileUrl(runId, mediaName);
  }
  $("dl-media").href = fileUrl(runId, mediaName);
  $("dl-media").setAttribute("download", `${runId}_${mediaName}`);
  $("dl-report").href = `${API}/results/${encodeURIComponent(runId)}`;
  $("dl-report").setAttribute("download", `${runId}_report.json`);

  // violation table
  const rows = rep.violations || [];
  show($("no-viol"), rows.length === 0);
  show($("viol-table"), rows.length > 0);
  $("col-extra").textContent = isVideo ? "Seen in video" : "No-helmet confidence";
  $("viol-body").innerHTML = rows.map((r, i) => {
    const evidence = r.evidence_image
      ? `<img src="${fileUrl(runId, r.evidence_image)}" alt="Evidence: rider without helmet">` : "–";
    const plateImg = r.plate_image
      ? `<img class="plate" src="${fileUrl(runId, r.plate_image)}" alt="Number plate crop">`
      : `<span class="muted small">${r.plate_detected ? "plate found" : "not visible"}</span>`;
    const plateText = r.plate_number
      ? `<div class="plate-text">${esc(r.plate_number)}</div><span class="tag ${r.plate_valid_format ? "tag-ok" : "tag-warn"}">${r.plate_valid_format ? "valid Indian format" : "check manually"}</span>`
      : `<span class="muted small">unreadable</span>`;
    const extra = isVideo ? `${fmtTime(r.start_s)} – ${fmtTime(r.end_s)}` : pct(r.no_helmet_confidence);
    return `<tr><td>${i + 1}</td><td>${evidence}</td><td>${plateImg}</td><td>${plateText}</td>` +
      // rider count is an estimate (heads linked to this motorcycle); in dense traffic it can include
      // neighbours, so no "triple riding" claim is made from it
      `<td>${r.plate_number ? pct(r.ocr_confidence) : "–"}</td><td>${r.rider_count ?? "–"}</td><td>${extra}</td></tr>`;
  }).join("");

  // all plates (only when "Read all number plates" was ticked)
  const plates = rep.options?.read_all_plates ? (rep.plates || []) : [];
  show($("plates-card"), !isVideo && rep.options?.read_all_plates);
  $("plates-body").innerHTML = plates.length ? plates.map((p, i) => {
    const img = p.plate_image ? `<img class="plate" src="${fileUrl(runId, p.plate_image)}" alt="Number plate crop">` : "–";
    const text = p.plate_number
      ? `<div class="plate-text">${esc(p.plate_number)}</div><span class="tag ${p.plate_valid_format ? "tag-ok" : "tag-warn"}">${p.plate_valid_format ? "valid Indian format" : "check manually"}</span>`
      : `<span class="muted small">unreadable</span>`;
    return `<tr><td>${i + 1}</td><td>${img}</td><td>${text}</td><td>${pct(p.plate_confidence)}</td>` +
      `<td>${p.plate_number ? pct(p.ocr_confidence) : "–"}</td><td>${esc(p.status)}</td></tr>`;
  }).join("") : `<tr><td colspan="6" class="muted">No number plate was found on the detected motorcycles.</td></tr>`;

  show($("progress-card"), false);
  show($("results"), true);
  $("results").scrollIntoView({ behavior: "smooth", block: "start" });
}

// ------------------------------------------------------------------ wiring
function init() {
  const dz = $("dropzone");
  $("file-input").addEventListener("change", (e) => e.target.files[0] && selectFile(e.target.files[0]));
  ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", (e) => e.dataTransfer.files[0] && selectFile(e.dataTransfer.files[0]));
  $("analyse-btn").addEventListener("click", analyse);
  $("change-btn").addEventListener("click", resetUpload);
  $("again-btn").addEventListener("click", resetUpload);
  checkHealth();
}

init();
