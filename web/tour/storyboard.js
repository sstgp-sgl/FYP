// Storyboard editing: shot cards, validation feedback, save/download and the
// saved-storyboard list. Text is rendered via textContent (no innerHTML for
// user data); structure markup is generated with escaping helpers.
import { getJSON, postJSON, putJSON, downloadUrl } from "./api.js";
import { state, setState, markDirty } from "./state.js";

let storyboardDraft = null; // {storyboard_id?, revision?, title, language, shots}

function makeId(prefix) {
  return `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 7)}`;
}

export function initStoryboards(loadStoryboardCallback) {
  document.getElementById("addShotBtn").addEventListener("click", () => {
    if (!storyboardDraft) return;
    storyboardDraft.shots.push(blankShot());
    renderStoryboardDraft();
  });
  document.getElementById("saveSbBtn").addEventListener("click", saveStoryboard);
  document.getElementById("validateSbBtn").addEventListener("click", validateStoryboard);
  document.getElementById("sbList").addEventListener("click", (event) => {
    const button = event.target.closest("button");
    if (!button) return;
    loadStoryboardCallback(button.dataset.sb);
  });
  document.getElementById("shotList").addEventListener("click", (event) => {
    const button = event.target.closest("button");
    if (!button) return;
    const index = Number(button.dataset.shotIndex);
    const act = button.dataset.act;
    if (act === "remove") {
      storyboardDraft.shots.splice(index, 1);
      renderStoryboardDraft();
    }
    if (act === "view") {
      const shot = storyboardDraft.shots[index];
      const view = state.scene?.views.find((v) => v.view_id === shot.view_id);
      if (view && window.__tourViewer) {
        window.__tourViewer.setView(view);
        setState({ selectedViewId: view.view_id });
      }
    }
  });
  document.getElementById("shotList").addEventListener("input", (event) => {
    const index = Number(event.target.dataset.shotIndex);
    if (Number.isNaN(index)) return;
    const shot = storyboardDraft?.shots[index];
    if (!shot) return;
    applyShotInput(shot, event.target);
  });
  document.getElementById("shotList").addEventListener("change", (event) => {
    const index = Number(event.target.dataset.shotIndex);
    if (Number.isNaN(index)) return;
    const shot = storyboardDraft?.shots[index];
    if (!shot) return;
    if (event.target.classList.contains("view-select")) {
      shot.view_id = event.target.value;
      // focus must stay within the human-approved set of the chosen view
      const view = state.scene?.views.find((v) => v.view_id === shot.view_id);
      if (view) {
        shot.focus_entity_ids = shot.focus_entity_ids.filter((id) =>
          view.approved_entity_ids.includes(id)
        );
        // facts must belong to focused entities and stay inside their scope
        const allowed = new Set();
        for (const entity of state.scene?.entities || []) {
          if (!shot.focus_entity_ids.includes(entity.entity_id)) continue;
          for (const fact of entity.facts || []) {
            if (!fact.scope_view_id || fact.scope_view_id === shot.view_id) {
              allowed.add(fact.fact_id);
            }
          }
        }
        shot.fact_ids = shot.fact_ids.filter((id) => allowed.has(id));
      } else {
        shot.focus_entity_ids = [];
        shot.fact_ids = [];
      }
      renderStoryboardDraft();
      return;
    }
    if (event.target.classList.contains("focus-check")) {
      const entityId = event.target.value;
      if (event.target.checked) {
        if (!shot.focus_entity_ids.includes(entityId)) shot.focus_entity_ids.push(entityId);
      } else {
        shot.focus_entity_ids = shot.focus_entity_ids.filter((id) => id !== entityId);
      }
      renderStoryboardDraft();
      return;
    }
    if (event.target.classList.contains("fact-check")) {
      const factId = event.target.value;
      if (event.target.checked) {
        if (!shot.fact_ids.includes(factId)) shot.fact_ids.push(factId);
      } else {
        shot.fact_ids = shot.fact_ids.filter((id) => id !== factId);
      }
      renderStoryboardDraft();
    }
  });
}

function blankShot() {
  const view = state.scene?.views[0];
  return {
    shot_id: makeId("shot"),
    view_id: view ? view.view_id : "",
    focus_entity_ids: [],
    narration: "",
    fact_ids: [],
    duration_seconds: 10,
    transition: "cut",
  };
}

export function newStoryboardDraft() {
  storyboardDraft = {
    storyboard_id: null,
    revision: null,
    title: state.scene ? `${state.scene.title}导览` : "导览",
    language: "zh-CN",
    shots: [],
  };
  renderStoryboardDraft();
}

export function getStoryboardDraft() {
  return storyboardDraft;
}

export function isDraftPlayable() {
  return Boolean(storyboardDraft && storyboardDraft.shots && storyboardDraft.shots.length > 0);
}

export function loadDraftFromStoryboard(storyboard) {
  storyboardDraft = {
    storyboard_id: storyboard.storyboard_id,
    revision: storyboard.revision,
    title: storyboard.title,
    language: storyboard.language,
    shots: storyboard.shots.map((s) => ({ ...s, focus_entity_ids: [...s.focus_entity_ids], fact_ids: [...s.fact_ids] })),
  };
  renderStoryboardDraft();
  const download = document.getElementById("downloadSb");
  download.classList.remove("hidden");
  download.href = downloadUrl(storyboard.storyboard_id);
}

function applyShotInput(shot, input) {
  const name = input.dataset.field;
  if (!name) return;
  if (name === "narration") shot.narration = input.value;
  if (name === "duration") {
    const value = Number(input.value);
    if (Number.isFinite(value)) shot.duration_seconds = Math.max(1, Math.min(60, Math.round(value)));
  }
  updateTotals();
}

function updateTotals() {
  const total = (storyboardDraft?.shots || []).reduce((sum, shot) => sum + shot.duration_seconds, 0);
  document.getElementById("sbTotals").textContent =
    `段数 ${storyboardDraft?.shots.length || 0} · 总时长 ${total}s`;
}

export function renderStoryboardDraft() {
  const title = document.getElementById("sbTitle");
  const list = document.getElementById("shotList");
  if (!storyboardDraft) {
    list.innerHTML = "";
    updateTotals();
    return;
  }
  title.value = storyboardDraft.title;
  title.oninput = () => {
    storyboardDraft.title = title.value;
  };
  const scene = state.scene;
  list.innerHTML = storyboardDraft.shots
    .map((shot, index) => {
      const view = scene?.views.find((v) => v.view_id === shot.view_id);
      const focusable = view ? view.approved_entity_ids : [];
      const facts = collectFacts(shot.focus_entity_ids);
      const viewOptions = (scene?.views || [])
        .map((v) => `<option value="${v.view_id}" ${v.view_id === shot.view_id ? "selected" : ""}>${escapeAttr(v.name)}</option>`)
        .join("");
      return `
        <div class="shot" data-shot-index="${index}">
          <div class="head">
            <b>第 ${index + 1} 段</b>
            <button data-act="view" data-shot-index="${index}" title="预览该镜头">🎥</button>
            <button class="remove-btn" data-act="remove" data-shot-index="${index}">✕</button>
          </div>
          <select class="view-select" data-shot-index="${index}">
            <option value="">（选择镜头）</option>${viewOptions}
          </select>
          <div class="checks">
            ${(scene?.entities || []).map((entity) => `
              <label title="${view ? "" : "先选择镜头"}">
                <input type="checkbox" class="focus-check" value="${entity.entity_id}" data-shot-index="${index}"
                  ${shot.focus_entity_ids.includes(entity.entity_id) ? "checked" : ""}
                  ${focusable.includes(entity.entity_id) ? "" : "disabled"}>
                ${escapeHtml(entity.label)}
              </label>`).join("")}
          </div>
          <div class="checks">
            ${facts.map((fact) => `
              <label>
                <input type="checkbox" class="fact-check" value="${fact.fact_id}" data-shot-index="${index}"
                  ${shot.fact_ids.includes(fact.fact_id) ? "checked" : ""}>
                ${escapeHtml(fact.text.slice(0, 40))}${fact.text.length > 40 ? "…" : ""}
              </label>`).join("")}
          </div>
          <textarea rows="2" data-shot-index="${index}" data-field="narration" placeholder="旁白字幕">${escapeHtml(shot.narration)}</textarea>
          <div class="row">
            <label>时长(s) <input type="number" min="2" max="60" value="${shot.duration_seconds}" data-shot-index="${index}" data-field="duration"></label>
            <span class="mini">cut</span>
          </div>
        </div>`;
    })
    .join("");
  updateTotals();
}

function collectFacts(focusEntityIds) {
  const facts = [];
  for (const entity of state.scene?.entities || []) {
    if (!focusEntityIds.includes(entity.entity_id)) continue;
    for (const fact of entity.facts || []) facts.push(fact);
  }
  return facts;
}

function draftPayload() {
  return {
    scene_id: state.scene.scene_id,
    scene_revision: state.scene.revision,
    title: storyboardDraft.title,
    language: storyboardDraft.language,
    shots: storyboardDraft.shots,
    storyboard_id: storyboardDraft.storyboard_id || undefined,
    expected_revision: storyboardDraft.revision ?? undefined,
  };
}

async function saveStoryboard() {
  const status = document.getElementById("sbStatus");
  if (!storyboardDraft || !state.scene) return;
  try {
    const payload = draftPayload();
    const saved = storyboardDraft.storyboard_id
      ? await putJSON(`/api/tour/storyboards/${storyboardDraft.storyboard_id}`, payload)
      : await postJSON("/api/tour/storyboards", payload);
    storyboardDraft.storyboard_id = saved.storyboard_id;
    storyboardDraft.revision = saved.revision;
    const download = document.getElementById("downloadSb");
    download.classList.remove("hidden");
    download.href = downloadUrl(saved.storyboard_id);
    status.textContent = `已保存 ✓（rev ${saved.revision}）`;
    status.className = "hint ok-text";
    document.dispatchEvent(new CustomEvent("storyboard-saved", { detail: saved }));
    await refreshStoryboardList();
  } catch (err) {
    status.textContent = `保存失败: ${err.message}`;
    status.className = "hint error-text";
  }
}

async function validateStoryboard() {
  const status = document.getElementById("sbStatus");
  if (!storyboardDraft || !state.scene) return;
  try {
    const result = await postJSON("/api/tour/storyboards/validate", {
      scene_id: state.scene.scene_id,
      scene_revision: state.scene.revision,
      title: storyboardDraft.title,
      language: storyboardDraft.language,
      shots: storyboardDraft.shots,
    });
    const errors = (result.errors || []).map((e) => `✗ ${e}`).join("\n");
    const warnings = (result.warnings || []).map((w) => `△ ${w}`).join("\n");
    status.textContent = result.status === "valid"
      ? `校验通过 ✓${warnings ? "\n" + warnings : ""}`
      : errors + (warnings ? "\n" + warnings : "");
    status.className = result.status === "valid" ? "hint ok-text" : "hint error-text";
  } catch (err) {
    status.textContent = `校验失败: ${err.message}`;
    status.className = "hint error-text";
  }
}

export async function refreshStoryboardList() {
  const box = document.getElementById("sbList");
  if (!state.scene) {
    box.innerHTML = "";
    return;
  }
  const data = await getJSON(`/api/tour/storyboards?scene_id=${encodeURIComponent(state.scene.scene_id)}`);
  const items = data.storyboards || [];
  box.innerHTML = items.length
    ? items
        .map((sb) => `
          <div class="sb-item">
            <b>${escapeHtml(sb.title)}</b>
            <span class="mini">（rev ${sb.revision} · ${sb.source} · 场景rev ${sb.scene_revision}）</span>
            <div class="actions">
              <button data-sb="${sb.storyboard_id}">载入</button>
              <a href="${downloadUrl(sb.storyboard_id)}">下载 JSON</a>
            </div>
          </div>`)
        .join("")
    : '<span class="mini">还没有已保存的导览</span>';
}

function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = value ?? "";
  return div.innerHTML;
}

function escapeAttr(value) {
  return escapeHtml(value).replaceAll('"', "&quot;");
}
