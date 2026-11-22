// Entity annotation forms and view bookmarks. All numeric edits go through
// the six AABB inputs; the viewer shows wireframes live. Saving sends the
// full annotations payload with expected_revision (409 on stale writes).
import { getJSON, putJSON } from "./api.js";
import { state, setState, markDirty, clearDirty, notify } from "./state.js";

let saveInFlight = false;

export function initAnnotations(getViewer) {
  const entityList = document.getElementById("entityList");
  const viewList = document.getElementById("viewList");

  document.getElementById("addEntityBtn").addEventListener("click", addEntity);
  document.getElementById("saveViewBtn").addEventListener("click", saveCurrentView);

  entityList.addEventListener("click", (event) => {
    const button = event.target.closest("button");
    if (!button) return;
    const entityId = button.dataset.entity;
    const act = button.dataset.act;
    if (act === "remove") removeEntity(entityId);
    if (act === "confirm") toggleStatus(entityId);
    if (act === "fact-add") addFact(entityId);
    if (act === "fact-remove") removeFact(entityId, button.dataset.fact);
  });
  entityList.addEventListener("input", (event) => {
    const entityId = event.target.closest(".entity")?.dataset.entity;
    if (!entityId) return;
    const entity = findEntity(entityId);
    if (!entity) return;
    applyEntityInput(entity, event.target);
    // Do NOT re-render here: rebuilding the list would detach the element the
    // user is typing into. Input values already live in the DOM; only the
    // viewer boxes need refreshing.
    updateViewerBoxes(getViewer());
  });
  entityList.addEventListener("change", (event) => {
    if (event.target.classList.contains("status-sel")) {
      const entityId = event.target.closest(".entity")?.dataset.entity;
      const entity = findEntity(entityId);
      if (entity) {
        entity.status = event.target.value;
        renderEntities();
        renderViews();
        updateViewerBoxes(getViewer());
      }
    }
  });
  entityList.addEventListener("click", (event) => {
    const root = event.target.closest(".entity");
    if (root && !event.target.closest("button") && !event.target.closest("input") && !event.target.closest("select")) {
      setState({ selectedEntityId: root.dataset.entity });
      updateViewerBoxes(getViewer());
    }
  });

  viewList.addEventListener("click", (event) => {
    const button = event.target.closest("button");
    if (button) {
      const viewId = button.dataset.view;
      if (button.dataset.act === "remove") removeView(viewId);
      if (button.dataset.act === "apply") {
        const view = findView(viewId);
        if (view) {
          getViewer().setView(view);
          setState({ selectedViewId: viewId });
        }
      }
      return;
    }
    const checks = event.target.closest(".checks");
    if (checks) return;
    const root = event.target.closest(".view");
    if (root) {
      const view = findView(root.dataset.view);
      if (view) {
        getViewer().setView(view);
        setState({ selectedViewId: root.dataset.view });
      }
    }
  });
  viewList.addEventListener("change", (event) => {
    if (event.target.classList.contains("approved-check")) {
      const viewId = event.target.closest(".view")?.dataset.view;
      const entityId = event.target.value;
      const view = findView(viewId);
      if (!view) return;
      if (event.target.checked) {
        if (!view.approved_entity_ids.includes(entityId)) {
          view.approved_entity_ids.push(entityId);
        }
        view.confirmation_source = "manual";
        view.confirmed_at = new Date().toISOString();
      } else {
        view.approved_entity_ids = view.approved_entity_ids.filter((id) => id !== entityId);
        view.confirmed_at = null;
      }
      markDirty();
      renderViews();
      updateViewerBoxes(getViewer());
    }
  });

  document.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "s") {
      event.preventDefault();
      saveAnnotations();
    }
  });
  document.getElementById("saveAnnotationsBtn")?.addEventListener("click", saveAnnotations);
}

function findEntity(id) {
  return state.scene?.entities.find((e) => e.entity_id === id);
}
function findView(id) {
  return state.scene?.views.find((v) => v.view_id === id);
}

function makeId(prefix) {
  return `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 7)}`;
}

function addEntity() {
  if (!state.scene) return;
  const sceneBounds = state.scene.entities[0]?.bounds_asset || defaultBounds();
  const entity = {
    entity_id: makeId("entity"),
    label: "新实体",
    description: "",
    bounds_asset: [
      [sceneBounds[0][0], sceneBounds[0][1], sceneBounds[0][2]],
      [sceneBounds[1][0], sceneBounds[1][1], sceneBounds[1][2]],
    ],
    facts: [],
    status: "draft",
  };
  state.scene.entities.push(entity);
  setState({ selectedEntityId: entity.entity_id });
  markDirty();
  renderEntities();
}

function defaultBounds() {
  return [[0, 0, 0], [1, 1, 1]];
}

function removeEntity(entityId) {
  if (!state.scene) return;
  state.scene.entities = state.scene.entities.filter((e) => e.entity_id !== entityId);
  for (const view of state.scene.views) {
    view.approved_entity_ids = view.approved_entity_ids.filter((id) => id !== entityId);
  }
  markDirty();
  setState({ selectedEntityId: null });
  renderEntities();
  renderViews();
}

function toggleStatus(entityId) {
  const entity = findEntity(entityId);
  if (!entity) return;
  entity.status = entity.status === "confirmed" ? "draft" : "confirmed";
  markDirty();
  renderEntities();
}

function addFact(entityId) {
  const entity = findEntity(entityId);
  if (!entity) return;
  entity.facts.push({
    fact_id: makeId("fact"),
    text: "",
    source: "user_confirmed",
    kind: "descriptive",
    scope_view_id: null,
  });
  markDirty();
  renderEntities();
}

function removeFact(entityId, factId) {
  const entity = findEntity(entityId);
  if (!entity) return;
  entity.facts = entity.facts.filter((f) => f.fact_id !== factId);
  markDirty();
  renderEntities();
}

function applyEntityInput(entity, input) {
  const name = input.dataset.field;
  if (!name) return;
  if (name === "label") entity.label = input.value;
  if (name === "description") entity.description = input.value;
  if (name === "fact_text") entity.facts[Number(input.dataset.factIndex)].text = input.value;
  if (name === "fact_scope") entity.facts[Number(input.dataset.factIndex)].scope_view_id = input.value || null;
  if (name.startsWith("bound:")) {
    const [axis, side] = name.slice(6).split(":");
    const value = Number(input.value);
    if (!Number.isFinite(value)) return;
    entity.bounds_asset[side === "min" ? 0 : 1][["x", "y", "z"].indexOf(axis)] = value;
  }
  markDirty();
}

export function renderEntities() {
  const box = document.getElementById("entityList");
  if (!state.scene) {
    box.innerHTML = "";
    return;
  }
  box.innerHTML = state.scene.entities
    .map((entity) => {
      const [lo, hi] = entity.bounds_asset;
      const bounds = (side) => `
        <input type="number" step="any" data-field="bound:x:${side}" value="${side === "min" ? lo[0] : hi[0]}" title="x">
        <input type="number" step="any" data-field="bound:y:${side}" value="${side === "min" ? lo[1] : hi[1]}" title="y">
        <input type="number" step="any" data-field="bound:z:${side}" value="${side === "min" ? lo[2] : hi[2]}" title="z">`;
      const facts = (entity.facts || [])
        .map((fact, index) => `
          <div class="fact-row">
            <input type="text" data-field="fact_text" data-fact-index="${index}" value="${escapeAttr(fact.text)}" placeholder="事实描述">
            <select data-field="fact_scope" data-fact-index="${index}" title="依赖镜头（可选）">
              <option value="">无</option>
              ${(state.scene.views || []).map((v) => `<option value="${v.view_id}" ${fact.scope_view_id === v.view_id ? "selected" : ""}>${escapeAttr(v.name)}</option>`).join("")}
            </select>
            <button class="remove-btn" data-act="fact-remove" data-entity="${entity.entity_id}" data-fact="${fact.fact_id}">✕</button>
          </div>`)
        .join("");
      return `
        <div class="entity ${entity.entity_id === state.selectedEntityId ? "selected" : ""}" data-entity="${entity.entity_id}">
          <div class="head">
            <input type="text" data-field="label" value="${escapeAttr(entity.label)}">
            <select class="status-sel" title="状态">
              <option value="draft" ${entity.status === "draft" ? "selected" : ""}>草稿</option>
              <option value="confirmed" ${entity.status === "confirmed" ? "selected" : ""}>已确认</option>
            </select>
            <button class="remove-btn" data-act="remove" data-entity="${entity.entity_id}">✕</button>
          </div>
          <input type="text" data-field="description" value="${escapeAttr(entity.description || "")}" placeholder="简述（人工说明）" style="width:100%;margin-top:4px;">
          <div class="bounds">
            <label class="mini">xmin/ymin/zmin</label><label class="mini">xmax/ymax/zmax</label>
            ${bounds("min")}${bounds("max")}
          </div>
          <button data-act="fact-add" data-entity="${entity.entity_id}" style="margin-top:6px;">＋ 事实</button>
          ${facts}
        </div>`;
    })
    .join("");
}

function saveCurrentView() {
  if (!state.scene) return;
  const viewer = window.__tourViewer;
  if (!viewer || !viewer.transform) return;
  const name = document.getElementById("viewName").value.trim() || `镜头 ${state.scene.views.length + 1}`;
  const camera = viewer.getCameraInAssetCoordinates();
  const view = {
    view_id: makeId("view"),
    name,
    camera,
    approved_entity_ids: [],
    confirmation_source: "manual",
    confirmed_at: null,
  };
  state.scene.views.push(view);
  setState({ selectedViewId: view.view_id });
  markDirty();
  renderViews();
  renderEntities(); // fact scope selectors depend on views
}

function removeView(viewId) {
  if (!state.scene) return;
  state.scene.views = state.scene.views.filter((v) => v.view_id !== viewId);
  for (const entity of state.scene.entities) {
    for (const fact of entity.facts) {
      if (fact.scope_view_id === viewId) fact.scope_view_id = null;
    }
  }
  markDirty();
  renderViews();
  renderEntities();
}

export function renderViews() {
  const box = document.getElementById("viewList");
  if (!state.scene) {
    box.innerHTML = "";
    return;
  }
  box.innerHTML = state.scene.views
    .map((view) => `
      <div class="view" data-view="${view.view_id}">
        <div class="head">
          <b title="点击恢复该视角">🎥 ${escapeHtml(view.name)}</b>
          <button data-act="apply" data-view="${view.view_id}">恢复</button>
          <button class="remove-btn" data-act="remove" data-view="${view.view_id}">✕</button>
        </div>
        <div class="checks">
          ${(state.scene.entities || [])
            .map((entity) => `
              <label>
                <input type="checkbox" class="approved-check" value="${entity.entity_id}"
                  ${view.approved_entity_ids.includes(entity.entity_id) ? "checked" : ""}>
                ${escapeHtml(entity.label)}${entity.status !== "confirmed" ? "（草稿）" : ""}
              </label>`)
            .join("")}
          ${(state.scene.entities || []).length === 0 ? '<span class="mini">先添加实体</span>' : ""}
        </div>
        <div class="mini">${view.confirmed_at ? "已人工确认 ✓" : "确认已失效/未确认"}</div>
      </div>`)
    .join("");
}

export async function saveAnnotations() {
  if (!state.scene || saveInFlight) return;
  saveInFlight = true;
  const status = document.getElementById("saveStatus");
  try {
    const updated = await putJSON(`/api/tour/scenes/${state.scene.scene_id}/annotations`, {
      entities: state.scene.entities,
      views: state.scene.views,
      expected_revision: state.scene.revision,
    });
    state.scene = updated;
    clearDirty();
    setState({ scene: updated });
    status.textContent = `已保存 ✓（rev ${updated.revision}）`;
    status.className = "status-text ok-text";
    setTimeout(() => {
      status.textContent = "";
    }, 2000);
    renderEntities();
    renderViews();
    renderRequiredEntities();
    renderOpeningView();
  } catch (err) {
    status.textContent = `保存失败: ${err.message}`;
    status.className = "status-text error-text";
  } finally {
    saveInFlight = false;
  }
}

export function updateViewerBoxes(viewer) {
  if (!viewer || !state.scene) return;
  viewer.showEntityBoxes(state.scene.entities, {
    selectedId: state.selectedEntityId,
  });
}

function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = value ?? "";
  return div.innerHTML;
}

function escapeAttr(value) {
  return escapeHtml(value).replaceAll('"', "&quot;");
}

// Re-exported for main.js (rendering required-entity checkboxes and opening view).
export function renderRequiredEntities() {
  const box = document.getElementById("requiredEntities");
  if (!state.scene) {
    box.innerHTML = "";
    return;
  }
  const confirmed = state.scene.entities.filter((e) => e.status === "confirmed");
  box.innerHTML = confirmed.length
    ? confirmed
        .map((entity) => `
          <label>
            <input type="checkbox" class="required-check" value="${entity.entity_id}" checked>
            ${escapeHtml(entity.label)}
          </label>`)
        .join("")
    : '<span class="mini">（无已确认实体）</span>';
}

export function renderOpeningView() {
  const select = document.getElementById("openingViewSel");
  if (!state.scene) {
    select.innerHTML = '<option value="">（不限）</option>';
    return;
  }
  const current = select.value;
  select.innerHTML = '<option value="">（不限）</option>' +
    state.scene.views.map((v) => `<option value="${v.view_id}">${escapeAttr(v.name)}</option>`).join("");
  if ([...select.options].some((o) => o.value === current)) select.value = current;
}
