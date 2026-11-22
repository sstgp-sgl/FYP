// Page orchestration: boot, scene loading/import, generation runs, playback.
import { getJSON, postJSON, uploadPLY } from "./api.js";
import { state, setState, subscribe, markDirty, clearDirty } from "./state.js";
import { Viewer } from "./viewer.js";
import { Player } from "./player.js";
import {
  initAnnotations,
  renderEntities,
  renderViews,
  saveAnnotations,
  updateViewerBoxes,
  renderRequiredEntities,
  renderOpeningView,
} from "./annotations.js";
import {
  initStoryboards,
  newStoryboardDraft,
  loadDraftFromStoryboard,
  renderStoryboardDraft,
  refreshStoryboardList,
  getStoryboardDraft,
  isDraftPlayable,
} from "./storyboard.js";

let viewer = null;
let pollTimer = null;

const player = new Player({
  onShot: onPlayerShot,
  onTick: onPlayerTick,
  onState: onPlayerState,
});
window.__tourViewer = null;

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------- boot
async function boot() {
  initAnnotations(() => viewer);
  initStoryboards(loadStoryboardById);

  viewer = new Viewer($("canvasHost"), $("stage"));
  try {
    viewer.init();
  } catch (err) {
    const box = $("webglError");
    box.textContent = err.message;
    box.classList.remove("hidden");
  }
  window.__tourViewer = viewer;
  viewer.setInteractionEnabled(true);

  wireEvents();

  try {
    const capabilities = await getJSON("/api/tour/capabilities");
    setState({ capabilities });
    const genStatus = $("genStatus");
    if (!capabilities.llm.configured) {
      genStatus.textContent = capabilities.llm.enabled
        ? "LLM 未配置（缺 base_url/model/API key）；可先手动分镜"
        : "LLM 未启用；可先手动分镜";
    }
  } catch (err) {
    showFatal("无法连接导览服务: " + err.message);
    return;
  }
  await reloadScenes();
}

function showFatal(message) {
  $("webglError").textContent = message;
  $("webglError").classList.remove("hidden");
}

function wireEvents() {
  $("importBtn").addEventListener("click", importPly);
  $("plyFile").addEventListener("change", () => {
    const file = $("plyFile").files[0];
    if (file && !$("newTitle").value) $("newTitle").value = file.name.replace(/\.ply$/i, "");
  });
  $("sceneSelect").addEventListener("change", () => selectScene($("sceneSelect").value));

  $("generateBtn").addEventListener("click", startGeneration);
  $("playDraftBtn").addEventListener("click", playDraft);

  // player bar
  $("playBtn").addEventListener("click", () => {
    if (player.state === "playing") player.pause();
    else player.play();
  });
  $("prevBtn").addEventListener("click", () => player.previous());
  $("nextBtn").addEventListener("click", () => player.next());
  $("replayBtn").addEventListener("click", () => player.replay());

  document.addEventListener("visibilitychange", () => {
    if (document.hidden && player.state === "playing") player.pause();
  });

  // a freshly saved storyboard becomes the playable snapshot immediately
  document.addEventListener("storyboard-saved", (event) => {
    const saved = event.detail;
    if (!state.scene) return;
    setPlayerSnapshot(saved, state.scene);
    player.load(saved, state.scene);
  });

  window.addEventListener("beforeunload", (event) => {
    if (state.sceneDirty) {
      event.preventDefault();
      event.returnValue = "有未保存的标注修改";
    }
  });
}

// ---------------------------------------------------------------- scenes
async function reloadScenes() {
  const data = await getJSON("/api/tour/scenes");
  const select = $("sceneSelect");
  const previous = select.value;
  select.innerHTML = '<option value="">（未选择）</option>' +
    data.scenes.map((s) => `<option value="${s.scene_id}">${escapeAttr(s.title)}（${s.source_kind === "synthetic" ? "合成" : "真实"}）</option>`).join("");
  if (previous && [...select.options].some((o) => o.value === previous)) {
    select.value = previous;
  } else if (data.scenes.length && !state.scene) {
    select.value = data.scenes[0].scene_id;
    await selectScene(data.scenes[0].scene_id);
  }
}

async function selectScene(sceneId) {
  stopPlayer();
  if (state.sceneDirty && state.scene && state.scene.scene_id !== sceneId) {
    const leave = window.confirm("当前场景有未保存的标注修改，切换后将丢失。确定切换吗？");
    if (!leave) {
      $("sceneSelect").value = state.scene.scene_id;
      return;
    }
  }
  if (!sceneId) {
    setState({ scene: null, preview: null, selectedEntityId: null, selectedViewId: null, sceneDirty: false });
    renderSceneUI();
    return;
  }
  try {
    const scene = await getJSON(`/api/tour/scenes/${encodeURIComponent(sceneId)}`);
    const preview = await getJSON(`/api/tour/scenes/${encodeURIComponent(sceneId)}/preview`);
    await viewer.loadScene(scene, preview);
    setState({
      scene,
      preview,
      previewTransform: { center: preview.display_center, scale: preview.display_scale },
      selectedEntityId: null,
      selectedViewId: null,
      sceneDirty: false,
    });
    $("sourceBadge").textContent = scene.source_kind === "synthetic" ? "合成场景" : "真实场景";
    $("sourceBadge").classList.remove("hidden");
    renderSceneUI();
    await refreshStoryboardList();
    newStoryboardDraft();
  } catch (err) {
    showFatal(`加载场景失败: ${err.message}`);
  }
}

async function importPly() {
  const file = $("plyFile").files[0];
  if (!file) {
    $("importStatus").textContent = "请选择 .ply 文件";
    return;
  }
  const title = $("newTitle").value.trim() || file.name;
  $("importStatus").textContent = "导入中…";
  try {
    const scene = await uploadPLY(file, title);
    $("importStatus").textContent = `已导入 ${scene.scene_id} ✓`;
    await reloadScenes();
    $("sceneSelect").value = scene.scene_id;
    await selectScene(scene.scene_id);
  } catch (err) {
    $("importStatus").textContent = `导入失败: ${err.message}`;
  }
}

function renderSceneUI() {
  const sceneName = $("sceneName");
  if (!state.scene) {
    sceneName.textContent = "未选择场景";
    renderEntities();
    renderViews();
    renderStoryboardDraft();
    renderRequiredEntities();
    renderOpeningView();
    updateViewerBoxes(viewer);
    return;
  }
  sceneName.textContent = `${state.scene.title} · rev ${state.scene.revision}`;
  $("saveStatus").textContent = "";
  renderEntities();
  renderViews();
  renderRequiredEntities();
  renderOpeningView();
  updateViewerBoxes(viewer);
}

// ---------------------------------------------------------------- playback
function startPlayback(storyboard, scene) {
  try {
    player.load(storyboard, scene);
    $("playBtn").textContent = "▶";
    player.play();
  } catch (err) {
    $("playerState").textContent = "无法播放: " + err.message;
  }
}

function stopPlayer() {
  player.stop();
  $("subtitle").classList.add("hidden");
  $("subtitle").textContent = "";
  updateViewerBoxes(viewer);
  if (viewer) viewer.setInteractionEnabled(true);
  $("playerState").textContent = "";
  $("shotInfo").textContent = "- / -";
  $("timeInfo").textContent = "0.0s / 0.0s";
}

function onPlayerShot({ index, shot, count, totalElapsed }) {
  if (!state.playerSnapshot) return;
  const { storyboard, scene } = state.playerSnapshot;
  const view = scene.views.find((v) => v.view_id === shot.view_id);
  if (view && viewer) viewer.setView(view);
  if (viewer) {
    viewer.showEntityBoxes(scene.entities, { highlightIds: shot.focus_entity_ids });
  }
  const subtitle = $("subtitle");
  subtitle.textContent = shot.narration;
  subtitle.classList.remove("hidden");
  $("shotInfo").textContent = `${index + 1} / ${count}`;
}

function onPlayerTick({ shotIndex, shotElapsed, shotDuration, totalElapsed }) {
  const total = state.playerSnapshot?.storyboard.shots.reduce((sum, s) => sum + s.duration_seconds, 0) || 0;
  $("timeInfo").textContent = `${totalElapsed.toFixed(1)}s / ${total.toFixed(1)}s`;
}

function onPlayerState(playerState) {
  $("playerState").textContent = {
    idle: "已停止",
    playing: "播放中",
    paused: "已暂停",
    finished: "播放完成",
  }[playerState] || "";
  $("playBtn").textContent = playerState === "playing" ? "⏸" : "▶";
  if (viewer) viewer.setInteractionEnabled(playerState !== "playing");
  if (playerState === "idle") {
    $("subtitle").classList.add("hidden");
    updateViewerBoxes(viewer);
  }
}

function setPlayerSnapshot(storyboard, scene) {
  setState({ playerSnapshot: { storyboard, scene } });
}

// ---------------------------------------------------------------- storyboard load
async function loadStoryboardById(storyboardId) {
  stopPlayer();
  try {
    const storyboard = await getJSON(`/api/tour/storyboards/${encodeURIComponent(storyboardId)}`);
    if (!state.scene || state.scene.scene_id !== storyboard.scene_id) {
      $("sceneSelect").value = storyboard.scene_id;
      await selectScene(storyboard.scene_id);
    }
    loadDraftFromStoryboard(storyboard);
    setPlayerSnapshot(storyboard, state.scene);
    player.load(storyboard, state.scene); // ready for the ▶ button
    $("sbStatus").textContent = `已载入 ${storyboard.title}（rev ${storyboard.revision}），可播放或编辑`;
    $("sbStatus").className = "hint";
  } catch (err) {
    $("sbStatus").textContent = `载入失败: ${err.message}`;
    $("sbStatus").className = "hint error-text";
  }
}

// ---------------------------------------------------------------- generation
async function startGeneration() {
  if (!state.scene) return;
  stopPlayer();
  const status = $("genStatus");
  const required = [...document.querySelectorAll(".required-check:checked")].map((c) => c.value);
  const opening = $("openingViewSel").value || null;
  const request = {
    scene_id: state.scene.scene_id,
    scene_revision: state.scene.revision,
    user_prompt: $("prompt").value.trim(),
    language: $("langSel").value,
    target_duration_seconds: Number($("durationSel").value),
    shot_count: Number($("shotCountSel").value),
    required_entity_ids: required,
    opening_view_id: opening,
    request_id: `gen_${Date.now().toString(36)}`,
  };
  if (!request.user_prompt) {
    status.textContent = "请填写导览要求";
    status.className = "hint error-text";
    return;
  }
  status.textContent = "提交生成…";
  status.className = "hint";
  try {
    const result = await postJSON("/api/tour/generations", request);
    status.textContent = `运行 ${result.run_id} 已启动`;
    $("generateBtn").disabled = true;
    pollTimer = setInterval(() => pollRun(result.run_id), 1000);
  } catch (err) {
    status.textContent = `生成失败: ${err.message}`;
    status.className = "hint error-text";
  }
}

async function pollRun(runId) {
  try {
    const run = await getJSON(`/api/tour/runs/${encodeURIComponent(runId)}`);
    const status = $("genStatus");
    if (run.status === "pending" || run.status === "running") {
      status.textContent = `生成中…（${run.attempts?.length || 0} 次调用）`;
      return;
    }
    clearInterval(pollTimer);
    pollTimer = null;
    $("generateBtn").disabled = false;
    if (run.status === "succeeded") {
      status.textContent = "生成成功 ✓";
      status.className = "hint ok-text";
      const storyboard = await getJSON(`/api/tour/storyboards/${run.result_storyboard_id}`);
      loadDraftFromStoryboard(storyboard);
      setPlayerSnapshot(storyboard, state.scene);
      await refreshStoryboardList();
    } else if (run.status === "needs_clarification") {
      status.textContent = `需要澄清: ${run.clarification_question}`;
      status.className = "hint error-text";
    } else {
      status.textContent = `失败(${run.error_code}): ${run.error_message || "未知错误"}`;
      status.className = "hint error-text";
    }
  } catch (err) {
    clearInterval(pollTimer);
    pollTimer = null;
    $("generateBtn").disabled = false;
    $("genStatus").textContent = `轮询失败: ${err.message}`;
  }
}

// ---------------------------------------------------------------- live boxes
subscribe((s) => {
  if (viewer && s.scene && !s.playerSnapshot) {
    // keep boxes in sync when the selection changes outside playback
    const playing = player.state === "playing" || player.state === "paused" || player.state === "finished";
    if (!playing) {
      viewer.showEntityBoxes(s.scene.entities, { selectedId: s.selectedEntityId });
    }
  }
});

// ---------------------------------------------------------------- draft playback
async function playDraft() {
  if (!state.scene) return;
  const draft = getStoryboardDraft();
  if (!isDraftPlayable()) {
    $("sbStatus").textContent = "请先添加分镜";
    $("sbStatus").className = "hint error-text";
    return;
  }
  try {
    const result = await postJSON("/api/tour/storyboards/validate", {
      scene_id: state.scene.scene_id,
      scene_revision: state.scene.revision,
      title: draft.title,
      language: draft.language,
      shots: draft.shots,
    });
    if (result.status !== "valid") {
      $("sbStatus").textContent = "校验未通过: " + result.errors.join("；");
      $("sbStatus").className = "hint error-text";
      return;
    }
    const snapshot = {
      storyboard_id: draft.storyboard_id || "draft",
      revision: draft.revision || 1,
      scene_id: state.scene.scene_id,
      scene_revision: state.scene.revision,
      asset_sha256: state.scene.asset_sha256,
      title: draft.title,
      language: draft.language,
      source: "manual",
      shots: draft.shots,
      validation: { schema_version: 1, status: "valid", errors: [], warnings: result.warnings || [] },
    };
    setPlayerSnapshot(snapshot, state.scene);
    startPlayback(snapshot, state.scene);
  } catch (err) {
    $("sbStatus").textContent = `播放失败: ${err.message}`;
    $("sbStatus").className = "hint error-text";
  }
}

function escapeAttr(value) {
  const div = document.createElement("div");
  div.textContent = value ?? "";
  return div.innerHTML.replaceAll('"', "&quot;");
}

boot();
