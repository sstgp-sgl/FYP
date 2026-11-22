// Single-page UI state: current scene / revision / selection / drafts.
// No network access here; api.js and the feature modules own their I/O.
const listeners = new Set();

export const state = {
  capabilities: null,
  scenes: [],
  scene: null,            // loaded TourScene (server truth, incl. revision)
  sceneDirty: false,
  preview: null,          // preview metadata
  previewTransform: null, // {center:[..], scale}
  selectedEntityId: null,
  selectedViewId: null,
  storyboards: [],
  loadedStoryboard: null, // storyboard JSON being viewed/edited
  activeRun: null,        // polled generation run
  playerSnapshot: null,   // storyboard+scene frozen for playback
  errors: [],
};

export function setState(patch) {
  Object.assign(state, patch);
  notify();
}

export function notify() {
  listeners.forEach((fn) => {
    try {
      fn(state);
    } catch (err) {
      console.error("state listener failed", err);
    }
  });
}

// Debug/playwright introspection; harmless in production.
if (typeof window !== "undefined") {
  window.__tourState = state;
}

export function subscribe(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

export function markDirty() {
  setState({ sceneDirty: true });
}

export function clearDirty() {
  setState({ sceneDirty: false });
}

export function displayTransformOf(scene) {
  return scene?.display_transform || null;
}
