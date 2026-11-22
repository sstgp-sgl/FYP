// Three.js r128 viewer: single render loop, fixed 16:9 letterbox stage,
// entity AABB wireframes + labels, camera bookmarks. Asset<->display
// conversion uses the scene's persisted display transform only.
import { fetchBinary } from "./api.js";

const ASPECT = 16 / 9;
const BOX_COLORS = { normal: 0x8ab4f8, selected: 0xf0c040, highlight: 0xffd54d };

export class Viewer {
  constructor(host, stage) {
    this.host = host;
    this.stage = stage;
    this.canvasHost = null;
    this.renderer = null;
    this.scene = null;
    this.camera = null;
    this.controls = null;
    this.points = null;
    this.boxes = new Map();     // entity_id -> THREE.LineSegments
    this.labels = new Map();    // entity_id -> THREE.Sprite
    this.transform = null;
    this.rafId = 0;
    this.onResize = this.resize.bind(this);
    this.interactionEnabled = true;
  }

  init() {
    try {
      this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    } catch (err) {
      throw new Error("WebGL 不可用：" + (err && err.message ? err.message : err));
    }
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x111318);
    this.camera = new THREE.PerspectiveCamera(60, ASPECT, 0.001, 100);
    this.camera.position.set(0, 1.2, 2.2);
    this.canvasHost = document.createElement("div");
    this.canvasHost.style.position = "absolute";
    this.host.appendChild(this.canvasHost);
    this.canvasHost.appendChild(this.renderer.domElement);
    this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.target.set(0, 0, 0);
    window.addEventListener("resize", this.onResize);
    this.resize();
    const loop = () => {
      if (!this.renderer) return;
      this.rafId = requestAnimationFrame(loop);
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    };
    loop();
  }

  resize() {
    if (!this.renderer || !this.stage) return;
    const width = this.stage.clientWidth;
    const height = this.stage.clientHeight;
    if (!width || !height) return;
    let w = width;
    let h = width / ASPECT;
    if (h > height) {
      h = height;
      w = height * ASPECT;
    }
    this.canvasHost.style.width = `${w}px`;
    this.canvasHost.style.height = `${h}px`;
    this.canvasHost.style.left = `${(width - w) / 2}px`;
    this.canvasHost.style.top = `${(height - h) / 2}px`;
    this.renderer.setSize(Math.round(w), Math.round(h), false);
    this.camera.aspect = ASPECT; // fixed 16:9, letterbox instead of re-frame
    this.camera.updateProjectionMatrix();
  }

  async loadScene(scene, preview) {
    this.clearSceneObjects();
    this.transform = {
      center: preview.display_center,
      scale: preview.display_scale,
    };
    const [pointsBuf, colorsBuf] = await Promise.all([
      fetchBinary(preview.points_url),
      fetchBinary(preview.colors_url),
    ]);
    const positions = new Float32Array(pointsBuf);
    const colors8 = new Uint8Array(colorsBuf);
    const colors = new Float32Array(colors8.length);
    for (let i = 0; i < colors8.length; i++) colors[i] = colors8[i] / 255;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    const material = new THREE.PointsMaterial({
      size: 0.012,
      vertexColors: true,
      sizeAttenuation: true,
    });
    this.points = new THREE.Points(geometry, material);
    this.scene.add(this.points);
  }

  toDisplayPoint(assetPoint) {
    const { center, scale } = this.transform;
    return [
      (assetPoint[0] - center[0]) * scale,
      (assetPoint[1] - center[1]) * scale,
      (assetPoint[2] - center[2]) * scale,
    ];
  }

  getCameraInAssetCoordinates() {
    const { center, scale } = this.transform;
    const toAsset = (v) => [
      v.x / scale + center[0],
      v.y / scale + center[1],
      v.z / scale + center[2],
    ];
    const up = this.camera.up.clone().normalize();
    return {
      position_asset: toAsset(this.camera.position),
      target_asset: toAsset(this.controls.target),
      up: [up.x, up.y, up.z],
      fov_y_deg: this.camera.fov,
      near_asset: this.camera.near / scale,
      far_asset: this.camera.far / scale,
      aspect_ratio: ASPECT,
    };
  }

  setView(viewPreset) {
    const { scale } = this.transform;
    const p = this.toDisplayPoint(viewPreset.camera.position_asset);
    const t = this.toDisplayPoint(viewPreset.camera.target_asset);
    const up = viewPreset.camera.up;
    this.controls.enableDamping = false; // no residual motion / drift
    this.camera.position.set(p[0], p[1], p[2]);
    this.camera.up.set(up[0], up[1], up[2]);
    this.camera.fov = viewPreset.camera.fov_y_deg;
    this.camera.near = Math.max(viewPreset.camera.near_asset * scale, 1e-5);
    this.camera.far = viewPreset.camera.far_asset * scale;
    this.camera.aspect = ASPECT;
    this.camera.updateProjectionMatrix();
    this.camera.lookAt(t[0], t[1], t[2]);
    this.controls.target.set(t[0], t[1], t[2]);
    this.controls.update();
    this.controls.enableDamping = true;
  }

  setInteractionEnabled(enabled) {
    this.interactionEnabled = enabled;
    this.controls.enabled = enabled;
  }

  showEntityBoxes(entities, { selectedId = null, highlightIds = [] } = {}) {
    this.clearBoxes();
    const highlight = new Set(highlightIds);
    for (const entity of entities) {
      const lo = this.toDisplayPoint(entity.bounds_asset[0]);
      const hi = this.toDisplayPoint(entity.bounds_asset[1]);
      const center = new THREE.Vector3((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2);
      const size = new THREE.Vector3(Math.abs(hi[0] - lo[0]), Math.abs(hi[1] - lo[1]), Math.abs(hi[2] - lo[2]));
      const color = highlight.has(entity.entity_id)
        ? BOX_COLORS.highlight
        : entity.entity_id === selectedId
          ? BOX_COLORS.selected
          : BOX_COLORS.normal;
      const box = new THREE.Box3().setFromCenterAndSize(center, size);
      const helper = new THREE.Box3Helper(box, new THREE.Color(color));
      helper.renderOrder = 1;
      this.scene.add(helper);
      this.boxes.set(entity.entity_id, helper);
      const label = makeLabelSprite(entity.label, color);
      label.position.copy(center);
      label.position.y = hi[1] + 0.06;
      this.scene.add(label);
      this.labels.set(entity.entity_id, label);
    }
  }

  clearBoxes() {
    for (const object of this.boxes.values()) {
      this.scene.remove(object);
      if (object.geometry) object.geometry.dispose();
      if (object.material) object.material.dispose();
    }
    this.boxes.clear();
    for (const sprite of this.labels.values()) {
      this.scene.remove(sprite);
      if (sprite.material) {
        if (sprite.material.map) sprite.material.map.dispose();
        sprite.material.dispose();
      }
    }
    this.labels.clear();
  }

  clearSceneObjects() {
    this.clearBoxes();
    if (this.points) {
      this.scene.remove(this.points);
      this.points.geometry.dispose();
      this.points.material.dispose();
      this.points = null;
    }
  }

  dispose() {
    cancelAnimationFrame(this.rafId);
    window.removeEventListener("resize", this.onResize);
    this.clearSceneObjects();
    if (this.controls) {
      this.controls.dispose();
      this.controls = null;
    }
    if (this.renderer) {
      this.renderer.dispose();
      if (this.renderer.domElement.parentNode) {
        this.renderer.domElement.parentNode.removeChild(this.renderer.domElement);
      }
      this.renderer = null;
    }
    if (this.canvasHost && this.canvasHost.parentNode) {
      this.canvasHost.parentNode.removeChild(this.canvasHost);
    }
  }
}

function makeLabelSprite(text, colorHex) {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  const fontPx = 40;
  const padding = 16;
  ctx.font = `${fontPx}px -apple-system, "PingFang SC", sans-serif`;
  const width = Math.ceil(ctx.measureText(text).width) + padding * 2;
  const height = fontPx + padding * 2;
  canvas.width = width;
  canvas.height = height;
  ctx.font = `${fontPx}px -apple-system, "PingFang SC", sans-serif`;
  ctx.fillStyle = `rgba(10,12,16,0.75)`;
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = `#${colorHex.toString(16).padStart(6, "0")}`;
  ctx.lineWidth = 3;
  ctx.strokeRect(1.5, 1.5, width - 3, height - 3);
  ctx.fillStyle = "#ffffff";
  ctx.textBaseline = "middle";
  ctx.fillText(text, padding, height / 2 + 2);
  const texture = new THREE.CanvasTexture(canvas);
  const material = new THREE.SpriteMaterial({ map: texture, depthTest: false, depthWrite: false });
  const sprite = new THREE.Sprite(material);
  const worldHeight = 0.16;
  sprite.scale.set((width / height) * worldHeight, worldHeight, 1);
  sprite.renderOrder = 2;
  return sprite;
}
