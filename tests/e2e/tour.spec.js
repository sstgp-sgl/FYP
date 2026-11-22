// @ts-check
// Browser acceptance for the scene tour. Prerequisites:
//   python scripts/40_tour.py seed-demo
//   python app.py --tour-only --host 127.0.0.1 --port 7861
// Browsers: npx playwright install chromium
const { test, expect } = require("@playwright/test");

const BASE = process.env.E2E_BASE_URL || "http://127.0.0.1:7861";

function makePly(count = 200, seed = 5) {
  // Deterministic small ASCII PLY in the [-2,2]^3 cube.
  const lines = [
    "ply",
    "format ascii 1.0",
    `element vertex ${count}`,
    "property float x",
    "property float y",
    "property float z",
    "property uchar red",
    "property uchar green",
    "property uchar blue",
    "end_header",
  ];
  let x = seed;
  for (let i = 0; i < count; i++) {
    x = (x * 1664525 + 1013904223) >>> 0;
    const a = (x / 2 ** 32) * 4 - 2;
    x = (x * 1664525 + 1013904223) >>> 0;
    const b = (x / 2 ** 32) * 4 - 2;
    x = (x * 1664525 + 1013904223) >>> 0;
    const c = (x / 2 ** 32) * 4 - 2;
    const r = (x >> 8) & 0xff;
    const g = (x >> 16) & 0xff;
    const bl = (x >> 24) & 0xff;
    lines.push(`${a.toFixed(4)} ${b.toFixed(4)} ${c.toFixed(4)} ${r} ${g} ${bl}`);
  }
  return lines.join("\n");
}

test.describe.configure({ mode: "serial" });

let page;
let consoleErrors = [];
let pageErrors = [];
let externalRequests = [];
let uploadedTitle = `上传夹具${Date.now().toString(36)}`;

test.beforeEach(async ({ browser }) => {
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  consoleErrors = [];
  pageErrors = [];
  externalRequests = [];
  context.on("page", (p) => p.on("pageerror", (e) => pageErrors.push(String(e))));
  await context.route("**/*", (route) => {
    const url = new URL(route.request().url());
    if (url.hostname === "127.0.0.1" || url.hostname === "localhost") return route.continue();
    externalRequests.push(url.href);
    return route.abort();
  });
  page = await context.newPage();
  page.on("console", (msg) => {
    if (msg.type() === "error") consoleErrors.push(msg.text());
  });
  page.on("pageerror", (e) => pageErrors.push(String(e)));
  if (process.env.E2E_DEBUG_REQUESTS) {
    page.on("request", (req) => {
      if (req.method() !== "GET" && req.url().includes("/api/tour/")) {
        console.log(`REQ ${req.method()} ${req.url()} ${(req.postData() || "").slice(0, 200)}`);
      }
    });
  }
});

test.afterEach(async () => {
  await page.context().close();
});

test("1 · tour page loads, WebGL works, geometry visible, no console errors, no external requests", async () => {
  await page.goto(`${BASE}/tour`);
  await page.selectOption("#sceneSelect", { label: "合成庭院演示（合成）" });
  await expect(page.locator("#sceneName")).toContainText("合成庭院演示", { timeout: 15000 });
  await expect(page.locator("#sourceBadge")).toHaveText("合成场景");
  // WebGL renderer exists and the demo point cloud is loaded
  await page.waitForFunction(() => window.__tourViewer && window.__tourViewer.points);
  const info = await page.evaluate(() => ({
    hasRenderer: Boolean(window.__tourViewer.renderer),
    pointCount: window.__tourViewer.points.geometry.attributes.position.count,
    canvasCount: document.querySelectorAll("#canvasHost canvas").length,
  }));
  expect(info.hasRenderer).toBe(true);
  expect(info.pointCount).toBeGreaterThan(500);
  expect(info.canvasCount).toBe(1);
  expect(consoleErrors).toEqual([]);
  expect(pageErrors).toEqual([]);
  expect(externalRequests).toEqual([]);
  await page.screenshot({ path: "tests/e2e/artifacts/tour-demo-loaded.png" });
});

test("2 · import fixture PLY, annotate entity, record view, confirm, save, refresh restores", async () => {
  await page.goto(`${BASE}/tour`);
  await page.setInputFiles("#plyFile", {
    name: "fixture.ply",
    mimeType: "application/octet-stream",
    buffer: Buffer.from(makePly()),
  });
  await page.fill("#newTitle", uploadedTitle);
  await page.click("#importBtn");
  await expect(page.locator("#sceneName")).toContainText(uploadedTitle, { timeout: 15000 });
  await expect(page.locator("#sourceBadge")).toHaveText("真实场景");

  // add entity with facts
  await page.click("#addEntityBtn");
  const entity = page.locator(".entity").last();
  await entity.locator("input[data-field='label']").fill("夹具区域");
  await entity.locator("input[data-field='bound:x:min']").fill("-1.2");
  await entity.locator("input[data-field='bound:y:min']").fill("-1.2");
  await entity.locator("input[data-field='bound:z:min']").fill("-1.2");
  await entity.locator("input[data-field='bound:x:max']").fill("1.2");
  await entity.locator("input[data-field='bound:y:max']").fill("1.2");
  await entity.locator("input[data-field='bound:z:max']").fill("1.2");
  await entity.locator("button[data-act=fact-add]").click();
  await entity.locator("input[data-field='fact_text']").fill("这是上传点云的人工标注区域。");
  await entity.locator("select.status-sel").selectOption("confirmed");

  // record a view and approve the entity
  await page.fill("#viewName", "夹具全景");
  await page.click("#saveViewBtn");
  const view = page.locator(".view").last();
  await expect(view.locator("b")).toHaveText("🎥 夹具全景");
  await view.locator("input.approved-check").check();

  await page.click("#saveAnnotationsBtn");
  await expect(page.locator("#saveStatus")).toContainText("已保存 ✓（rev 2）");

  // refresh must restore scene, entity and view
  await page.reload();
  await page.selectOption("#sceneSelect", { label: `${uploadedTitle}（真实）` });
  await expect(page.locator("#sceneName")).toContainText(uploadedTitle);
  await expect(page.locator(".entity")).toHaveCount(1);
  await expect(page.locator(".entity input[data-field='label']")).toHaveValue("夹具区域");
  await expect(page.locator(".view")).toHaveCount(1);
  await expect(page.locator(".view input.approved-check")).toBeChecked();
  expect(consoleErrors).toEqual([]);
  expect(externalRequests).toEqual([]);
});

test("3 · three-shot storyboard plays: subtitles, shot index and time advance", async () => {
  await page.goto(`${BASE}/tour`);
  await page.selectOption("#sceneSelect", { label: "合成庭院演示（合成）" });
  await expect(page.locator("#sceneName")).toContainText("合成庭院演示");

  // load the seeded storyboard and shorten each shot to 2s for the real-time check
  await page.locator("#sbList button").first().click();
  await expect(page.locator("#sbStatus")).toContainText("已载入 庭院导览");
  const durations = page.locator("#shotList input[data-field='duration']");
  await expect(durations).toHaveCount(3);
  for (let i = 0; i < 3; i++) {
    await durations.nth(i).fill("2");
  }
  await page.click("#validateSbBtn");
  await expect(page.locator("#sbStatus")).toContainText("校验通过");
  await page.click("#saveSbBtn");
  await expect(page.locator("#sbStatus")).toContainText("已保存");

  await page.click("#playBtn");
  await expect(page.locator("#playerState")).toHaveText("播放中");
  await expect(page.locator("#shotInfo")).toHaveText("1 / 3");
  await expect(page.locator("#subtitle")).toContainText("俯瞰庭院");
  await expect(page.locator("#subtitle")).toBeVisible();

  await expect(page.locator("#shotInfo")).toHaveText("2 / 3", { timeout: 6000 });
  await expect(page.locator("#subtitle")).toContainText("红色建筑");
  await expect(page.locator("#shotInfo")).toHaveText("3 / 3", { timeout: 6000 });
  await expect(page.locator("#subtitle")).toContainText("树丛");
  await expect(page.locator("#playerState")).toHaveText("播放完成", { timeout: 6000 });
  // finished keeps the last frame's subtitle
  await expect(page.locator("#subtitle")).toContainText("树丛");
  expect(consoleErrors).toEqual([]);
  expect(externalRequests).toEqual([]);
});

test("4 · pause freezes time, resume continues, prev/next/replay behave", async () => {
  await page.goto(`${BASE}/tour`);
  await page.selectOption("#sceneSelect", { label: "合成庭院演示（合成）" });
  await page.locator("#sbList button").first().click();
  await expect(page.locator("#sbStatus")).toContainText("已载入");

  await page.click("#playBtn");
  await page.waitForTimeout(600);
  await page.click("#playBtn"); // pause
  await expect(page.locator("#playerState")).toHaveText("已暂停");
  const frozen = await page.locator("#timeInfo").textContent();
  await page.waitForTimeout(700);
  await expect(page.locator("#timeInfo")).toHaveText(frozen); // clock frozen
  await page.click("#playBtn"); // resume, no restart
  await expect(page.locator("#playerState")).toHaveText("播放中");
  await page.waitForTimeout(500);
  await expect(page.locator("#timeInfo")).not.toHaveText(frozen);

  await page.click("#playBtn"); // pause again for deterministic stepping
  await expect(page.locator("#playerState")).toHaveText("已暂停");
  await page.click("#nextBtn");
  await expect(page.locator("#shotInfo")).toHaveText("2 / 3");
  await expect(page.locator("#playerState")).toHaveText("已暂停"); // stays paused
  await page.click("#prevBtn");
  await expect(page.locator("#shotInfo")).toHaveText("1 / 3");

  await page.click("#nextBtn");
  await page.click("#nextBtn");
  await expect(page.locator("#shotInfo")).toHaveText("3 / 3");
  await page.click("#nextBtn"); // at the end while paused: stays on the last shot
  await expect(page.locator("#shotInfo")).toHaveText("3 / 3");

  await page.click("#replayBtn");
  await expect(page.locator("#playerState")).toHaveText("播放中");
  await expect(page.locator("#shotInfo")).toHaveText("1 / 3");
  await page.click("#playBtn");
  await expect(page.locator("#playerState")).toHaveText("已暂停");
  expect(consoleErrors).toEqual([]);
});

test("5 · entity bounds edit invalidates view confirmations; invalid storyboard cannot play", async () => {
  await page.goto(`${BASE}/tour`);
  await page.selectOption("#sceneSelect", { label: "合成庭院演示（合成）" });
  await expect(page.locator("#sceneName")).toContainText("合成庭院演示");

  // the seeded overview view approves entity_tree; change tree bounds → approval cleared
  await page.locator(".entity input[data-field='label']").nth(1).waitFor();
  const treeInput = page.locator(".entity").nth(1).locator("input[data-field='bound:x:max']");
  const before = await treeInput.inputValue();
  const changed = String(Number(before) + 0.1);
  await treeInput.fill(changed);
  await page.click("#saveAnnotationsBtn");
  await expect(page.locator("#saveStatus")).toContainText("已保存 ✓（rev 2）");
  const overviewView = page.locator(".view", { hasText: "全景" });
  await overviewView.locator("input.approved-check[value=entity_tree]").waitFor();
  await expect(overviewView.locator("input.approved-check[value=entity_tree]")).not.toBeChecked();

  // the stored storyboard was built on the previous scene revision: loading it
  // into the player is refused until it is re-validated and saved again
  await page.locator("#sbList button").first().click();
  await expect(page.locator("#sbStatus")).toContainText("旧场景版本");

  // craft an invalid draft: shot 1 uses the building view and cites the
  // view-scoped fact fact_building_west (scope = view_overview)
  await page.locator(".shot").first().locator("select.view-select").selectOption("view_building");
  await page.locator(".shot").first().locator("input.fact-check[value=fact_building_west]").check();
  await page.click("#validateSbBtn");
  await expect(page.locator("#sbStatus")).toContainText("view-dependent fact");
  // play draft is blocked
  await page.click("#playDraftBtn");
  await expect(page.locator("#sbStatus")).toContainText("校验未通过");
  await expect(page.locator("#playerState")).not.toHaveText("播放中");

  // fix the draft, re-confirm the tree from its closeup view (bounds changed,
  // so every approval was invalidated), re-validate and re-save
  await page.locator(".shot").first().locator("input.fact-check[value=fact_building_west]").uncheck();
  const treeView = page.locator(".view", { hasText: "树丛近景" });
  await treeView.locator("input.approved-check[value=entity_tree]").check();
  await page.click("#saveAnnotationsBtn");
  await expect(page.locator("#saveStatus")).toContainText("已保存 ✓（rev 3）");
  await page.click("#validateSbBtn");
  await expect(page.locator("#sbStatus")).toContainText("校验通过");
  await page.click("#saveSbBtn");
  await expect(page.locator("#sbStatus")).toContainText("已保存");
  await page.click("#playBtn");
  await expect(page.locator("#playerState")).toHaveText("播放中");
  await expect(page.locator("#shotInfo")).toHaveText("1 / 3");
  await page.click("#playBtn"); // stop the run
  await expect(page.locator("#playerState")).toHaveText("已暂停");
  expect(consoleErrors).toEqual([]);
});

test("6 · generation without LLM config fails cleanly and keeps the scene revision", async () => {
  await page.goto(`${BASE}/tour`);
  await page.selectOption("#sceneSelect", { label: "合成庭院演示（合成）" });
  await expect(page.locator("#sceneName")).toContainText("合成庭院演示");
  const sceneNameBefore = await page.locator("#sceneName").textContent();
  await page.fill("#prompt", "先整体，再建筑，最后树丛，三段30秒");
  await page.click("#generateBtn");
  await expect(page.locator("#genStatus")).toContainText("失败(llm_not_configured)", { timeout: 15000 });
  await expect(page.locator("#sceneName")).toHaveText(sceneNameBefore);
  await expect(page.locator("#generateBtn")).toBeEnabled();
});

test("7 · repeated scene switching keeps a single render loop and disposes old objects", async () => {
  await page.goto(`${BASE}/tour`);
  await page.selectOption("#sceneSelect", { label: "合成庭院演示（合成）" });
  await expect(page.locator("#sceneName")).toContainText("合成庭院演示");
  for (let i = 0; i < 3; i++) {
    await page.selectOption("#sceneSelect", { label: `${uploadedTitle}（真实）` });
    await expect(page.locator("#sceneName")).toContainText(uploadedTitle);
    await page.selectOption("#sceneSelect", { label: "合成庭院演示（合成）" });
    await expect(page.locator("#sceneName")).toContainText("合成庭院演示");
  }
  const info = await page.evaluate(() => ({
    canvasCount: document.querySelectorAll("#canvasHost canvas").length,
    points: window.__tourViewer && window.__tourViewer.points
      ? window.__tourViewer.points.geometry.attributes.position.count
      : -1,
    boxes: window.__tourViewer ? window.__tourViewer.boxes.size : -1,
  }));
  expect(info.canvasCount).toBe(1);
  expect(info.points).toBeGreaterThan(500);
  expect(info.boxes).toBe(3); // demo entities re-rendered exactly once
  expect(consoleErrors).toEqual([]);
  expect(pageErrors).toEqual([]);
});

test("8 · fixed 16:9 letterbox survives window resize", async () => {
  await page.goto(`${BASE}/tour`);
  await page.selectOption("#sceneSelect", { label: "合成庭院演示（合成）" });
  await expect(page.locator("#sceneName")).toContainText("合成庭院演示");
  const measure = () =>
    page.evaluate(() => {
      const canvas = document.querySelector("#canvasHost canvas");
      return { w: canvas.clientWidth, h: canvas.clientHeight };
    });
  for (const viewport of [
    { width: 1600, height: 700 },
    { width: 1000, height: 1400 },
    { width: 1440, height: 900 },
  ]) {
    await page.setViewportSize(viewport);
    await page.waitForTimeout(150);
    const { w, h } = await measure();
    expect(w).toBeGreaterThan(0);
    expect(Math.abs(w / h - 16 / 9)).toBeLessThan(0.01);
  }
  expect(consoleErrors).toEqual([]);
});
