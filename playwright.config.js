// @ts-check
const { defineConfig } = require("@playwright/test");

// The CPU tour server must already be reachable at baseURL with a freshly
// seeded state (tests mutate scenes/storyboards and assume revision 1):
//   rm -rf outputs/tours-e2e
//   FYP_TOUR_OUTPUT_ROOT=outputs/tours-e2e python scripts/40_tour.py seed-demo
//   FYP_TOUR_OUTPUT_ROOT=outputs/tours-e2e python app.py --tour-only \
//     --host 127.0.0.1 --port 7861
// Browsers are downloaded once with `npx playwright install chromium`.
module.exports = defineConfig({
  testDir: "./tests/e2e",
  timeout: 60_000,
  expect: { timeout: 10_000 },
  fullyParallel: false,
  workers: 1,
  retries: 0,
  use: {
    baseURL: process.env.E2E_BASE_URL || "http://127.0.0.1:7861",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    viewport: { width: 1440, height: 900 },
  },
  reporter: [["list"]],
  projects: [{ name: "chromium", use: { browserName: "chromium" } }],
});
