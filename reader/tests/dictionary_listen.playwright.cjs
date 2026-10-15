// Requires the isolated fixture server and a local Playwright installation:
//   .venv\Scripts\python.exe tests\ui_fixture_server.py
//   node tests\dictionary_listen.playwright.cjs
const { chromium } = require("playwright");
const assert = require("node:assert/strict");

(async () => {
  const base = process.env.AURIS_QA_URL || "http://127.0.0.1:17861";
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  const errors = [];
  const listenStatuses = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
  });
  page.on("response", (r) => {
    if (r.url().includes("/api/pronunciation/listen")) listenStatuses.push(r.status());
  });

  try {
    const marker = await page.request.get(base + "/api/qa-fixture");
    assert.equal(
      marker.ok() && (await marker.json()).fixture,
      true,
      "Requires the isolated fixture server",
    );

    await page.goto(base + "/settings#dictionary");
    await page.locator("#settings-dictionary").waitFor({ state: "visible" });

    const button = page.locator("#dictionary-listen");
    assert.equal(await button.count(), 1, "a hallgatás-gomb megjelenik");
    assert.equal((await button.textContent()).trim(), "Hallgasd meg, hogyan ejti");

    // Üres űrlap: szabály nélkül nincs mit meghallgatni, API-hívás se történik.
    await button.click();
    await page.waitForFunction(() =>
      /Add meg az írott/.test(document.getElementById("dictionary-message").textContent),
    );
    assert.equal(listenStatuses.length, 0, "üres űrlap nem hív API-t");

    // Kitöltött szabály: a mentés ELKÉSZÜLÉSE előtt megszólal a felolvasási alak.
    await page.locator("#dictionary-source").fill("Gorcsev");
    await page.locator("#dictionary-replacement").fill("Gorcsef");
    const audioResponse = page.waitForResponse(
      (r) => r.url().includes("/api/audio/"),
      { timeout: 30000 },
    );
    await button.click();
    await page.waitForFunction(
      () => /Így hangzik:/.test(document.getElementById("dictionary-message").textContent),
      null,
      { timeout: 30000 },
    );
    const message = await page.locator("#dictionary-message").textContent();
    assert.match(message, /Gorcsef belépett a szobába/, "a szabály alkalmazva szól");
    assert.deepEqual(listenStatuses, [200], "a meghallgatás végpontja 200-at ad");
    const player = page.locator("#dictionary-audio");
    assert.equal(await player.isVisible(), true, "a lejátszó megjelenik");
    assert.match(await player.getAttribute("src"), /\/api\/audio\//, "a hang beágyazódik");
    // A médialejátszó tartománykérést (206) is küldhet, ezért mindkettő elfogadott.
    assert.ok(
      [200, 206].includes((await audioResponse).status()),
      "a hang kiszolgálásra kerül",
    );

    // A meghallgatás nem ment szabályt: a lista még üres.
    const rules = await page.request.get(base + "/api/pronunciation");
    assert.deepEqual(await rules.json(), [], "a meghallgatás nem hoz létre szabályt");

    await page.screenshot({ path: process.env.AURIS_QA_SHOT || "dictionary-listen.png" });
    assert.deepEqual(errors, [], "nincs konzolhiba");
    console.log("OK — hallgatás-gomb működik:", message.trim());
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
