// Rebuild the native launch artwork from the app's code-native SVG mark.
// Run from mobile/: node scripts/native-splash.mjs
import { chromium } from '@playwright/test';
import { readFile, readdir, stat } from 'node:fs/promises';
import { resolve } from 'node:path';

const icon = await readFile(resolve('public/icon.svg'), 'utf8');
const resources = resolve('android/app/src/main/res');
const files = [];
for (const directory of await readdir(resources)) {
  const file = resolve(resources, directory, 'splash.png');
  if (await stat(file).then(() => true).catch(() => false)) files.push(file);
}
const ios = resolve('ios/App/App/Assets.xcassets/Splash.imageset');
for (const name of await readdir(ios)) if (name.endsWith('.png')) files.push(resolve(ios, name));

const browser = await chromium.launch({
  executablePath: process.env.CHROME_PATH || '/bin/google-chrome',
  args: ['--no-sandbox'],
});
try {
  const page = await browser.newPage({ deviceScaleFactor: 1 });
  for (const file of files) {
    const png = await readFile(file);
    const width = png.readUInt32BE(16), height = png.readUInt32BE(20);
    // The square iOS artwork is cropped by LaunchScreen's scaleAspectFill.
    // A modest center mark remains visible in portrait and landscape crops.
    const mark = Math.round(Math.min(width, height) * 0.15);
    await page.setViewportSize({ width, height });
    await page.setContent(`<style>html,body{margin:0;width:100%;height:100%;background:#f5f6f0}body{display:grid;place-items:center}svg{width:${mark}px;height:${mark}px}</style>${icon}`);
    await page.screenshot({ path: file });
    process.stdout.write(`Branded ${file}\n`);
  }
} finally {
  await browser.close();
}
