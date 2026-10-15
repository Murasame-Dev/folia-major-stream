const { app, BrowserWindow, ipcMain } = require('electron');
const fs = require('fs');
const os = require('os');
const path = require('path');

// dev/beat-this-wgsl/main.cjs
// Test shell for the hand-written WGSL Beat This! runtime: a hidden window runs every chunk through
// the real WebGPU of this Electron build and writes the logits (and, optionally, every stage of
// chunk 0) next to the inputs. Never touches the app's own profile - userData is a throwaway dir.
//
//   npx rolldown dev/beat-this-wgsl/harness.ts -o $OUT/harness.js -f iife -p browser
//   BTW_DIR=$OUT BTW_WEIGHTS=... BTW_CHUNKS=chunks.f32 [BTW_STAGES=1] [BTW_REPEAT=5] npx electron dev/beat-this-wgsl/main.cjs
//
// Exit code 0 when the run completed; correctness is judged by the comparison script, not here.

const dir = process.env.BTW_DIR;
if (!dir) {
    console.error('BTW_DIR is required');
    process.exit(2);
}
app.setPath('userData', fs.mkdtempSync(path.join(os.tmpdir(), 'btw-userdata-')));

ipcMain.handle('btw-config', () => ({
    weights: process.env.BTW_WEIGHTS,
    chunks: process.env.BTW_CHUNKS,
    out: dir,
    stages: process.env.BTW_STAGES === '1',
    repeat: Number(process.env.BTW_REPEAT || 5),
    songs: (process.env.BTW_SONGS || '').split(',').filter(Boolean),
}));
ipcMain.on('btw-log', (_event, line) => console.log(line));
ipcMain.on('btw-done', (_event, code) => app.exit(code));

app.whenReady().then(() => {
    const html = path.join(dir, 'harness.html');
    fs.writeFileSync(html, '<!doctype html><meta charset="utf-8"><script src="harness.js"></script>');
    const win = new BrowserWindow({ show: false, webPreferences: { nodeIntegration: true, contextIsolation: false } });
    win.webContents.on('console-message', (event) => console.log(`[renderer] ${event.message}`));
    win.webContents.on('did-fail-load', (_event, code, description) => {
        console.error('load failed', code, description);
        app.exit(4);
    });
    win.webContents.on('render-process-gone', (_event, details) => {
        console.error('renderer gone', details);
        app.exit(3);
    });
    win.loadFile(html);
});
