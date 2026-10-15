/// <reference types="@webgpu/types" />
import { BeatThisWgsl, BEAT_THIS_WGSL_FRAMES, type BeatThisWgslManifest } from '../../src/services/automix/beatThisWgsl/runtime';
import { BEAT_THIS_SAMPLE_RATE, melSpectrogram, splitForModel } from '../../src/services/automix/beatThis';

// dev/beat-this-wgsl/harness.ts
// Renderer side of the test shell (see main.cjs): loads the exported weights and the mel chunks,
// runs each chunk once for the logits (plus every stage of chunk 0 when asked), then times
// `repeat` further passes over all chunks. Output: logits.f32 as [2, chunks, 1500] (beat then
// downbeat, the layout bt/cmp.py reads) and stage_<name>.f32.
//
// With BTW_SONGS (comma-separated audio paths) it instead runs whole tracks through the app's own
// front end (melSpectrogram + splitForModel) and writes, per track, the chunks, their placement and
// the WGSL logits, so compare-songs.ts can run the same chunks through ORT and diff the beat lists.

// Node's require, read off the window so the bundler leaves these imports to the renderer at runtime.
const nodeRequire = (globalThis as unknown as { require: (id: string) => any }).require;
const { ipcRenderer } = nodeRequire('electron');
const fs = nodeRequire('fs');
const path = nodeRequire('path');

const log = (line: string) => ipcRenderer.send('btw-log', line);

async function main() {
    const config = await ipcRenderer.invoke('btw-config');
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('no WebGPU adapter');
    const info = adapter.info;
    log(`adapter: ${info.vendor} ${info.architecture} ${info.description} f16=${adapter.features.has('shader-f16')}`);
    const device = await adapter.requestDevice();
    device.addEventListener('uncapturederror', (event) => log(`GPU error: ${(event as GPUUncapturedErrorEvent).error.message}`));

    const manifest: BeatThisWgslManifest = JSON.parse(fs.readFileSync(config.weights.replace(/\.bin$/, '.json'), 'utf8'));
    const raw = fs.readFileSync(config.weights);
    const weights = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
    if (config.songs.length) {
        await runSongs(device, weights, manifest, config.songs, config.out);
        ipcRenderer.send('btw-done', 0);
        return;
    }
    const chunkBytes = fs.readFileSync(config.chunks);
    const chunks = new Float32Array(chunkBytes.buffer, chunkBytes.byteOffset, chunkBytes.byteLength / 4);
    const perChunk = BEAT_THIS_WGSL_FRAMES * 128;
    const count = chunks.length / perChunk;

    let started = performance.now();
    const model = await BeatThisWgsl.create(device, weights, manifest);
    log(`create: ${(performance.now() - started).toFixed(0)} ms`);

    const T = BEAT_THIS_WGSL_FRAMES;
    const logits = new Float32Array(2 * count * T);
    for (let i = 0; i < count; i += 1) {
        started = performance.now();
        const result = await model.run(chunks.subarray(i * perChunk, (i + 1) * perChunk), { stages: config.stages && i === 0 });
        log(`chunk ${i}: ${(performance.now() - started).toFixed(1)} ms (gpu ${result.gpuMs.toFixed(1)} ms)`);
        logits.set(result.beat, i * T);
        logits.set(result.downbeat, (count + i) * T);
        for (const [name, data] of Object.entries(result.stages ?? {})) {
            fs.writeFileSync(path.join(config.out, `stage_${name}.f32`), Buffer.from(data.buffer));
        }
    }
    fs.writeFileSync(path.join(config.out, 'logits.f32'), Buffer.from(logits.buffer));

    const times: number[] = [];
    for (let r = 0; r < config.repeat; r += 1) {
        for (let i = 0; i < count; i += 1) {
            times.push((await model.run(chunks.subarray(i * perChunk, (i + 1) * perChunk))).gpuMs);
        }
    }
    if (times.length) {
        times.sort((a, b) => a - b);
        log(`timed ${times.length} runs: median ${times[times.length >> 1].toFixed(1)} ms, min ${times[0].toFixed(1)} ms`);
    }
    model.destroy();
    ipcRenderer.send('btw-done', 0);
}

/** Mid (L+R)/2 at the Beat This! rate; decodeAudioData does the resampling, as in the app. */
async function decodeMono(file: string): Promise<Float32Array> {
    const bytes = fs.readFileSync(file);
    const buffer = await new OfflineAudioContext(1, 1, BEAT_THIS_SAMPLE_RATE)
        .decodeAudioData(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
    const left = buffer.getChannelData(0);
    if (buffer.numberOfChannels < 2) return Float32Array.from(left);
    const right = buffer.getChannelData(1);
    return left.map((value, index) => (value + right[index]) / 2);
}

async function runSongs(device: GPUDevice, weights: Float32Array, manifest: BeatThisWgslManifest, songs: string[], out: string) {
    const model = await BeatThisWgsl.create(device, weights, manifest);
    const T = BEAT_THIS_WGSL_FRAMES;
    for (const song of songs) {
        const name = path.basename(song).replace(/\.[^.]+$/, '');
        const mel = await melSpectrogram(await decodeMono(song));
        const chunks = splitForModel(mel);
        if (chunks.some(chunk => chunk.frames !== T)) throw new Error(`${name}: chunk shorter than ${T} frames`);
        const input = new Float32Array(chunks.length * T * 128);
        const logits = new Float32Array(chunks.length * 2 * T);
        let gpuMs = 0;
        for (const [index, chunk] of chunks.entries()) {
            input.set(chunk.data, index * T * 128);
            const result = await model.run(chunk.data);
            logits.set(result.beat, index * 2 * T);
            logits.set(result.downbeat, (index * 2 + 1) * T);
            gpuMs += result.gpuMs;
        }
        fs.writeFileSync(path.join(out, `${name}.chunks.f32`), Buffer.from(input.buffer));
        fs.writeFileSync(path.join(out, `${name}.wgsl.f32`), Buffer.from(logits.buffer));
        fs.writeFileSync(path.join(out, `${name}.meta.json`), JSON.stringify({ frames: mel.frames, starts: chunks.map(chunk => chunk.start) }));
        log(`${name}: ${mel.frames} frames, ${chunks.length} chunks, gpu ${gpuMs.toFixed(0)} ms total`);
    }
    model.destroy();
}

main().catch((error) => {
    log(`FAILED: ${error?.stack ?? error}`);
    ipcRenderer.send('btw-done', 1);
});
