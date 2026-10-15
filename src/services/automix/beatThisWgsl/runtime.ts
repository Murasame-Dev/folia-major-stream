/// <reference types="@webgpu/types" />
import { ATTENTION, CONV_GELU, FLATTEN, MATMUL, RMS_NORM, ROPE, STEM_BN, SUM_HEAD } from './kernels';

// src/services/automix/beatThisWgsl/runtime.ts
// Runs one Beat This! chunk (mel [1500, 128] -> beat / downbeat logits [1500]) on a WebGPU device
// with the kernels in ./kernels. The graph is fixed, so every pipeline, buffer, uniform and bind
// group is built once in `create`; a run only uploads the mel, records the dispatches and reads the
// two logit tracks back. Weights come from build/beat_this_wgsl/export_weights.py.

export const BEAT_THIS_WGSL_FRAMES = 1500;
const MEL_BINS = 128;
const HEAD_DIM = 32;

export interface BeatThisWgslManifest {
    hyper_parameters: { transformer_dim: number; n_layers: number; stem_dim: number; ff_mult: number };
    tensors: Record<string, { offset: number; shape: number[] }>;
    floats: number;
}

export interface BeatThisWgslResult {
    beat: Float32Array;
    downbeat: Float32Array;
    /** Wall time from submit to the GPU finishing, ms. */
    gpuMs: number;
    /** Intermediate activations by stage name, only when asked for. Frontend stages are [F, T, C]. */
    stages?: Record<string, Float32Array>;
}

interface Binding { buffer: GPUBuffer; offset?: number; size?: number }
type Step =
    | { kind: 'dispatch'; pipeline: GPUComputePipeline; group: GPUBindGroup; count: [number, number, number] }
    | { kind: 'stage'; name: string; buffer: GPUBuffer; floats: number };

const product = (shape: number[]) => shape.reduce((a, b) => a * b, 1);

export class BeatThisWgsl {
    private constructor(
        private readonly device: GPUDevice,
        private readonly steps: Step[],
        private readonly melBuffer: GPUBuffer,
        private readonly outBuffer: GPUBuffer,
        private readonly owned: GPUBuffer[],
    ) {}

    /** Uploads the weights and builds the whole fixed graph for one 1500-frame chunk. */
    static async create(device: GPUDevice, weights: Float32Array, manifest: BeatThisWgslManifest): Promise<BeatThisWgsl> {
        const T = BEAT_THIS_WGSL_FRAMES;
        const owned: GPUBuffer[] = [];
        const storage = (floats: number, extra = 0) => {
            const buffer = device.createBuffer({
                size: Math.max(16, floats * 4),
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | extra,
            });
            owned.push(buffer);
            return buffer;
        };

        const weightBuffer = storage(weights.length);
        device.queue.writeBuffer(weightBuffer, 0, weights.buffer, weights.byteOffset, weights.byteLength);
        const tensor = (name: string): Binding => {
            const entry = manifest.tensors[name];
            if (!entry) throw new Error(`beat_this weights: missing tensor ${name}`);
            return { buffer: weightBuffer, offset: entry.offset * 4, size: product(entry.shape) * 4 };
        };
        const shape = (name: string) => manifest.tensors[name].shape;

        // Activations, sized for the largest stage that uses each (block 0 for most, see kernels.ts).
        const dim = manifest.hyper_parameters.transformer_dim;
        const stem = manifest.hyper_parameters.stem_dim;
        const ffMult = manifest.hyper_parameters.ff_mult;
        const tokens0 = (MEL_BINS / 4) * T; // block 0: 32 freqs x T frames
        const maxWidth = (c: number) => 3 * c + c / HEAD_DIM;
        const melBuffer = storage(T * MEL_BINS);
        const stemIn = storage(T * MEL_BINS);
        const x = [storage(tokens0 * stem), storage(tokens0 * stem)];
        const h = storage(tokens0 * stem);
        const qkv = storage(Math.max(tokens0 * maxWidth(stem), T * maxWidth(dim)));
        const o = storage(tokens0 * stem);
        const u = storage(Math.max(tokens0 * stem * ffMult, T * dim * ffMult));
        const flat = storage(T * 4 * stem * 8);
        const logits = storage(T * 2);
        const outBuffer = storage(T * 2);
        const zeros = storage(4096);

        // RoPE cos/sin per (position, pair), with the f32 angle torch uses (position * freq, rounded).
        const freqs = weights.subarray(manifest.tensors['rope.freqs'].offset, manifest.tensors['rope.freqs'].offset + HEAD_DIM / 2);
        const table = new Float32Array(T * 16 * 2);
        for (let pos = 0; pos < T; pos += 1) {
            for (let i = 0; i < 16; i += 1) {
                const angle = Math.fround(Math.fround(pos) * freqs[i]);
                table[(pos * 16 + i) * 2] = Math.cos(angle);
                table[(pos * 16 + i) * 2 + 1] = Math.sin(angle);
            }
        }
        const ropeTable = storage(table.length);
        device.queue.writeBuffer(ropeTable, 0, table);

        const pipelines = new Map<string, GPUComputePipeline>();
        const pipeline = (code: string) => {
            let found = pipelines.get(code);
            if (!found) {
                found = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' } });
                pipelines.set(code, found);
            }
            return found;
        };
        const uniform = (values: number[]) => {
            const data = new Uint32Array(Math.ceil(values.length / 4) * 4);
            data.set(values);
            const buffer = device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
            device.queue.writeBuffer(buffer, 0, data);
            owned.push(buffer);
            return buffer;
        };

        const steps: Step[] = [];
        const dispatch = (code: string, bindings: Binding[], params: number[], count: [number, number, number]) => {
            const compute = pipeline(code);
            const entries = [...bindings, { buffer: uniform(params) }].map((resource, binding) => ({ binding, resource }));
            steps.push({ kind: 'dispatch', pipeline: compute, group: device.createBindGroup({ layout: compute.getBindGroupLayout(0), entries }), count });
        };
        const groups = (invocations: number): [number, number, number] => [Math.ceil(invocations / 64), 1, 1];
        const stage = (name: string, buffer: GPUBuffer, floats: number) => steps.push({ kind: 'stage', name, buffer, floats });

        const rmsNorm = (input: GPUBuffer, gamma: string, rows: number, width: number) =>
            dispatch(RMS_NORM, [{ buffer: input }, tensor(gamma), { buffer: h }], [rows, width], [rows, 1, 1]);
        /** mode: 0 store, 1 GELU, 2 add into the output (residual). */
        const matmul = (input: GPUBuffer, weight: string, bias: string | null, output: GPUBuffer, m: number, mode: number) => {
            const [k, n] = shape(weight);
            dispatch(MATMUL, [{ buffer: input }, tensor(weight), bias ? tensor(bias) : { buffer: zeros }, { buffer: output }],
                [m, n, k, mode], [Math.ceil(n / 64), Math.ceil(m / 64), 1]);
        };

        /**
         * One pre-norm attention + feed-forward pair on the residual stream `stream` ([rows, c]).
         * `n`/`seqStride`/`batchStride`/`batches` say which axis is attended; RoPE positions follow it.
         */
        const transformerPair = (stream: GPUBuffer, prefix: string, rows: number, c: number,
            axis: { n: number; seqStride: number; batchStride: number; batches: number; posDiv: number }) => {
            const heads = c / HEAD_DIM;
            const width = 3 * c + heads;
            rmsNorm(stream, `${prefix}.attn.norm`, rows, c);
            matmul(h, `${prefix}.attn.qkvg.w`, `${prefix}.attn.qkvg.b`, qkv, rows, 0);
            dispatch(ROPE, [{ buffer: qkv }, { buffer: ropeTable }], [rows, heads, width, c, axis.posDiv, axis.n], groups(rows * heads * 16));
            dispatch(ATTENTION, [{ buffer: qkv }, { buffer: o }], [axis.n, heads, width, c, axis.seqStride, axis.batchStride],
                [Math.ceil(axis.n / 64), heads, axis.batches]);
            matmul(o, `${prefix}.attn.out.w`, null, stream, rows, 2);
            rmsNorm(stream, `${prefix}.ff.norm`, rows, c);
            matmul(h, `${prefix}.ff.w1`, `${prefix}.ff.b1`, u, rows, 1);
            matmul(u, `${prefix}.ff.w2`, `${prefix}.ff.b2`, stream, rows, 2);
        };

        // Stem: BatchNorm1d over mel bins, then conv (4, 3) / stride 4 on frequency.
        dispatch(STEM_BN, [{ buffer: melBuffer }, tensor('stem.bn.scale'), tensor('stem.bn.shift'), { buffer: stemIn }], [T, MEL_BINS], groups(T * MEL_BINS));
        let freqs_ = MEL_BINS / 4;
        let channels = stem;
        dispatch(CONV_GELU, [{ buffer: stemIn }, tensor('stem.conv.w'), tensor('stem.conv.b'), { buffer: x[0] }],
            [freqs_, T, 1, channels, 4], groups(freqs_ * T * channels));
        let cur = 0;
        stage('stem', x[cur], freqs_ * T * channels);

        for (let block = 0; block < 3; block += 1) {
            const rows = freqs_ * T;
            transformerPair(x[cur], `block${block}.freq`, rows, channels,
                { n: freqs_, seqStride: T, batchStride: 1, batches: T, posDiv: T });
            stage(`block${block}_freq`, x[cur], rows * channels);
            transformerPair(x[cur], `block${block}.time`, rows, channels,
                { n: T, seqStride: 1, batchStride: T, batches: freqs_, posDiv: 1 });
            stage(`block${block}_time`, x[cur], rows * channels);
            const next = 1 - cur;
            dispatch(CONV_GELU, [{ buffer: x[cur] }, tensor(`block${block}.conv.w`), tensor(`block${block}.conv.b`), { buffer: x[next] }],
                [freqs_ / 2, T, channels, channels * 2, 2], groups((freqs_ / 2) * T * channels * 2));
            cur = next;
            freqs_ /= 2;
            channels *= 2;
            stage(`block${block}_conv`, x[cur], freqs_ * T * channels);
        }

        dispatch(FLATTEN, [{ buffer: x[cur] }, { buffer: flat }], [freqs_, T, channels], groups(freqs_ * T * channels));
        cur = 1 - cur;
        matmul(flat, 'frontend.linear.w', 'frontend.linear.b', x[cur], T, 0);
        stage('frontend', x[cur], T * dim);

        for (let layer = 0; layer < manifest.hyper_parameters.n_layers; layer += 1) {
            transformerPair(x[cur], `layer${layer}`, T, dim, { n: T, seqStride: 1, batchStride: T, batches: 1, posDiv: 1 });
            stage(`layer${layer}`, x[cur], T * dim);
        }
        rmsNorm(x[cur], 'final.norm', T, dim);
        matmul(h, 'head.w', 'head.b', logits, T, 0);
        dispatch(SUM_HEAD, [{ buffer: logits }, { buffer: outBuffer }], [T], groups(T));
        stage('logits', outBuffer, T * 2);

        await device.queue.onSubmittedWorkDone();
        return new BeatThisWgsl(device, steps, melBuffer, outBuffer, owned);
    }

    /** One chunk. `mel` is [1500, 128] frame-major, the layout beatThis.ts produces. */
    async run(mel: Float32Array, options: { stages?: boolean } = {}): Promise<BeatThisWgslResult> {
        const T = BEAT_THIS_WGSL_FRAMES;
        if (mel.length !== T * MEL_BINS) throw new Error(`beat_this chunk must be ${T}x${MEL_BINS}`);
        const { device } = this;
        device.queue.writeBuffer(this.melBuffer, 0, mel.buffer, mel.byteOffset, mel.byteLength);

        const encoder = device.createCommandEncoder();
        const readbacks: { name: string; buffer: GPUBuffer }[] = [];
        let pass: GPUComputePassEncoder | null = null;
        for (const step of this.steps) {
            if (step.kind === 'dispatch') {
                pass ??= encoder.beginComputePass();
                pass.setPipeline(step.pipeline);
                pass.setBindGroup(0, step.group);
                pass.dispatchWorkgroups(...step.count);
                continue;
            }
            if (!options.stages && step.name !== 'logits') continue;
            pass?.end();
            pass = null;
            const readback = device.createBuffer({ size: step.floats * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
            encoder.copyBufferToBuffer(step.buffer, 0, readback, 0, step.floats * 4);
            readbacks.push({ name: step.name, buffer: readback });
        }
        pass?.end();

        const started = performance.now();
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        const gpuMs = performance.now() - started;

        const stages: Record<string, Float32Array> = {};
        for (const { name, buffer } of readbacks) {
            await buffer.mapAsync(GPUMapMode.READ);
            stages[name] = new Float32Array(buffer.getMappedRange().slice(0));
            buffer.destroy();
        }
        const out = stages.logits;
        return {
            beat: out.subarray(0, T),
            downbeat: out.subarray(T, 2 * T),
            gpuMs,
            ...(options.stages ? { stages } : {}),
        };
    }

    destroy(): void {
        this.owned.forEach(buffer => buffer.destroy());
    }
}
