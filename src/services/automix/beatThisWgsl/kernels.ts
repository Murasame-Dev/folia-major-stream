// src/services/automix/beatThisWgsl/kernels.ts
// WGSL sources for the hand-written Beat This! forward pass. Plain fp32, one kernel per step, no
// fusion beyond what is free (bias and activation in the matmul epilogue, the gate in the attention
// epilogue). This is the correctness baseline: the fp16 variants (f16 storage and multiplies, f32
// for the RMSNorm sum of squares, the softmax sum and long matmul accumulations) and the fused
// kernels are both measured against it.
//
// Activation layout everywhere in the frontend is [F, T, C] row-major (channel-last): a token is one
// (frequency, frame) pair, so every Linear is a plain [tokens, C] matmul, and the frequency- and
// time-directed attentions read the same buffer with different strides instead of transposing it.

/** Exact GELU through erf, Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7) - the form ORT's own WGSL uses. */
const GELU = /* wgsl */ `
fn erf_f32(v: f32) -> f32 {
  let a = abs(v);
  let x = 1.0 / (1.0 + 0.3275911 * a);
  let y = 1.0 - ((((1.061405429 * x - 1.453152027) * x + 1.421413741) * x - 0.284496736) * x + 0.254829592) * x * exp(-a * a);
  return select(-y, y, v >= 0.0);
}
fn gelu(v: f32) -> f32 { return 0.5 * v * (1.0 + erf_f32(v * 0.7071067811865476)); }
`;

/** Eval-mode BatchNorm1d over the 128 mel bins, written as the [F=128, T, C=1] input of the stem conv. */
export const STEM_BN = /* wgsl */ `
struct P { frames: u32, bins: u32 }
@group(0) @binding(0) var<storage, read> mel: array<f32>;      // [T, bins]
@group(0) @binding(1) var<storage, read> scale: array<f32>;
@group(0) @binding(2) var<storage, read> shift: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>; // [bins, T]
@group(0) @binding(4) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= p.frames * p.bins) { return; }
  let f = i / p.frames;
  let t = i % p.frames;
  out[i] = mel[t * p.bins + f] * scale[f] + shift[f];
}
`;

/**
 * Conv2d with kernel (kf, 3), stride (kf, 1), padding (0, 1), BatchNorm folded into the weights, then
 * GELU. [Fin, T, Cin] -> [Fin / kf, T, Cout]. One invocation per output element.
 */
export const CONV_GELU = /* wgsl */ `
${GELU}
struct P { fout: u32, frames: u32, cin: u32, cout: u32, kf: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;   // [cout, cin, kf, 3]
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= p.fout * p.frames * p.cout) { return; }
  let co = i % p.cout;
  let t = (i / p.cout) % p.frames;
  let fo = i / (p.cout * p.frames);
  var acc = b[co];
  for (var df = 0u; df < p.kf; df++) {
    let fi = fo * p.kf + df;
    for (var dt = 0u; dt < 3u; dt++) {
      let ti = i32(t) + i32(dt) - 1;
      if (ti < 0 || ti >= i32(p.frames)) { continue; }
      let xBase = (fi * p.frames + u32(ti)) * p.cin;
      for (var ci = 0u; ci < p.cin; ci++) {
        acc += w[((co * p.cin + ci) * p.kf + df) * 3u + dt] * x[xBase + ci];
      }
    }
  }
  y[i] = gelu(acc);
}
`;

/** RMSNorm as F.normalize(x) * sqrt(D) * gamma, the sqrt(D) pre-multiplied into gamma. One workgroup per row. */
export const RMS_NORM = /* wgsl */ `
struct P { rows: u32, dim: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u) {
  let row = wg.x + wg.y * 65535u;
  let base = row * p.dim;
  var s = 0.0;
  if (row < p.rows) {
    for (var c = l.x; c < p.dim; c += 64u) { let v = x[base + c]; s += v * v; }
  }
  red[l.x] = s;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (l.x < stride) { red[l.x] += red[l.x + stride]; }
    workgroupBarrier();
  }
  if (row >= p.rows) { return; }
  let inv = 1.0 / max(sqrt(red[0]), 1e-12);
  for (var c = l.x; c < p.dim; c += 64u) { y[base + c] = x[base + c] * inv * gamma[c]; }
}
`;

/**
 * Y[M, N] = X[M, K] . W[K, N] + bias, with an epilogue picked by `mode`:
 * 0 = store, 1 = GELU then store, 2 = add into Y (the residual stream, updated in place).
 * 64x64 output tile per workgroup, 4x4 per invocation, K in steps of 16 through workgroup memory.
 */
export const MATMUL = /* wgsl */ `
${GELU}
struct P { m: u32, n: u32, k: u32, mode: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
var<workgroup> as_: array<f32, 1024>; // [64 rows][16 k]
var<workgroup> bs_: array<f32, 1024>; // [16 k][64 cols]
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u, @builtin(local_invocation_index) li: u32) {
  let row0 = wg.y * 64u;
  let col0 = wg.x * 64u;
  var acc: array<array<f32, 4>, 4>;
  for (var k0 = 0u; k0 < p.k; k0 += 16u) {
    for (var i = 0u; i < 4u; i++) {
      let idx = li + i * 256u;
      let ar = row0 + idx / 16u;
      let ak = k0 + idx % 16u;
      var av = 0.0;
      if (ar < p.m && ak < p.k) { av = x[ar * p.k + ak]; }
      as_[idx] = av;
      let bk = k0 + idx / 64u;
      let bc = col0 + idx % 64u;
      var bv = 0.0;
      if (bk < p.k && bc < p.n) { bv = w[bk * p.n + bc]; }
      bs_[idx] = bv;
    }
    workgroupBarrier();
    for (var kk = 0u; kk < 16u; kk++) {
      var a: array<f32, 4>;
      var b: array<f32, 4>;
      for (var i = 0u; i < 4u; i++) { a[i] = as_[(l.y * 4u + i) * 16u + kk]; }
      for (var j = 0u; j < 4u; j++) { b[j] = bs_[kk * 64u + l.x * 4u + j]; }
      for (var i = 0u; i < 4u; i++) {
        for (var j = 0u; j < 4u; j++) { acc[i][j] += a[i] * b[j]; }
      }
    }
    workgroupBarrier();
  }
  for (var i = 0u; i < 4u; i++) {
    let r = row0 + l.y * 4u + i;
    if (r >= p.m) { continue; }
    for (var j = 0u; j < 4u; j++) {
      let c = col0 + l.x * 4u + j;
      if (c >= p.n) { continue; }
      var v = acc[i][j] + bias[c];
      let o = r * p.n + c;
      if (p.mode == 1u) { v = gelu(v); }
      if (p.mode == 2u) { v += y[o]; }
      y[o] = v;
    }
  }
}
`;

/**
 * Interleaved rotary embedding applied in place to the q and k columns of a [rows, 3D + H] QKV+gate
 * buffer. The position of a row along the attended axis is (row / posDiv) % posMod; cos/sin come from
 * a table built on the CPU with the same f32 angle torch computes, so large positions round alike.
 */
export const ROPE = /* wgsl */ `
struct P { rows: u32, heads: u32, width: u32, dmodel: u32, posDiv: u32, posMod: u32 }
@group(0) @binding(0) var<storage, read_write> qkv: array<f32>;
@group(0) @binding(1) var<storage, read> table: array<f32>; // [pos][16][cos, sin]
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= p.rows * p.heads * 16u) { return; }
  let pair = i % 16u;
  let h = (i / 16u) % p.heads;
  let row = i / (16u * p.heads);
  let pos = (row / p.posDiv) % p.posMod;
  let c = table[(pos * 16u + pair) * 2u];
  let s = table[(pos * 16u + pair) * 2u + 1u];
  for (var part = 0u; part < 2u; part++) {
    let o = row * p.width + part * p.dmodel + h * 32u + pair * 2u;
    let x0 = qkv[o];
    let x1 = qkv[o + 1u];
    qkv[o] = x0 * c - x1 * s;
    qkv[o + 1u] = x1 * c + x0 * s;
  }
}
`;

/**
 * Gated multi-head attention, head dim 32, online softmax (no score matrix is ever stored, so the
 * 1500x1500 time-axis attention never meets the storage binding size limit). One invocation per
 * query; a workgroup covers 64 queries of one (batch, head) and streams keys/values through
 * workgroup memory 32 at a time. Query row = batch * batchStride + index * seqStride.
 */
export const ATTENTION = /* wgsl */ `
struct P { n: u32, heads: u32, width: u32, dmodel: u32, seqStride: u32, batchStride: u32 }
@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<f32>; // [rows, dmodel]
@group(0) @binding(2) var<uniform> p: P;
var<workgroup> ks: array<f32, 1024>;
var<workgroup> vs: array<f32, 1024>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u) {
  let b = wg.z;
  let h = wg.y;
  let qi = wg.x * 64u + l.x;
  let valid = qi < p.n;
  let qrow = b * p.batchStride + qi * p.seqStride;
  var q: array<f32, 32>;
  var acc: array<f32, 32>;
  if (valid) {
    for (var d = 0u; d < 32u; d++) { q[d] = qkv[qrow * p.width + h * 32u + d] * 0.17677669529663687; }
  }
  var m = -3.0e38;
  var sum = 0.0;
  for (var k0 = 0u; k0 < p.n; k0 += 32u) {
    for (var i = 0u; i < 16u; i++) {
      let idx = l.x + i * 64u;
      let kj = k0 + idx / 32u;
      var kv = 0.0;
      var vv = 0.0;
      if (kj < p.n) {
        let base = (b * p.batchStride + kj * p.seqStride) * p.width + h * 32u + idx % 32u;
        kv = qkv[base + p.dmodel];
        vv = qkv[base + 2u * p.dmodel];
      }
      ks[idx] = kv;
      vs[idx] = vv;
    }
    workgroupBarrier();
    if (valid) {
      let count = min(32u, p.n - k0);
      for (var j = 0u; j < count; j++) {
        var s = 0.0;
        for (var d = 0u; d < 32u; d++) { s += q[d] * ks[j * 32u + d]; }
        if (s > m) {
          let r = exp(m - s);
          sum *= r;
          for (var d = 0u; d < 32u; d++) { acc[d] *= r; }
          m = s;
        }
        let e = exp(s - m);
        sum += e;
        for (var d = 0u; d < 32u; d++) { acc[d] += e * vs[j * 32u + d]; }
      }
    }
    workgroupBarrier();
  }
  if (!valid) { return; }
  let gate = 1.0 / (1.0 + exp(-qkv[qrow * p.width + 3u * p.dmodel + h]));
  let scale = gate / sum;
  for (var d = 0u; d < 32u; d++) { o[qrow * p.dmodel + h * 32u + d] = acc[d] * scale; }
}
`;

/** [F, T, C] -> [T, F * C]: the frontend's final flatten, in the (f c) order the exported linear expects. */
export const FLATTEN = /* wgsl */ `
struct P { freqs: u32, frames: u32, channels: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= p.freqs * p.frames * p.channels) { return; }
  let c = i % p.channels;
  let t = (i / p.channels) % p.frames;
  let f = i / (p.channels * p.frames);
  y[(t * p.freqs + f) * p.channels + c] = x[i];
}
`;

/** SumHead: [T, 2] logits -> beat = l0 + l1, downbeat = l1, written as [2, T]. */
export const SUM_HEAD = /* wgsl */ `
struct P { frames: u32 }
@group(0) @binding(0) var<storage, read> l: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let t = id.x;
  if (t >= p.frames) { return; }
  out[t] = l[t * 2u] + l[t * 2u + 1u];
  out[p.frames + t] = l[t * 2u + 1u];
}
`;
