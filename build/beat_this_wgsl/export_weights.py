"""Exports a Beat This! checkpoint into the flat f32 layout the WGSL runtime reads.

build/beat_this_wgsl/export_weights.py

    python export_weights.py CKPT OUT.bin

Writes OUT.bin (little-endian f32, every tensor 256-byte aligned) and OUT.json (name -> offset in
floats, shape). Everything that is a constant transform of the weights happens here, once, so the
kernels never redo it per chunk:

- BatchNorms folded: bn1d into a per-frequency scale/shift, every bn2d into its convolution.
- RMSNorm gammas pre-multiplied by sqrt(dim).
- Linear weights transposed to [in, out], the layout the matmul kernel reads row by row.
- Each attention's gate projection appended to its QKV projection as H extra output columns, so one
  matmul produces q, k, v and the gate logits together (bias is zero for the QKV columns).
- The frontend's flatten is "b c f t -> b t (c f)"; the runtime flattens as (f c) instead, so the
  rows of frontend.linear are permuted to match.
"""
import json
import math
import sys

import numpy as np
import torch

HEAD_DIM = 32
ALIGN = 64  # floats, i.e. 256 bytes - WebGPU's minStorageBufferOffsetAlignment


def bn_affine(sd, prefix, eps=1e-5):
    scale = sd[f'{prefix}.weight'] / torch.sqrt(sd[f'{prefix}.running_var'] + eps)
    shift = sd[f'{prefix}.bias'] - sd[f'{prefix}.running_mean'] * scale
    return scale, shift


def main():
    ckpt_path, out_path = sys.argv[1:3]
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    sd = {k.removeprefix('model.'): v.double() for k, v in ckpt['state_dict'].items()}
    hp = ckpt['hyper_parameters']
    tensors = {}

    def put(name, value):
        tensors[name] = np.ascontiguousarray(value.numpy() if torch.is_tensor(value) else value, dtype=np.float32)

    def put_attention(prefix, out):
        dim = sd[f'{prefix}.norm.gamma'].shape[0]
        heads = dim // HEAD_DIM
        put(f'{out}.norm', sd[f'{prefix}.norm.gamma'] * math.sqrt(dim))
        weight = torch.cat([sd[f'{prefix}.to_qkv.weight'], sd[f'{prefix}.to_gates.weight']], 0)  # [3D+H, D]
        put(f'{out}.qkvg.w', weight.T)
        put(f'{out}.qkvg.b', torch.cat([torch.zeros(3 * dim, dtype=torch.float64), sd[f'{prefix}.to_gates.bias']]))
        put(f'{out}.out.w', sd[f'{prefix}.to_out.0.weight'].T)
        assert heads == sd[f'{prefix}.to_gates.weight'].shape[0]

    def put_ff(prefix, out):
        dim = sd[f'{prefix}.net.0.gamma'].shape[0]
        put(f'{out}.norm', sd[f'{prefix}.net.0.gamma'] * math.sqrt(dim))
        put(f'{out}.w1', sd[f'{prefix}.net.1.weight'].T)
        put(f'{out}.b1', sd[f'{prefix}.net.1.bias'])
        put(f'{out}.w2', sd[f'{prefix}.net.4.weight'].T)
        put(f'{out}.b2', sd[f'{prefix}.net.4.bias'])

    def put_conv(conv, bn, out):
        scale, shift = bn_affine(sd, bn)
        put(f'{out}.w', sd[f'{conv}.weight'] * scale[:, None, None, None])  # [cout, cin, kf, 3]
        put(f'{out}.b', shift)

    scale, shift = bn_affine(sd, 'frontend.stem.bn1d')
    put('stem.bn.scale', scale)
    put('stem.bn.shift', shift)
    put_conv('frontend.stem.conv2d', 'frontend.stem.bn2d', 'stem.conv')
    for i in range(3):
        p = f'frontend.blocks.{i}.partial'
        put_attention(f'{p}.attnF', f'block{i}.freq.attn')
        put_ff(f'{p}.ffF', f'block{i}.freq.ff')
        put_attention(f'{p}.attnT', f'block{i}.time.attn')
        put_ff(f'{p}.ffT', f'block{i}.time.ff')
        put_conv(f'frontend.blocks.{i}.conv2d', f'frontend.blocks.{i}.norm', f'block{i}.conv')

    # [512, (c f)] -> rows reordered to (f c), then transposed to [(f c), 512].
    linear = sd['frontend.linear.weight']
    channels = sd['frontend.blocks.2.norm.weight'].shape[0]
    freqs = linear.shape[1] // channels
    linear = linear.view(-1, channels, freqs).permute(0, 2, 1).reshape(linear.shape[0], -1)
    put('frontend.linear.w', linear.T)
    put('frontend.linear.b', sd['frontend.linear.bias'])
    for i in range(hp['n_layers']):
        put_attention(f'transformer_blocks.layers.{i}.0', f'layer{i}.attn')
        put_ff(f'transformer_blocks.layers.{i}.1', f'layer{i}.ff')
    put('final.norm', sd['transformer_blocks.norm.gamma'] * math.sqrt(sd['transformer_blocks.norm.gamma'].shape[0]))
    put('head.w', sd['task_heads.beat_downbeat_lin.weight'].T)
    put('head.b', sd['task_heads.beat_downbeat_lin.bias'])

    # One rotary table is shared by every attention in the model; check rather than assume.
    rope = [v for k, v in sd.items() if k.endswith('rotary_embed.freqs')]
    assert all(torch.equal(r, rope[0]) for r in rope), 'rotary freqs differ between layers'
    put('rope.freqs', rope[0])

    manifest = {'hyper_parameters': {k: hp[k] for k in ('spect_dim', 'transformer_dim', 'ff_mult', 'n_layers', 'stem_dim', 'head_dim')},
                'tensors': {}}
    offset = 0
    with open(out_path, 'wb') as out:
        for name, array in tensors.items():
            manifest['tensors'][name] = {'offset': offset, 'shape': list(array.shape)}
            out.write(array.tobytes())
            offset += array.size
            pad = -offset % ALIGN
            out.write(np.zeros(pad, np.float32).tobytes())
            offset += pad
    manifest['floats'] = offset
    with open(out_path.removesuffix('.bin') + '.json', 'w') as f:
        json.dump(manifest, f, indent=1)
    print(f'{len(tensors)} tensors, {offset * 4 / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
