# services/analysis

曲目分析的公共底层。automix（`../automix/`）和节拍效果（`../beatFx/`，商籁的智能动感模式）都从这里读，彼此不互相依赖。

| 文件 | 职责 |
|---|---|
| `trackAudio.ts` | 读一首歌**已经在本机**的字节：播放 representation → 媒体缓存 → 本地 URL → 本地文件句柄（App 注册的读取器）。从不下载；要不要下载由调用方决定（只有 automix 的 profile 会）。另含按分析采样率解码、拆 mid/side。 |
| `beatThis.ts` | Beat This! 的前处理契约：梅尔频谱、1500 帧分块、峰值挑选、网格折算。推理先走 GPU，`cpuFallback` 决定能否退到 CPU。 |
| `beatThisGpu.ts` + `beatThisWgsl/` | 渲染进程 Worker 里的 WebGPU 推理；`probeBeatThisGpu` 给设置页判断有没有可用 GPU。 |
| `beatGrid.ts` | 整首歌的拍点：先读存盘结果，正在跑就等同一次，否则跑一次并存盘（`beat_this_grid_<分析 key>`，归入「分析」缓存类别）。只存整首；只读了开头的 automix 分析自己跑、不存。 |
| `modelAvailability.ts` | 两个模型（beat_this / htdemucs）的权重是否在盘上，可同步读取。 |

约定：

- 存结果的调用方用 `getPlaybackAnalysisTarget` 一次取出 key 和 representation，读字节用同一个 representation，避免转码中途注册导致结果存错 key。
- 节拍效果只走 GPU（不传 `cpuFallback`）；automix 可以退到 CPU。两者同时要同一首歌时，`beatGrid.ts` 合并成一次推理；只有 GPU 跑空、而调用方允许 CPU 时才会再跑一次。
