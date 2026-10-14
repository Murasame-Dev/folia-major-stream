import type { AudioBands } from '../../../types';
import { beatImpact, downbeatImpact } from '../../../stores/motionSignals';
import { isBeatFxEnabled } from '../../../services/beatFx/beatFxDevFlag';
import { getBeatFxRotationSign } from '../../../services/beatFx/beatFxLive';
import type { SonnetAnimationCue } from './types';

// src/components/visualizer/sonnet/sonnetBeatFx.ts
// Experimental Sonnet beat FX (dev flag `folia.dev.beatFx` only). Pure envelope math plus a
// per-runtime frame reader; each effect follows its own signal:
//   camera punch  <- beatImpact      (zoom + alternating tilt, layered on the shot camera)
//   mono tear     <- downbeatImpact  (shares the transition glitch filter, combined with max)
//   glow flare    <- downbeatImpact  (halo layer alpha)
//   icon pop      <- lyric segment onsets (replaces the band-driven icon breathing)
//   line flash    <- treble band, smoothed, continuous
// Constants mirror the approved mockup (folia-header-mock/sonnet-beat/index.html).

export const SONNET_BEAT_FX = {
    /** Mockup used 3% at 118 BPM; at ~160 BPM that read as drift, so beats punch harder and shorter. */
    zoomPercent: 6,
    rotationDeg: 0.6,
    downbeatZoomPercent: 5,
    tear: 0.6,
    tearRate: 24,
    flare: 0.7,
    iconPop: 0.4,
    iconDecayMs: 300,
    /** Line layers sit at this alpha and the treble band lifts them to 1. */
    lineBaseAlpha: 0.78,
} as const;

export interface SonnetBeatFxFrame {
    /** Camera scale multiplier. */
    zoom: number;
    /** Extra camera rotation in radians. */
    rotation: number;
    /** Glitch filter amount from the downbeat envelope (0 when idle). */
    tear: number;
    tearSeed: number;
    /** 0..~0.7 glow flare added to the halo layer. */
    flare: number;
    /** 0..1 smoothed treble level for the line flash. */
    lineLift: number;
}

const normalizeLevel = (value: number) => Math.min(1, Math.max(0, value > 1 ? value / 255 : value));

/** Envelope math for one frame, from the two impacts and the alternating tilt sign. */
export const resolveSonnetBeatFxFrame = (
    k: number,
    d: number,
    rotationSign: number,
    wallTimeSec: number,
    lineLift: number,
): SonnetBeatFxFrame => ({
    zoom: 1 + (SONNET_BEAT_FX.zoomPercent / 100) * k + (SONNET_BEAT_FX.downbeatZoomPercent / 100) * d,
    rotation: (SONNET_BEAT_FX.rotationDeg * Math.PI / 180) * k * rotationSign,
    tear: d > 0.02 ? SONNET_BEAT_FX.tear * d : 0,
    tearSeed: Math.floor(wallTimeSec * SONNET_BEAT_FX.tearRate) % 997,
    flare: SONNET_BEAT_FX.flare * d,
    lineLift,
});

/**
 * Creates the per-runtime frame reader. Returns null every frame while the flag is off, so callers
 * keep their original code path untouched. Holds only the treble smoothing state.
 */
export const createSonnetBeatFxReader = () => {
    let smoothedTreble = 0;
    return (audioBands: AudioBands | undefined): SonnetBeatFxFrame | null => {
        if (!isBeatFxEnabled()) return null;
        const treble = normalizeLevel(audioBands?.treble?.get() ?? 0);
        // Ignore the idle breathing floor, then ease: quick attack, slow release, no discrete trigger.
        const target = Math.min(1, Math.max(0, (treble - 0.04) / 0.35));
        smoothedTreble += (target - smoothedTreble) * (target > smoothedTreble ? 0.25 : 0.08);
        return resolveSonnetBeatFxFrame(
            beatImpact.get(),
            downbeatImpact.get(),
            getBeatFxRotationSign(),
            performance.now() / 1000,
            smoothedTreble,
        );
    };
};

/** Alpha of the MG line layers for a given treble lift. */
export const resolveSonnetLineFlashAlpha = (lineLift: number) =>
    SONNET_BEAT_FX.lineBaseAlpha + (1 - SONNET_BEAT_FX.lineBaseAlpha) * Math.min(1, Math.max(0, lineLift));

/**
 * Icon pop driven by the lyric segment onsets of the shot: a 0..1 envelope that restarts on every
 * cue and decays over ~300 ms. Stateless (latest cue at or before `time`), so seeking is safe.
 */
export const resolveSonnetIconOnsetPop = (time: number, cues: readonly SonnetAnimationCue[]) => {
    let lastAt = Number.NEGATIVE_INFINITY;
    for (let index = 0; index < cues.length; index += 1) {
        const at = cues[index].at;
        if (at <= time && at > lastAt) lastAt = at;
    }
    const elapsedMs = (time - lastAt) * 1000;
    const k = elapsedMs >= 0 && Number.isFinite(elapsedMs)
        ? Math.exp(-elapsedMs / SONNET_BEAT_FX.iconDecayMs * 2.3)
        : 0;
    return { k, scale: 1 + SONNET_BEAT_FX.iconPop * k * (1.15 - 0.15 * k) };
};

const haloBaseAlpha = new WeakMap<object, number>();

/** Lifts a shot's glow layer towards opaque on a downbeat; restores the built alpha otherwise. */
export const applySonnetHaloFlare = (haloLayer: { alpha: number }, flare: number) => {
    let base = haloBaseAlpha.get(haloLayer);
    if (base === undefined) {
        base = haloLayer.alpha;
        haloBaseAlpha.set(haloLayer, base);
    }
    haloLayer.alpha = Math.min(1, base + (1 - base) * Math.min(1, flare));
};
