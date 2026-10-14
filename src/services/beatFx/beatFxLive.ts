import { beatImpact, downbeatImpact } from '../../stores/motionSignals';
import { createBeatFxClockState, stepBeatFxClock, type BeatFxEvent } from '../../utils/beatFxClock';
import { getBeatFxOffsetSec, isBeatFxEnabled } from './beatFxDevFlag';
import type { BeatFxSourceKind } from './beatFxSource';

// src/services/beatFx/beatFxLive.ts
// The one live beat clock: fed the current track's beat list, stepped once per frame from the
// existing playback RAF loop, and written into the `beatImpact` / `downbeatImpact` MotionValues.
// No React state anywhere on this path.

interface BeatFxDebug {
    beats: number;
    downbeats: number;
    lastT: number | null;
    heardT: number | null;
    source: BeatFxSourceKind | 'loading';
    note: string | null;
    trackKey: string | null;
    eventCount: number;
    latencyMs: number;
    offsetMs: number;
    beatImpact: number;
    downbeatImpact: number;
}

declare global {
    interface Window {
        __foliaBeatFx?: BeatFxDebug;
    }
}

const state = createBeatFxClockState();
let trackKey: string | null = null;
let lastStepAt: number | null = null;

const debug: BeatFxDebug = {
    beats: 0,
    downbeats: 0,
    lastT: null,
    heardT: null,
    source: 'none',
    note: null,
    trackKey: null,
    eventCount: 0,
    latencyMs: 0,
    offsetMs: getBeatFxOffsetSec() * 1000,
    beatImpact: 0,
    downbeatImpact: 0,
};
if (isBeatFxEnabled() && typeof window !== 'undefined') window.__foliaBeatFx = debug;

/** Switches the clock to a new track; events arrive later via `setBeatFxEvents`. */
export const beginBeatFxTrack = (key: string | null) => {
    trackKey = key;
    state.events = [];
    state.cursor = 0;
    state.lastTime = null;
    debug.trackKey = key;
    debug.source = key ? 'loading' : 'none';
    debug.note = null;
    debug.eventCount = 0;
};

/** Installs the beat list for `key`; ignored if the track changed while it was loading. */
export const setBeatFxEvents = (key: string, events: BeatFxEvent[], source: BeatFxSourceKind, note?: string) => {
    if (key !== trackKey) return;
    state.events = events;
    // Resync on the next step rather than firing everything before the current position.
    state.lastTime = null;
    debug.source = source;
    debug.note = note ?? null;
    debug.eventCount = events.length;
};

/** Alternating ±1 for the camera tilt; flips on every fired beat. */
export const getBeatFxRotationSign = (): number => state.rotationSign;

/**
 * One frame of the beat clock. `audioElement` is the deck being heard, or null when nothing is
 * playing; `context` is the AudioContext it is routed through, used for output latency.
 */
export const tickBeatFx = (audioElement: HTMLMediaElement | null, context: BaseAudioContext | null | undefined) => {
    const now = performance.now();
    const dtSec = lastStepAt === null ? 0 : Math.min(0.1, (now - lastStepAt) / 1000);
    lastStepAt = now;

    let heardTime: number | null = null;
    if (audioElement && !audioElement.paused && !audioElement.ended) {
        const live = context as (BaseAudioContext & { outputLatency?: number; baseLatency?: number }) | null | undefined;
        const latency = (Number(live?.outputLatency) || 0) + (Number(live?.baseLatency) || 0);
        debug.latencyMs = latency * 1000;
        // What the listener hears lags what the element has decoded by the output latency.
        heardTime = audioElement.currentTime - latency - getBeatFxOffsetSec();
    }

    stepBeatFxClock(state, heardTime, dtSec);
    beatImpact.set(state.beatImpact);
    downbeatImpact.set(state.downbeatImpact);

    debug.beats = state.firedBeats;
    debug.downbeats = state.firedDownbeats;
    debug.lastT = state.lastFiredT;
    debug.heardT = heardTime;
    debug.beatImpact = state.beatImpact;
    debug.downbeatImpact = state.downbeatImpact;
};
