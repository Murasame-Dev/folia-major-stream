import type { SongResult } from '../../types';
import { mergeBeatFxEvents, type BeatFxEvent } from '../../utils/beatFxClock';
import { analyseBeatGrid, canRunBeatThis } from '../automix/beatThis';
import { readStoredBeatGrid, storeBeatGrid } from '../automix/beatGridCache';
import { melSpectrogramOffThread } from '../automix/analysisOffThread';
import { refreshModelAvailability } from '../automix/modelAvailability';
import { PROFILE_SAMPLE_RATE } from '../automix/trackProfile';
import { getCachedSongAudioBlob } from '../onlineMusic/resourceCache';
import { getPlaybackAnalysisKey, getPlaybackRepresentation } from '../playbackRecovery/representationRegistry';

// src/services/beatFx/beatFxSource.ts
// Where Sonnet's 智能动感 gets the playing track's beat list from.
//
// First the beat list stored for this track (automix/beatGridCache.ts) - written by an earlier play,
// or by automix's profile, which runs the same model. Otherwise the in-app Beat This! pass on the
// track's own bytes, with the pieces automix uses: representation or media-cache bytes, decode at the
// analysis rate, mel off-thread, `analyseBeatGrid`. Never downloads anything: a track that is not
// cached is not analysed.
//
// The pass runs on the renderer's GPU only. `analyseBeatGrid` is called without `cpuFallback`, so a
// machine whose GPU cannot take it gets no beats - and Sonnet its original picture - rather than a
// CPU inference competing with playback for the length of a track.

export type BeatFxSourceKind = 'beat_this' | 'stored' | 'none';

export interface BeatFxTrackBeats {
    events: BeatFxEvent[];
    source: BeatFxSourceKind;
    /** Why the preferred source was not used, when it was not. */
    note?: string;
}

const memory = new Map<string, Promise<BeatFxTrackBeats>>();
const MAX_TRACKS = 24;

/** Reads the playing track's bytes from local storage only (representation, then media cache). */
const readLocalBytes = async (song: SongResult): Promise<ArrayBuffer | null> => {
    const representation = getPlaybackRepresentation(song);
    if (representation) {
        try {
            const response = await fetch(representation.url);
            if (response.ok) return await response.arrayBuffer();
        } catch {
            // Fall through to the media cache.
        }
    }
    const cached = await getCachedSongAudioBlob(song);
    return cached ? cached.arrayBuffer() : null;
};

/** Mono (mid) samples at the Beat This! rate; decodeAudioData does the resampling. */
const decodeMono = async (bytes: ArrayBuffer): Promise<Float32Array | null> => {
    const OfflineContext = window.OfflineAudioContext;
    if (!OfflineContext) return null;
    const buffer = await new OfflineContext(1, 1, PROFILE_SAMPLE_RATE).decodeAudioData(bytes);
    const left = buffer.getChannelData(0);
    if (buffer.numberOfChannels < 2) return Float32Array.from(left);
    const right = buffer.getChannelData(1);
    const mid = new Float32Array(buffer.length);
    for (let index = 0; index < mid.length; index += 1) mid[index] = (left[index] + right[index]) / 2;
    return mid;
};

/** Runs the model on the cached bytes; null with a reason when that path is not available. */
const analyseInApp = async (song: SongResult): Promise<BeatFxTrackBeats | string> => {
    await refreshModelAvailability();
    if (!canRunBeatThis()) return 'beat_this model not available in this build';
    // Asked before the decode and the spectrogram, which are seconds of work for nobody without one.
    if (typeof navigator === 'undefined' || !('gpu' in navigator)) return 'no WebGPU in this renderer';
    const bytes = await readLocalBytes(song);
    if (!bytes) return 'track bytes not in the local media cache';
    const mono = await decodeMono(bytes);
    if (!mono) return 'decode failed';
    // GPU only - see the header.
    const grid = await analyseBeatGrid(await melSpectrogramOffThread(mono));
    if (!grid || grid.beats.length === 0) return 'beat_this returned no beats (no usable GPU, or a failed run)';
    await storeBeatGrid(getPlaybackAnalysisKey(song), grid);
    return { events: mergeBeatFxEvents(grid.beats, grid.downbeats), source: 'beat_this' };
};

const resolveTrackBeats = async (song: SongResult): Promise<BeatFxTrackBeats> => {
    const stored = await readStoredBeatGrid(getPlaybackAnalysisKey(song));
    if (stored) return { events: mergeBeatFxEvents(stored.beats, stored.downbeats), source: 'stored' };
    try {
        const inApp = await analyseInApp(song);
        if (typeof inApp !== 'string') return inApp;
        return { events: [], source: 'none', note: inApp };
    } catch (error) {
        return { events: [], source: 'none', note: `beat_this failed: ${String(error)}` };
    }
};

/** Beat list for a track, computed once per session and shared by concurrent callers. */
export const loadBeatFxTrackBeats = (song: SongResult): Promise<BeatFxTrackBeats> => {
    const key = getPlaybackAnalysisKey(song);
    const existing = memory.get(key);
    if (existing) return existing;
    const pending = resolveTrackBeats(song);
    memory.set(key, pending);
    while (memory.size > MAX_TRACKS) {
        const oldest = memory.keys().next().value;
        if (oldest === undefined) break;
        memory.delete(oldest);
    }
    // A failed lookup is not remembered, so a later play (e.g. once the track is cached) retries.
    void pending.then((result) => {
        if (result.source === 'none' && memory.get(key) === pending) memory.delete(key);
    });
    return pending;
};
