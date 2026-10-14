import type { SongResult } from '../../types';
import { getPlaybackSourceRef } from '../../utils/appPlaybackGuards';
import { mergeBeatFxEvents, type BeatFxEvent } from '../../utils/beatFxClock';
import { analyseBeatGrid, canRunBeatThis } from '../automix/beatThis';
import { melSpectrogramOffThread } from '../automix/analysisOffThread';
import { refreshModelAvailability } from '../automix/modelAvailability';
import { PROFILE_SAMPLE_RATE } from '../automix/trackProfile';
import { getCachedSongAudioBlob } from '../onlineMusic/resourceCache';
import { getPlaybackAnalysisKey, getPlaybackRepresentation } from '../playbackRecovery/representationRegistry';

// src/services/beatFx/beatFxSource.ts
// Where the experimental beat FX get their beat list from.
//
// Preferred: the in-app Beat This! pass, run on the playing track's own bytes. The automix profile
// service runs the same model but only keeps the collapsed `{offset, period}` grid (and only when
// automix asks for one), so the raw beats/downbeats are produced here with the same exported
// pieces - representation or media-cache bytes, decode at the analysis rate, mel off-thread,
// `analyseBeatGrid`. Never downloads anything: a track that is not cached is not analysed.
//
// Fallback: a precomputed beat list shipped in `public/dev-beats/<mediaId>.json`, used only when
// the model cannot run or the bytes are not reachable. Kept in memory only; nothing is persisted.

export type BeatFxSourceKind = 'beat_this' | 'dev-json' | 'none';

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
    const bytes = await readLocalBytes(song);
    if (!bytes) return 'track bytes not in the local media cache';
    const mono = await decodeMono(bytes);
    if (!mono) return 'decode failed';
    const grid = await analyseBeatGrid(await melSpectrogramOffThread(mono));
    if (!grid || grid.beats.length === 0) return 'beat_this returned no beats';
    return { events: mergeBeatFxEvents(grid.beats, grid.downbeats), source: 'beat_this' };
};

interface DevJsonEvent { t: number; kind: string }

/** Loads `public/dev-beats/<mediaId>.json` (mockup format `[{t, kind}]`), if one exists. */
const loadDevJson = async (song: SongResult): Promise<BeatFxEvent[] | null> => {
    const ref = getPlaybackSourceRef(song);
    const mediaId = song.qqMid || ('mediaId' in ref ? ref.mediaId : null);
    if (!mediaId || !/^[\w-]+$/.test(String(mediaId))) return null;
    try {
        const response = await fetch(`./dev-beats/${mediaId}.json`);
        if (!response.ok) return null;
        const raw = await response.json() as DevJsonEvent[];
        if (!Array.isArray(raw)) return null;
        return raw
            .filter(event => Number.isFinite(event?.t))
            .map(event => ({ t: event.t, kind: event.kind === 'downbeat' ? 'downbeat' as const : 'beat' as const }))
            .sort((a, b) => a.t - b.t);
    } catch {
        return null;
    }
};

const resolveTrackBeats = async (song: SongResult): Promise<BeatFxTrackBeats> => {
    let note: string;
    try {
        const inApp = await analyseInApp(song);
        if (typeof inApp !== 'string') return inApp;
        note = inApp;
    } catch (error) {
        note = `beat_this failed: ${String(error)}`;
    }
    const fallback = await loadDevJson(song);
    if (fallback?.length) return { events: fallback, source: 'dev-json', note };
    return { events: [], source: 'none', note };
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
