import { getFromCache, saveToCache } from '../db';
import type { BeatGrid } from './beatThis';

// src/services/automix/beatGridCache.ts
// Beat This!'s raw answer for a whole track, kept on disk under the track's analysis key.
//
// Two features want it and neither should pay twice. Automix's profile only keeps the collapsed
// `{offset, period}` grid, while Sonnet's 智能动感 needs every beat and downbeat; whichever runs the
// model first stores the list here and the other reads it. It is also what lets a replayed track have
// beat FX from its first second instead of after a fresh analysis.
//
// Only whole tracks are stored: automix may measure just the head of a track that is still
// streaming, and a beat list that stops at twelve seconds would leave the FX silent for the rest.

/** Bumped when the model, the spectrogram or the peak picking changes what a track's beats are. */
const BEAT_GRID_VERSION = 1;

interface StoredBeatGrid {
    version: number;
    beats: number[];
    downbeats: number[];
}

const storageKey = (songKey: string) => `beat_this_grid_${songKey}`;

const isTimeList = (value: unknown): value is number[] =>
    Array.isArray(value) && value.every(time => typeof time === 'number' && Number.isFinite(time));

/** The stored whole-track beat list for `songKey`, or null. */
export const readStoredBeatGrid = async (songKey: string): Promise<BeatGrid | null> => {
    const stored = await getFromCache<StoredBeatGrid>(storageKey(songKey));
    if (!stored || stored.version !== BEAT_GRID_VERSION) return null;
    if (!isTimeList(stored.beats) || !isTimeList(stored.downbeats) || !stored.beats.length) return null;
    return { beats: stored.beats, downbeats: stored.downbeats };
};

/** Stores a whole-track beat list; an empty one is not worth keeping. */
export const storeBeatGrid = async (songKey: string, grid: BeatGrid): Promise<void> => {
    if (!grid.beats.length) return;
    const entry: StoredBeatGrid = { version: BEAT_GRID_VERSION, beats: [...grid.beats], downbeats: [...grid.downbeats] };
    await saveToCache(storageKey(songKey), entry);
};
