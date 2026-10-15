import { beforeEach, describe, expect, it, vi } from 'vitest';

// test/unit/automix/beatGridCache.test.ts
// The whole-track beat list shared by automix and Sonnet's 智能动感模式.

const store = new Map<string, unknown>();
vi.mock('@/services/db', () => ({
    getFromCache: vi.fn(async (key: string) => store.get(key) ?? null),
    saveToCache: vi.fn(async (key: string, data: unknown) => {
        store.set(key, data);
    }),
}));

const { readStoredBeatGrid, storeBeatGrid } = await import('@/services/automix/beatGridCache');

describe('beat grid cache', () => {
    beforeEach(() => store.clear());

    it('round-trips a beat list under the track key', async () => {
        await storeBeatGrid('song-1', { beats: [0.5, 1, 1.5, 2], downbeats: [0.5, 2] });
        expect(await readStoredBeatGrid('song-1')).toEqual({ beats: [0.5, 1, 1.5, 2], downbeats: [0.5, 2] });
        expect(await readStoredBeatGrid('song-2')).toBeNull();
    });

    it('does not keep an empty list', async () => {
        await storeBeatGrid('song-1', { beats: [], downbeats: [] });
        expect(store.size).toBe(0);
    });

    it('ignores an entry from another version or a malformed one', async () => {
        store.set('beat_this_grid_song-1', { version: 0, beats: [1], downbeats: [] });
        expect(await readStoredBeatGrid('song-1')).toBeNull();
        store.set('beat_this_grid_song-1', { version: 1, beats: [1, 'x'], downbeats: [] });
        expect(await readStoredBeatGrid('song-1')).toBeNull();
    });
});
