import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { SongResult } from '@/types';

// test/unit/analysis/trackAudio.test.ts
// The one "where are this track's bytes" ladder every analyser reads through. Never the network.

const { getCachedSongAudioBlob } = vi.hoisted(() => ({ getCachedSongAudioBlob: vi.fn() }));
vi.mock('@/services/onlineMusic/resourceCache', () => ({ getCachedSongAudioBlob }));

const { readLocalTrackBytes, registerLocalTrackFileReader } = await import('@/services/analysis/trackAudio');

const online = { id: 'o', sourceRef: { kind: 'online', providerId: 'kugou', mediaId: 'o' } } as unknown as SongResult;
const local = { id: 'l', isLocal: true, localRef: { songId: 'l' }, sourceRef: { kind: 'local', songId: 'l' } } as unknown as SongResult;
const bytes = (n: number) => new ArrayBuffer(n);

describe('readLocalTrackBytes', () => {
    beforeEach(() => {
        getCachedSongAudioBlob.mockReset().mockResolvedValue(null);
        registerLocalTrackFileReader(null);
        vi.unstubAllGlobals();
    });

    it('reads the representation it is handed, and does not fall through when that fails', async () => {
        vi.stubGlobal('fetch', vi.fn(async () => ({ ok: false, status: 404 })));
        getCachedSongAudioBlob.mockResolvedValue(new Blob([new Uint8Array(3)]));
        const found = await readLocalTrackBytes(online, {
            representation: { url: 'blob:rep' } as never,
        });
        expect(found).toEqual({ skipped: 'the playback representation could not be read' });
        expect(getCachedSongAudioBlob).not.toHaveBeenCalled();
    });

    it('answers null for an uncached online track, leaving any download to the caller', async () => {
        const fetch = vi.fn();
        vi.stubGlobal('fetch', fetch);
        expect(await readLocalTrackBytes(online, { representation: null, audioUrl: 'https://cdn/x.mp3' })).toBeNull();
        expect(fetch).not.toHaveBeenCalled();
    });

    it('reads a local track with no URL through the registered file reader', async () => {
        const reader = vi.fn(async () => bytes(5));
        registerLocalTrackFileReader(reader);
        const found = await readLocalTrackBytes(local, { representation: null });
        expect(found && 'bytes' in found ? found.bytes.byteLength : -1).toBe(5);
        expect(reader).toHaveBeenCalledWith(local);
    });

    it('never asks the registered file reader about an online track', async () => {
        const reader = vi.fn(async () => bytes(5));
        registerLocalTrackFileReader(reader);
        expect(await readLocalTrackBytes(online, { representation: null })).toBeNull();
        expect(reader).not.toHaveBeenCalled();
    });
});
