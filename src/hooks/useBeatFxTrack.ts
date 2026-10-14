import { useEffect } from 'react';
import { selectDisplaySong, usePlaybackStore } from '../stores/usePlaybackStore';
import { isBeatFxEnabled } from '../services/beatFx/beatFxDevFlag';
import { beginBeatFxTrack, setBeatFxEvents } from '../services/beatFx/beatFxLive';
import { loadBeatFxTrackBeats } from '../services/beatFx/beatFxSource';
import { getPlaybackAnalysisKey } from '../services/playbackRecovery/representationRegistry';

// src/hooks/useBeatFxTrack.ts
// Loads the beat list of the on-screen track into the live beat clock whenever the track changes.
// Only the track identity enters React; the per-frame clock lives in services/beatFx/beatFxLive.

export function useBeatFxTrack() {
    const song = usePlaybackStore(selectDisplaySong);
    const key = isBeatFxEnabled() && song ? getPlaybackAnalysisKey(song) : null;

    useEffect(() => {
        if (!isBeatFxEnabled()) return;
        beginBeatFxTrack(key);
        if (!key || !song) return;
        let cancelled = false;
        void loadBeatFxTrackBeats(song).then((result) => {
            if (cancelled) return;
            setBeatFxEvents(key, result.events, result.source, result.note);
            console.info(`[BeatFx] ${result.events.length} events from ${result.source}${result.note ? ` (${result.note})` : ''}`);
        });
        return () => {
            cancelled = true;
        };
        // Keyed on the track identity only; the song object itself changes for unrelated reasons.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [key]);
}
