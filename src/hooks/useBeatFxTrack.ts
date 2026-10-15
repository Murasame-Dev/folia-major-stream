import { useEffect } from 'react';
import { selectDisplaySong, usePlaybackStore } from '../stores/usePlaybackStore';
import { useThemeSettingsStore } from '../stores/useThemeSettingsStore';
import { useVisualizerSettingsStore } from '../stores/useVisualizerSettingsStore';
import { isSmartBeatWanted } from '../services/beatFx/beatFxConfig';
import { beginBeatFxTrack, setBeatFxEvents, stopBeatFx } from '../services/beatFx/beatFxLive';
import { loadBeatFxTrackBeats } from '../services/beatFx/beatFxSource';
import { getPlaybackAnalysisKey } from '../services/playbackRecovery/representationRegistry';

// src/hooks/useBeatFxTrack.ts
// Loads the beat list of the on-screen track into the live beat clock whenever the track changes,
// while Sonnet's 智能动感 is on. Only the track identity and the switch enter React; the per-frame
// clock lives in services/beatFx/beatFxLive.

/** Whether the beat clock should be stepped; the caller's frame loop reads it. */
export function useBeatFxTrack(): boolean {
    const staticMode = useThemeSettingsStore(state => state.staticMode);
    const wanted = useVisualizerSettingsStore(state => isSmartBeatWanted(state.visualizerMode, state.sonnetTuning, staticMode));
    const song = usePlaybackStore(selectDisplaySong);
    const key = wanted && song ? getPlaybackAnalysisKey(song) : null;

    useEffect(() => {
        if (!wanted) {
            stopBeatFx();
            return;
        }
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
        // Keyed on the track identity and the switch only; the song object changes for unrelated reasons.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [key, wanted]);

    return wanted;
}
