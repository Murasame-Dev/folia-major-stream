import type { SonnetTuning, VisualizerMode } from '../../types';

// src/services/beatFx/beatFxConfig.ts
// When the beat FX run, and the one developer knob left from the experiment.
//
// The FX are Sonnet's 智能动感 setting (`SonnetTuning.smartBeatEnabled`), so they run only while
// Sonnet is the visualizer on screen with that setting on, and never in static mode. Anything else
// skips every beat FX code path - no analysis, no clock - and the visualizers render as before.
//
//   localStorage['folia.dev.beatFxOffsetMs'] = '-20'  optional extra offset, positive = FX later

export const BEAT_FX_OFFSET_KEY = 'folia.dev.beatFxOffsetMs';

const readStorage = (key: string): string | null => {
    try {
        return typeof localStorage === 'undefined' ? null : localStorage.getItem(key);
    } catch {
        return null;
    }
};

const offsetMs = (() => {
    const value = Number(readStorage(BEAT_FX_OFFSET_KEY));
    return Number.isFinite(value) ? value : 0;
})();

/** Whether the beat FX should run: Sonnet on screen, 智能动感 on, not in static mode. */
export const isSmartBeatWanted = (
    mode: VisualizerMode,
    tuning: Pick<SonnetTuning, 'smartBeatEnabled'>,
    staticMode: boolean,
): boolean => mode === 'sonnet' && tuning.smartBeatEnabled && !staticMode;

/** The dev offset in seconds; positive delays the FX relative to the audio. */
export const getBeatFxOffsetSec = (): number => offsetMs / 1000;
