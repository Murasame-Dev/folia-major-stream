// src/services/beatFx/beatFxDevFlag.ts
// Experimental beat-reactive visual FX, gated behind a developer-only localStorage flag.
//
// Not a setting: no settings UI, no import/export, no command palette. Read once at module load,
// so toggling it needs a reload. With the flag off every beat FX code path is skipped and the
// visualizers render exactly as before.
//
//   localStorage['folia.dev.beatFx'] = '1'           enable
//   localStorage['folia.dev.beatFxOffsetMs'] = '-20'  optional extra offset, positive = FX later

export const BEAT_FX_FLAG_KEY = 'folia.dev.beatFx';
export const BEAT_FX_OFFSET_KEY = 'folia.dev.beatFxOffsetMs';

const readStorage = (key: string): string | null => {
    try {
        return typeof localStorage === 'undefined' ? null : localStorage.getItem(key);
    } catch {
        return null;
    }
};

const enabled = readStorage(BEAT_FX_FLAG_KEY) === '1';
const offsetMs = (() => {
    const value = Number(readStorage(BEAT_FX_OFFSET_KEY));
    return Number.isFinite(value) ? value : 0;
})();

export const isBeatFxEnabled = (): boolean => enabled;

/** The dev offset in seconds; positive delays the FX relative to the audio. */
export const getBeatFxOffsetSec = (): number => offsetMs / 1000;
