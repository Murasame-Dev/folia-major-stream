import { useEffect, useState, useSyncExternalStore } from 'react';
import { modelsPresent, refreshModelAvailability, subscribeModelAvailability } from '../services/analysis/modelAvailability';
import { probeBeatThisGpu } from '../services/analysis/beatThisGpu';

// src/hooks/useSmartBeatAvailability.ts
// Whether beat FX (Sonnet's 智能动感模式, and any mode that adds them) can do anything on this
// install, for a setting to say so next to its switch. The switch itself stays usable either way - it
// is a preference, and it starts working the moment the reason goes away (a download lands) - but a
// switch that is on and silently inert is the failure this is here to prevent.

export type SmartBeatAvailability = 'ready' | 'desktopOnly' | 'noGpu' | 'noModel';

const hasBeatThisBridge = () => typeof window !== 'undefined' && typeof window.electron?.runBeatThis === 'function';

/**
 * Asked only while `enabled` - the switch is on. Off, nothing is probed (no adapter request, no model
 * check) and the answer is 'ready', which a caller shows nothing for.
 */
export function useSmartBeatAvailability(enabled: boolean): SmartBeatAvailability {
    const present = useSyncExternalStore(subscribeModelAvailability, modelsPresent, modelsPresent);
    // Optimistic until the probe answers: a hint flashing "no GPU" on every open would be worse.
    const [gpu, setGpu] = useState(true);

    useEffect(() => {
        if (!enabled) return;
        let cancelled = false;
        void refreshModelAvailability();
        void probeBeatThisGpu().then((found) => {
            if (!cancelled) setGpu(found);
        });
        return () => {
            cancelled = true;
        };
    }, [enabled]);

    if (!enabled) return 'ready';
    if (!hasBeatThisBridge()) return 'desktopOnly';
    if (!gpu) return 'noGpu';
    return present.beat_this ? 'ready' : 'noModel';
}
