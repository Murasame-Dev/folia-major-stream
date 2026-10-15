import { useEffect, useSyncExternalStore } from 'react';
import { modelsPresent, refreshModelAvailability, subscribeModelAvailability } from '../services/automix/modelAvailability';

// src/hooks/useSmartBeatAvailability.ts
// Whether Sonnet's 智能动感模式 can do anything on this install, for the setting to say so next to its
// switch. The switch itself stays usable either way - it is a preference, and it starts working the
// moment the reason goes away (a download lands) - but a switch that is on and silently inert is the
// failure this is here to prevent.

export type SmartBeatAvailability = 'ready' | 'desktopOnly' | 'noGpu' | 'noModel';

const hasBeatThisBridge = () => typeof window !== 'undefined' && typeof window.electron?.runBeatThis === 'function';
const hasWebGpu = () => typeof navigator !== 'undefined' && 'gpu' in navigator;

export function useSmartBeatAvailability(): SmartBeatAvailability {
    const present = useSyncExternalStore(subscribeModelAvailability, modelsPresent, modelsPresent);

    useEffect(() => {
        void refreshModelAvailability();
    }, []);

    if (!hasBeatThisBridge()) return 'desktopOnly';
    if (!hasWebGpu()) return 'noGpu';
    return present.beat_this ? 'ready' : 'noModel';
}
