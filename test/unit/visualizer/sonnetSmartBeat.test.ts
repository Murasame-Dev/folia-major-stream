import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { DEFAULT_SONNET_TUNING, type SonnetTuning } from '@/types';
import { useVisualizerSettingsStore } from '@/stores/useVisualizerSettingsStore';
import { compressConfig, decompressConfig } from '@/utils/appearanceCodec';
import { beatImpact, downbeatImpact } from '@/stores/motionSignals';
import { beginBeatFxTrack, isBeatFxReady, setBeatFxEvents, stopBeatFx, tickBeatFx } from '@/services/beatFx/beatFxLive';
import { visualizerWantsBeatClock } from '@/components/visualizer/registry';
import {
    SONNET_BEAT_FX,
    applySonnetBeatFxLayers,
    createSonnetBeatFxReader,
    onlyFrameReadTuningChanged,
    restoreSonnetBeatFxLayers,
} from '@/components/visualizer/sonnet/sonnetBeatFx';

// test/unit/visualizer/sonnetSmartBeat.test.ts
// Sonnet's 智能动感模式: the setting at the store and codec boundaries, when the FX run at all, and how
// the 轻拍 / 重拍 strengths reach the frame.

const createLocalStorageMock = (): Storage => {
    const values = new Map<string, string>();
    return {
        get length() {
            return values.size;
        },
        getItem: key => values.get(key) ?? null,
        key: index => Array.from(values.keys())[index] ?? null,
        setItem: (key, value) => values.set(key, value),
        removeItem: key => values.delete(key),
        clear: () => values.clear(),
    };
};

const tuning = (patch: Partial<SonnetTuning> = {}): SonnetTuning => ({ ...DEFAULT_SONNET_TUNING, ...patch });

describe('Sonnet 智能动感模式 setting', () => {
    let storage: Storage;

    beforeEach(() => {
        storage = createLocalStorageMock();
        vi.stubGlobal('localStorage', storage);
        vi.stubGlobal('window', { localStorage: storage });
        useVisualizerSettingsStore.setState({ sonnetTuning: { ...DEFAULT_SONNET_TUNING } });
    });

    afterEach(() => {
        vi.unstubAllGlobals();
    });

    it('is off by default at full strength', () => {
        expect(DEFAULT_SONNET_TUNING.smartBeatEnabled).toBe(false);
        expect(DEFAULT_SONNET_TUNING.beatIntensity).toBe(1);
        expect(DEFAULT_SONNET_TUNING.downbeatIntensity).toBe(1);
    });

    it('persists the switch and clamps both strengths to 0..2', () => {
        useVisualizerSettingsStore.getState().handleSetSonnetTuning({
            smartBeatEnabled: true,
            beatIntensity: 5,
            downbeatIntensity: -1,
        });
        expect(useVisualizerSettingsStore.getState().sonnetTuning).toMatchObject({
            smartBeatEnabled: true,
            beatIntensity: 2,
            downbeatIntensity: 0,
        });
        expect(JSON.parse(storage.getItem('sonnet_tuning') ?? '{}')).toMatchObject({
            smartBeatEnabled: true,
            beatIntensity: 2,
            downbeatIntensity: 0,
        });
    });

    it('round-trips through visual config export and import', () => {
        const sonnetTuning = tuning({ smartBeatEnabled: true, beatIntensity: 1.4, downbeatIntensity: 0.3 });
        const decoded = decompressConfig(compressConfig({ sonnetTuning }));
        expect(decoded.sonnetTuning).toMatchObject({ smartBeatEnabled: true, beatIntensity: 1.4, downbeatIntensity: 0.3 });
    });

    it('imports an older config without the fields as off', () => {
        const { smartBeatEnabled: _s, beatIntensity: _b, downbeatIntensity: _d, ...older } = DEFAULT_SONNET_TUNING;
        const decoded = decompressConfig(compressConfig({ sonnetTuning: older }));
        expect(decoded.sonnetTuning).toMatchObject({ smartBeatEnabled: false, beatIntensity: 1, downbeatIntensity: 1 });
    });
});

describe('when the beat FX run', () => {
    it('is asked of the mode on screen: Sonnet with 智能动感模式 on, and no other mode', () => {
        expect(visualizerWantsBeatClock('sonnet', { sonnetTuning: tuning({ smartBeatEnabled: true }) })).toBe(true);
        expect(visualizerWantsBeatClock('sonnet', { sonnetTuning: tuning() })).toBe(false);
        expect(visualizerWantsBeatClock('tempera', { sonnetTuning: tuning({ smartBeatEnabled: true }) })).toBe(false);
    });
});

describe('beat clock readiness across a track change', () => {
    afterEach(() => {
        vi.useRealTimers();
        stopBeatFx();
    });

    it('holds the previous track ready while the next one loads, then drops if it has none', () => {
        vi.useFakeTimers();
        beginBeatFxTrack('a');
        setBeatFxEvents('a', [{ t: 1, kind: 'beat' }], 'beat_this');
        beginBeatFxTrack('b');
        expect(isBeatFxReady()).toBe(true);
        vi.advanceTimersByTime(2000);
        expect(isBeatFxReady()).toBe(false);
    });

    it('does not hold when the previous track had no beats either', () => {
        beginBeatFxTrack('a');
        beginBeatFxTrack('b');
        expect(isBeatFxReady()).toBe(false);
    });

    it('keeps holding across a run of quick skips', () => {
        vi.useFakeTimers();
        beginBeatFxTrack('a');
        setBeatFxEvents('a', [{ t: 1, kind: 'beat' }], 'beat_this');
        beginBeatFxTrack('b');
        vi.advanceTimersByTime(500);
        beginBeatFxTrack('c');
        expect(isBeatFxReady()).toBe(true);
        vi.advanceTimersByTime(1000);
        // 1.5 s after the first skip, but only 1 s after the second.
        expect(isBeatFxReady()).toBe(true);
        vi.advanceTimersByTime(1000);
        expect(isBeatFxReady()).toBe(false);
    });

    it('ends the hold the moment the next track answers', () => {
        beginBeatFxTrack('a');
        setBeatFxEvents('a', [{ t: 1, kind: 'beat' }], 'beat_this');
        beginBeatFxTrack('b');
        setBeatFxEvents('b', [], 'none');
        expect(isBeatFxReady()).toBe(false);
    });
});

describe('Sonnet beat FX frame', () => {
    afterEach(() => {
        stopBeatFx();
    });

    const installBeats = () => {
        beginBeatFxTrack('track');
        setBeatFxEvents('track', [{ t: 1, kind: 'beat' }, { t: 2, kind: 'downbeat' }], 'beat_this');
    };

    it('is null while the switch is off, even with beats installed', () => {
        installBeats();
        expect(createSonnetBeatFxReader()(undefined, tuning())).toBeNull();
    });

    it('is null while the track has no beats yet, so the original frame is drawn', () => {
        beginBeatFxTrack('track');
        expect(createSonnetBeatFxReader()(undefined, tuning({ smartBeatEnabled: true }))).toBeNull();
    });

    it('scales the beat layer by 轻拍 and the downbeat layer by 重拍, independently', () => {
        installBeats();
        beatImpact.set(1);
        downbeatImpact.set(1);
        const read = createSonnetBeatFxReader();

        const noDownbeat = read(undefined, tuning({ smartBeatEnabled: true, beatIntensity: 1, downbeatIntensity: 0 }))!;
        expect(noDownbeat.zoom).toBeCloseTo(1 + SONNET_BEAT_FX.zoomPercent / 100);
        expect(noDownbeat.tear).toBe(0);
        expect(noDownbeat.flare).toBe(0);
        expect(Math.abs(noDownbeat.rotation)).toBeCloseTo(SONNET_BEAT_FX.rotationDeg * Math.PI / 180);

        const noBeat = read(undefined, tuning({ smartBeatEnabled: true, beatIntensity: 0, downbeatIntensity: 2 }))!;
        expect(noBeat.zoom).toBeCloseTo(1 + 2 * SONNET_BEAT_FX.downbeatZoomPercent / 100);
        expect(noBeat.rotation).toBe(0);
        expect(noBeat.tear).toBeCloseTo(2 * SONNET_BEAT_FX.tear);
        expect(noBeat.flare).toBeCloseTo(2 * SONNET_BEAT_FX.flare);
    });

    it('keeps the line flash at rest while nothing is heard', () => {
        installBeats();
        const treble = { get: () => 200 };
        const bands = { treble } as unknown as Parameters<ReturnType<typeof createSonnetBeatFxReader>>[0];
        const read = createSonnetBeatFxReader();
        tickBeatFx(null, null);
        expect(read(bands, tuning({ smartBeatEnabled: true }))!.lineLift).toBe(0);
    });

    it('puts the line and halo layers back to their built alpha', () => {
        const view = {
            haloLayer: { alpha: 0.35 },
            haloBaseAlpha: 0.35,
            mgGeoLayer: { alpha: 1 },
            mgFixedGeoLayer: { alpha: 1 },
        };
        applySonnetBeatFxLayers(view, { zoom: 1, rotation: 0, tear: 0, tearSeed: 0, flare: 1, lineLift: 0 });
        expect(view.haloLayer.alpha).toBe(1);
        expect(view.mgGeoLayer.alpha).toBe(SONNET_BEAT_FX.lineBaseAlpha);
        restoreSonnetBeatFxLayers(view);
        expect(view.mgGeoLayer.alpha).toBe(1);
        expect(view.mgFixedGeoLayer.alpha).toBe(1);
        expect(view.haloLayer.alpha).toBe(0.35);
    });
});

describe('Sonnet tuning change', () => {
    const base: SonnetTuning = { ...DEFAULT_SONNET_TUNING, smartBeatEnabled: true };

    it('needs no scene rebuild when only the 轻拍 / 重拍 strengths moved', () => {
        expect(onlyFrameReadTuningChanged(base, { ...base, beatIntensity: 1.6, downbeatIntensity: 0.2 })).toBe(true);
        expect(onlyFrameReadTuningChanged(base, { ...base })).toBe(true);
    });

    it('rebuilds for anything a scene is built from, the switch included', () => {
        expect(onlyFrameReadTuningChanged(base, { ...base, smartBeatEnabled: false })).toBe(false);
        expect(onlyFrameReadTuningChanged(base, { ...base, beatIntensity: 1.6, enableTransitions: !base.enableTransitions })).toBe(false);
    });
});
