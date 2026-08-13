#pragma once

// A single resonant (two-pole) filter, the building block formant_synth.c
// combines three of (one per formant) to shape a source waveform into a
// vowel-like or consonant-like spectrum. Verified against a standalone
// numerical test before use: unit gain at the target frequency, genuine
// selectivity (off-target frequencies attenuated), and stable (bounded)
// impulse response across the realistic formant frequency/bandwidth
// range.

typedef struct
{
    float a1, a2, b0;
    float y1, y2; // filter history (previous two outputs)
} Resonator;

void resonatorSetCoefficients(Resonator* r, float freq, float bandwidth, float sampleRate);
void resonatorReset(Resonator* r);
void resonatorSetup(Resonator* r, float freq, float bandwidth, float sampleRate);
float resonatorProcess(Resonator* r, float x);