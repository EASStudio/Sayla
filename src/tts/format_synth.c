// Module includes
#include "format_synth.h"
#include "resonator.h"

// Core includes
#include <math.h>

static unsigned int synthRngState = 12345;
static float synthRandFloat(void)
{
    synthRngState = synthRngState * 1103515245u + 12345u;
    return ((float)((synthRngState >> 16) & 0x7fff) / 32768.0f) * 2.0f - 1.0f;
}

int synthesizeSpeech(const Phoneme* phonemes, int phonemeCount,
                      short* outSamples, int maxSamples, int sampleRate)
{
    if (phonemeCount <= 0) return 0;

    Resonator r1, r2, r3;

    const PhonemeInfo* first = &phonemeTable[phonemes[0]];
    float curF1 = first->f1 > 0 ? first->f1 : 300.0f;
    float curF2 = first->f2 > 0 ? first->f2 : 1200.0f;
    float curF3 = first->f3 > 0 ? first->f3 : 2500.0f;
    float curBW1 = first->bw1 > 0 ? first->bw1 : 100.0f;
    float curBW2 = first->bw2 > 0 ? first->bw2 : 100.0f;
    float curBW3 = first->bw3 > 0 ? first->bw3 : 150.0f;

    resonatorSetup(&r1, curF1, curBW1, (float)sampleRate);
    resonatorSetup(&r2, curF2, curBW2, (float)sampleRate);
    resonatorSetup(&r3, curF3, curBW3, (float)sampleRate);

    double totalMs = 0.0;
    for (int i = 0; i < phonemeCount; i++) totalMs += phonemeTable[phonemes[i]].durationMs;
    if (totalMs < 1.0) totalMs = 1.0;

    double elapsedMs = 0.0;
    float pitchPhase = 0.0f; // samples into the current pitch period
    int samplesWritten = 0;

    for (int p = 0; p < phonemeCount && samplesWritten < maxSamples; p++)
    {
        const PhonemeInfo* info = &phonemeTable[phonemes[p]];
        int phonemeSamples = (info->durationMs * sampleRate) / 1000;
        if (phonemeSamples < 1) phonemeSamples = 1;

        int transitionSamples = (int)(FORMANT_TRANSITION_MS * sampleRate / 1000.0f);
        if (transitionSamples > phonemeSamples) transitionSamples = phonemeSamples;
        if (transitionSamples < 1) transitionSamples = 1;

        int attackSamples = (int)(phonemeSamples * ATTACK_FRACTION);
        int decaySamples = (int)(phonemeSamples * DECAY_FRACTION);

        float startF1 = curF1, startF2 = curF2, startF3 = curF3;
        float startBW1 = curBW1, startBW2 = curBW2, startBW3 = curBW3;
        float targetF1 = info->f1 > 0 ? info->f1 : startF1;
        float targetF2 = info->f2 > 0 ? info->f2 : startF2;
        float targetF3 = info->f3 > 0 ? info->f3 : startF3;
        float targetBW1 = info->bw1 > 0 ? info->bw1 : startBW1;
        float targetBW2 = info->bw2 > 0 ? info->bw2 : startBW2;
        float targetBW3 = info->bw3 > 0 ? info->bw3 : startBW3;

        for (int s = 0; s < phonemeSamples && samplesWritten < maxSamples; s++)
        {
            float t = (float)s / (float)transitionSamples;
            if (t > 1.0f) t = 1.0f;

            curF1 = startF1 + (targetF1 - startF1) * t;
            curF2 = startF2 + (targetF2 - startF2) * t;
            curF3 = startF3 + (targetF3 - startF3) * t;
            curBW1 = startBW1 + (targetBW1 - startBW1) * t;
            curBW2 = startBW2 + (targetBW2 - startBW2) * t;
            curBW3 = startBW3 + (targetBW3 - startBW3) * t;

            // Retune without resetting -- see resonator.h for why that
            // distinction matters.
            resonatorSetCoefficients(&r1, curF1, curBW1, (float)sampleRate);
            resonatorSetCoefficients(&r2, curF2, curBW2, (float)sampleRate);
            resonatorSetCoefficients(&r3, curF3, curBW3, (float)sampleRate);

            float source;
            if (info->voiced)
            {
                double frac = elapsedMs / totalMs;
                float f0 = BASE_F0_START + (BASE_F0_END - BASE_F0_START) * (float)frac;
                float periodSamples = (float)sampleRate / f0;

                pitchPhase += 1.0f;
                if (pitchPhase >= periodSamples)
                {
                    pitchPhase -= periodSamples;
                    source = 1.0f; // one impulse per pitch period
                }
                else
                {
                    source = 0.0f;
                }
            }
            else
            {
                source = synthRandFloat() * 0.3f; // noise source for unvoiced sounds
            }

            float out1 = resonatorProcess(&r1, source) * F1_WEIGHT;
            float out2 = resonatorProcess(&r2, source) * F2_WEIGHT;
            float out3 = resonatorProcess(&r3, source) * F3_WEIGHT;
            float mixed = out1 + out2 + out3;

            float envelope = 1.0f;
            if (attackSamples > 0 && s < attackSamples)
                envelope = (float)s / (float)attackSamples;
            else if (decaySamples > 0 && s >= phonemeSamples - decaySamples)
                envelope = (float)(phonemeSamples - s) / (float)decaySamples;
            if (envelope < 0.0f) envelope = 0.0f;
            if (envelope > 1.0f) envelope = 1.0f;

            float sampleValue = mixed * envelope * info->amplitude * OUTPUT_GAIN;
            if (sampleValue > 32000.0f) sampleValue = 32000.0f;
            if (sampleValue < -32000.0f) sampleValue = -32000.0f;

            outSamples[samplesWritten++] = (short)sampleValue;
            elapsedMs += 1000.0 / sampleRate;
        }
    }

    return samplesWritten;
}