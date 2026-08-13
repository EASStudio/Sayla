// Module includes
#include "audio_features.h"
#include "../tts/resonator.h"

// Core includes
#include <math.h>

static const float filterFreqs[FEATURE_DIM] = { 200, 400, 700, 1100, 1600, 2200, 3000, 4000 };
static const float filterBWs[FEATURE_DIM]   = { 100, 150, 200, 250, 300, 400, 500, 600 };

int extractFeatures(const float* samples, int sampleCount, int sampleRate, float* outFeatures, int maxFrames)
{
    int frameSize = (sampleRate * ANALYSIS_FRAME_MS) / 1000;
    int hopSize = (sampleRate * ANALYSIS_HOP_MS) / 1000;
    if (frameSize < 1) frameSize = 1;
    if (hopSize < 1) hopSize = 1;

    int frameCount = 0;
    int pos = 0;

    while (pos + frameSize <= sampleCount && frameCount < maxFrames)
    {
        for (int f = 0; f < FEATURE_DIM; f++)
        {
            Resonator r;
            resonatorSetup(&r, filterFreqs[f], filterBWs[f], (float)sampleRate);

            double sumSquares = 0.0;
            for (int i = 0; i < frameSize; i++)
            {
                float y = resonatorProcess(&r, samples[pos + i]);
                sumSquares += (double)y * y;
            }
            float energy = (float)sqrt(sumSquares / frameSize);

            outFeatures[frameCount * FEATURE_DIM + f] = logf(energy + 1e-6f);
        }

        frameCount++;
        pos += hopSize;
    }

    return frameCount;
}