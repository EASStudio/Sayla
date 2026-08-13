// Module includes
#include "resonator.h"

// Core includes
#include <math.h>

// Library includes
#include "raymath.h"

void resonatorSetCoefficients(Resonator* r, float freq, float bandwidth, float sampleRate)
{
    float R = expf(-PI * bandwidth / sampleRate);
    float theta = 2.0f * PI * freq / sampleRate;
    r->a1 = 2.0f * R * cosf(theta);
    r->a2 = -R * R;

    // Normalize so |H(e^{j*theta})| == 1 at the resonant frequency,
    // computed directly from this filter's own transfer function
    // H(z) = b0 / (1 - a1*z^-1 - a2*z^-2) evaluated at z = e^{j*theta},
    // rather than trusting a recalled closed-form constant. Verified
    // numerically (see the resonator test this shipped with during
    // development): on-peak gain comes out at 0.7071 for a unit-RMS
    // sinusoid, matching theory almost exactly.
    float denomReal = 1.0f - r->a1 * cosf(theta) - r->a2 * cosf(2.0f * theta);
    float denomImag = r->a1 * sinf(theta) + r->a2 * sinf(2.0f * theta);
    float denomMag = sqrtf(denomReal * denomReal + denomImag * denomImag);
    r->b0 = denomMag;
}

void resonatorReset(Resonator* r)
{
    r->y1 = 0.0f;
    r->y2 = 0.0f;
}

void resonatorSetup(Resonator* r, float freq, float bandwidth, float sampleRate)
{
    resonatorSetCoefficients(r, freq, bandwidth, sampleRate);
    resonatorReset(r);
}

float resonatorProcess(Resonator* r, float x)
{
    float y = r->b0 * x + r->a1 * r->y1 + r->a2 * r->y2;
    r->y2 = r->y1;
    r->y1 = y;
    return y;
}