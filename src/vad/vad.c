// Module includes
#include "vad.h"

// Core includes
#include <string.h>
#include <math.h>

typedef enum
{
    VAD_STATE_SILENCE,
    VAD_STATE_MAYBE_SPEECH, // energy is up, confirming it's not just a blip
    VAD_STATE_SPEECH
} VadState;

static int frameSize = 320; // recomputed in initVAD() from the real sample rate
static float frameBuffer[VAD_MAX_FRAME_SAMPLES];
static int frameBufferFill = 0;

static VadState state = VAD_STATE_SILENCE;
static int activeFrameCount = 0; // consecutive active frames while in MAYBE_SPEECH
static int hangoverCount = 0;    // consecutive-inactive countdown while in SPEECH

static float noiseFloor = VAD_MIN_NOISE_FLOOR;
static float smoothedLevel = 0.0f; // UI-facing only, not used in the decision

static int warmupFramesRemaining = VAD_WARMUP_FRAMES;
static double warmupEnergySum = 0.0;

void initVAD(int sampleRate)
{
    frameSize = (sampleRate * VAD_FRAME_MS) / 1000;
    if (frameSize < 1) frameSize = 1;
    if (frameSize > VAD_MAX_FRAME_SAMPLES) frameSize = VAD_MAX_FRAME_SAMPLES;

    resetVAD();
}

void resetVAD(void)
{
    frameBufferFill = 0;
    state = VAD_STATE_SILENCE;
    activeFrameCount = 0;
    hangoverCount = 0;
    noiseFloor = VAD_MIN_NOISE_FLOOR;
    smoothedLevel = 0.0f;
    warmupFramesRemaining = VAD_WARMUP_FRAMES;
    warmupEnergySum = 0.0;
}

static float frameEnergy(const float* frame, int n)
{
    double sumSquares = 0.0;
    for (int i = 0; i < n; i++)
        sumSquares += (double)frame[i] * (double)frame[i];
    return (float)sqrt(sumSquares / (double)n); // RMS
}

// Runs the state machine for exactly one analysis frame.
static VadEvent processOneFrame(const float* frame, int n)
{
    float energy = frameEnergy(frame, n);
    smoothedLevel = smoothedLevel * 0.7f + energy * 0.3f;

    if (warmupFramesRemaining > 0)
    {
        // Pure calibration: seed the noise floor from real incoming
        // audio before making any speech/silence decision, instead of
        // guessing a fixed constant that could be wildly wrong for the
        // actual microphone and room. No detection happens yet.
        warmupEnergySum += energy;
        warmupFramesRemaining--;

        if (warmupFramesRemaining == 0)
        {
            noiseFloor = (float)(warmupEnergySum / VAD_WARMUP_FRAMES);
            if (noiseFloor < VAD_MIN_NOISE_FLOOR) noiseFloor = VAD_MIN_NOISE_FLOOR;
        }

        return VAD_EVENT_NONE;
    }

    float threshold = noiseFloor * VAD_THRESHOLD_MULTIPLIER;
    bool active = energy > threshold;

    VadEvent event = VAD_EVENT_NONE;

    switch (state)
    {
        case VAD_STATE_SILENCE:
            // Only adapt the floor while confidently silent -- adapting
            // during speech would let sustained talking slowly drag the
            // floor up until the VAD stopped noticing speech at all.
            noiseFloor = noiseFloor * (1.0f - VAD_NOISE_ADAPT_RATE) + energy * VAD_NOISE_ADAPT_RATE;
            if (noiseFloor < VAD_MIN_NOISE_FLOOR) noiseFloor = VAD_MIN_NOISE_FLOOR;

            if (active)
            {
                state = VAD_STATE_MAYBE_SPEECH;
                activeFrameCount = 1;
            }
            break;

        case VAD_STATE_MAYBE_SPEECH:
            if (active)
            {
                activeFrameCount++;
                if (activeFrameCount >= VAD_START_FRAMES)
                {
                    state = VAD_STATE_SPEECH;
                    hangoverCount = VAD_HANGOVER_FRAMES;
                    event = VAD_EVENT_SPEECH_STARTED;
                }
            }
            else
            {
                // Energy dropped back down before speech was confirmed
                // -- it was a blip (click, pop, cough), not real speech.
                state = VAD_STATE_SILENCE;
                activeFrameCount = 0;
            }
            break;

        case VAD_STATE_SPEECH:
            if (active)
            {
                hangoverCount = VAD_HANGOVER_FRAMES; // still talking -- reset the countdown
            }
            else
            {
                hangoverCount--;
                if (hangoverCount <= 0)
                {
                    state = VAD_STATE_SILENCE;
                    activeFrameCount = 0;
                    event = VAD_EVENT_SPEECH_ENDED;
                }
            }
            break;
    }

    return event;
}

VadEvent vadProcess(const float* samples, int sampleCount)
{
    VadEvent resultEvent = VAD_EVENT_NONE;
    int pos = 0;

    while (pos < sampleCount)
    {
        int room = frameSize - frameBufferFill;
        int take = sampleCount - pos;
        if (take > room) take = room;

        memcpy(frameBuffer + frameBufferFill, samples + pos, sizeof(float) * (size_t)take);
        frameBufferFill += take;
        pos += take;

        if (frameBufferFill >= frameSize)
        {
            VadEvent event = processOneFrame(frameBuffer, frameSize);
            if (event != VAD_EVENT_NONE) resultEvent = event;
            frameBufferFill = 0;
        }
    }

    return resultEvent;
}

float vadGetLevel(void)
{
    // Normalized against the same threshold used for the speech
    // decision (times a further 2x so it doesn't peg at 1.0 the instant
    // energy crosses the trigger point), so "1.0" roughly means
    // "clearly speech" rather than an arbitrary fixed scale that would
    // need separate tuning per microphone.
    float threshold = noiseFloor * VAD_THRESHOLD_MULTIPLIER;
    if (threshold < VAD_MIN_NOISE_FLOOR) threshold = VAD_MIN_NOISE_FLOOR;

    float level = smoothedLevel / (threshold * 2.0f);
    if (level < 0.0f) level = 0.0f;
    if (level > 1.0f) level = 1.0f;
    return level;
}

bool vadIsSpeaking(void)
{
    return state == VAD_STATE_SPEECH;
}