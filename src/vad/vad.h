#pragma once

#include <stdbool.h>

#define VAD_FRAME_MS 20
#define VAD_MAX_FRAME_SAMPLES 2048
#define VAD_START_FRAMES 3
#define VAD_HANGOVER_FRAMES 20
#define VAD_THRESHOLD_MULTIPLIER 2.5f
#define VAD_NOISE_ADAPT_RATE 0.05f
#define VAD_MIN_NOISE_FLOOR 0.0001f
#define VAD_WARMUP_FRAMES 15

typedef enum
{
    VAD_EVENT_NONE,
    VAD_EVENT_SPEECH_STARTED,
    VAD_EVENT_SPEECH_ENDED
} VadEvent;

void initVAD(int sampleRate);
VadEvent vadProcess(const float* samples, int sampleCount);
float vadGetLevel(void);
bool vadIsSpeaking(void);
void resetVAD(void);