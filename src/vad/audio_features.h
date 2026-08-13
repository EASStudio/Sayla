#pragma once

#define FEATURE_DIM 8
#define ANALYSIS_FRAME_MS 25
#define ANALYSIS_HOP_MS 10 // frames overlap (hop < frame length) for smoother time resolution

int extractFeatures(const float* samples, int sampleCount, int sampleRate, float* outFeatures, int maxFrames);