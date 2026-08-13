#pragma once

#include "../vad/audio_features.h" // FEATURE_DIM

float dtwDistance(const float* a, int aFrames, const float* b, int bFrames);