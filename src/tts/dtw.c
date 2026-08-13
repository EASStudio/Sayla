// Module includes
#include "dtw.h"

// Core includes
#include <math.h>
#include <stdlib.h>
#include <float.h>

static float frameDist(const float* a, const float* b)
{
    float sum = 0.0f;
    for (int i = 0; i < FEATURE_DIM; i++)
    {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sqrtf(sum);
}

float dtwDistance(const float* a, int aFrames, const float* b, int bFrames)
{
    if (aFrames <= 0 || bFrames <= 0) return FLT_MAX;

    int rows = aFrames + 1, cols = bFrames + 1;
    float* D = (float*)malloc(sizeof(float) * (size_t)rows * (size_t)cols);
    int* pathLen = (int*)malloc(sizeof(int) * (size_t)rows * (size_t)cols);
    if (D == NULL || pathLen == NULL) { free(D); free(pathLen); return FLT_MAX; }

    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            D[i * cols + j] = FLT_MAX;
    D[0] = 0.0f;
    pathLen[0] = 0;

    for (int i = 1; i <= aFrames; i++)
    {
        for (int j = 1; j <= bFrames; j++)
        {
            float cost = frameDist(&a[(i - 1) * FEATURE_DIM], &b[(j - 1) * FEATURE_DIM]);

            float diag = D[(i - 1) * cols + (j - 1)];
            float up   = D[(i - 1) * cols + j];
            float left = D[i * cols + (j - 1)];

            float best = diag;
            int bestLen = pathLen[(i - 1) * cols + (j - 1)];
            if (up < best)   { best = up;   bestLen = pathLen[(i - 1) * cols + j]; }
            if (left < best) { best = left; bestLen = pathLen[i * cols + (j - 1)]; }

            D[i * cols + j] = cost + best;
            pathLen[i * cols + j] = bestLen + 1;
        }
    }

    float totalCost = D[aFrames * cols + bFrames];
    int totalPathLen = pathLen[aFrames * cols + bFrames];
    float result = (totalPathLen > 0) ? (totalCost / (float)totalPathLen) : FLT_MAX;

    free(D);
    free(pathLen);
    return result;
}