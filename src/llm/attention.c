// Module includes
#include "attention.h"

// Core includes
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Start close to zero so early training isn't dominated by a lucky/unlucky large weight.
static double randomSmall(void)
{
    return ((double)rand() / (double)RAND_MAX) * 0.2 - 0.1;
}

static double* allocMatrix(int rows, int cols)
{
    return (double*)malloc(sizeof(double) * (size_t)rows * (size_t)cols);
}

void initAttentionLayer(AttentionLayer* attn, int maxSeqLen, int embedDim, int headDim)
{
    attn->maxSeqLen = maxSeqLen;
    attn->embedDim = embedDim;
    attn->headDim = headDim;
    attn->lastSeqLen = 0;

    attn->Wq = allocMatrix(embedDim, headDim);
    attn->Wk = allocMatrix(embedDim, headDim);
    attn->Wv = allocMatrix(embedDim, headDim);

    for (int i = 0; i < embedDim * headDim; i++)
    {
        attn->Wq[i] = randomSmall();
        attn->Wk[i] = randomSmall();
        attn->Wv[i] = randomSmall();
    }

    attn->Q = allocMatrix(maxSeqLen, headDim);
    attn->K = allocMatrix(maxSeqLen, headDim);
    attn->V = allocMatrix(maxSeqLen, headDim);
    attn->alpha = allocMatrix(maxSeqLen, maxSeqLen);
    attn->out = allocMatrix(maxSeqLen, headDim);
}

double* attentionForward(AttentionLayer* attn, const double* X, int seqLen)
{
    int d = attn->embedDim;
    int hd = attn->headDim;
    int T = attn->maxSeqLen;
    attn->lastSeqLen = seqLen;

    // Q = X Wq, K = X Wk, V = X Wv
    for (int i = 0; i < seqLen; i++)
    {
        for (int k = 0; k < hd; k++)
        {
            double q = 0.0, kk = 0.0, v = 0.0;
            for (int e = 0; e < d; e++)
            {
                double x = X[i * d + e];
                q  += x * attn->Wq[e * hd + k];
                kk += x * attn->Wk[e * hd + k];
                v  += x * attn->Wv[e * hd + k];
            }
            attn->Q[i * hd + k] = q;
            attn->K[i * hd + k] = kk;
            attn->V[i * hd + k] = v;
        }
    }

    double scale = 1.0 / sqrt((double)hd);

    for (int i = 0; i < seqLen; i++)
    {
        // Raw causal scores for row i, positions 0..i only.
        double maxScore = -1.0e300;
        for (int j = 0; j <= i; j++)
        {
            double s = 0.0;
            for (int k = 0; k < hd; k++)
                s += attn->Q[i * hd + k] * attn->K[j * hd + k];
            s *= scale;
            attn->alpha[i * T + j] = s;
            if (s > maxScore) maxScore = s;
        }

        // Softmax over 0..i (numerically stable).
        double sumExp = 0.0;
        for (int j = 0; j <= i; j++)
        {
            double e = exp(attn->alpha[i * T + j] - maxScore);
            attn->alpha[i * T + j] = e;
            sumExp += e;
        }

        for (int j = 0; j <= i; j++)
            attn->alpha[i * T + j] /= sumExp;

        // out[i] = sum_{j<=i} alpha[i][j] * V[j]
        for (int k = 0; k < hd; k++)
        {
            double o = 0.0;
            for (int j = 0; j <= i; j++)
                o += attn->alpha[i * T + j] * attn->V[j * hd + k];
            attn->out[i * hd + k] = o;
        }
    }

    return attn->out;
}

void attentionBackward(AttentionLayer* attn, const double* X, const double* dOut, int seqLen, double* dX, double lr)
{
    int d = attn->embedDim;
    int hd = attn->headDim;
    int T = attn->maxSeqLen;
    double scale = 1.0 / sqrt((double)hd);

    double* dQ = allocMatrix(seqLen, hd);
    double* dK = allocMatrix(seqLen, hd);
    double* dV = allocMatrix(seqLen, hd);
    memset(dK, 0, sizeof(double) * (size_t)seqLen * (size_t)hd);
    memset(dV, 0, sizeof(double) * (size_t)seqLen * (size_t)hd);

    double* dAlpha = (double*)malloc(sizeof(double) * (size_t)seqLen);
    double* dScore = (double*)malloc(sizeof(double) * (size_t)seqLen);

    // Pass 1: for each query position i, backprop through the softmax and
    // the out[i] = alpha[i] . V weighted sum, producing dQ[i] directly and
    // accumulating into dK/dV (since K[j]/V[j] affect every i >= j).
    for (int i = 0; i < seqLen; i++)
    {
        for (int j = 0; j <= i; j++)
        {
            double dot = 0.0;
            for (int k = 0; k < hd; k++)
                dot += dOut[i * hd + k] * attn->V[j * hd + k];
            dAlpha[j] = dot;
        }

        double weightedSum = 0.0;
        for (int j = 0; j <= i; j++)
            weightedSum += attn->alpha[i * T + j] * dAlpha[j];

        for (int j = 0; j <= i; j++)
            dScore[j] = attn->alpha[i * T + j] * (dAlpha[j] - weightedSum);

        for (int k = 0; k < hd; k++)
        {
            double dq = 0.0;
            for (int j = 0; j <= i; j++)
                dq += dScore[j] * attn->K[j * hd + k];
            dQ[i * hd + k] = dq * scale;
        }

        for (int j = 0; j <= i; j++)
        {
            for (int k = 0; k < hd; k++)
            {
                dK[j * hd + k] += dScore[j] * attn->Q[i * hd + k] * scale;
                dV[j * hd + k] += attn->alpha[i * T + j] * dOut[i * hd + k];
            }
        }
    }

    // Pass 2: dWq/dWk/dWv = X^T dQ / dK / dV, and dX[i] from dQ[i]/dK[i]/dV[i]
    // via the transposed projections. Accumulate weight gradients first...
    double* dWq = allocMatrix(d, hd);
    double* dWk = allocMatrix(d, hd);
    double* dWv = allocMatrix(d, hd);
    memset(dWq, 0, sizeof(double) * (size_t)d * (size_t)hd);
    memset(dWk, 0, sizeof(double) * (size_t)d * (size_t)hd);
    memset(dWv, 0, sizeof(double) * (size_t)d * (size_t)hd);

    for (int i = 0; i < seqLen; i++)
    {
        for (int e = 0; e < d; e++)
        {
            double x = X[i * d + e];
            for (int k = 0; k < hd; k++)
            {
                dWq[e * hd + k] += x * dQ[i * hd + k];
                dWk[e * hd + k] += x * dK[i * hd + k];
                dWv[e * hd + k] += x * dV[i * hd + k];
            }
        }
    }

    // ...then dX, using the (pre-update) Wq/Wk/Wv.
    for (int i = 0; i < seqLen; i++)
    {
        for (int e = 0; e < d; e++)
        {
            double sum = 0.0;
            for (int k = 0; k < hd; k++)
            {
                sum += dQ[i * hd + k] * attn->Wq[e * hd + k];
                sum += dK[i * hd + k] * attn->Wk[e * hd + k];
                sum += dV[i * hd + k] * attn->Wv[e * hd + k];
            }
            dX[i * d + e] = sum;
        }
    }

    // SGD step on the projection weights.
    for (int i = 0; i < d * hd; i++)
    {
        attn->Wq[i] -= lr * dWq[i];
        attn->Wk[i] -= lr * dWk[i];
        attn->Wv[i] -= lr * dWv[i];
    }

    free(dQ); free(dK); free(dV);
    free(dAlpha); free(dScore);
    free(dWq); free(dWk); free(dWv);
}

void freeAttentionLayer(AttentionLayer* attn)
{
    free(attn->Wq); free(attn->Wk); free(attn->Wv);
    free(attn->Q); free(attn->K); free(attn->V);
    free(attn->alpha); free(attn->out);
}