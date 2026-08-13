// Module includes
#include "embedding.h"

// Core includes
#include <stdlib.h>

// Returns a small random double in [-0.1, 0.1] -- kept small so embeddings
// start close to zero, same rationale as the weight init in neuron.c.
static double randomEmbeddingValue(void)
{
    return ((double)rand() / (double)RAND_MAX) * 0.2 - 0.1;
}

void initEmbeddingTable(EmbeddingTable* e, int vocabSize, int embeddingDim)
{
    e->vocabSize = vocabSize;
    e->embeddingDim = embeddingDim;
    e->vectors = (double**)malloc(sizeof(double*) * vocabSize);

    for (int i = 0; i < vocabSize; i++)
    {
        e->vectors[i] = (double*)malloc(sizeof(double) * embeddingDim);
        for (int d = 0; d < embeddingDim; d++)
            e->vectors[i][d] = randomEmbeddingValue();
    }
}

const double* getEmbedding(EmbeddingTable* e, int tokenId)
{
    if (tokenId < 0 || tokenId >= e->vocabSize) tokenId = 0;
    return e->vectors[tokenId];
}

void updateEmbedding(EmbeddingTable* e, int tokenId, const double* gradient, double lr)
{
    if (tokenId < 0 || tokenId >= e->vocabSize) return;

    for (int d = 0; d < e->embeddingDim; d++)
        e->vectors[tokenId][d] += lr * gradient[d];
}

void freeEmbeddingTable(EmbeddingTable* e)
{
    for (int i = 0; i < e->vocabSize; i++)
        free(e->vectors[i]);

    free(e->vectors);
    e->vectors = NULL;
    e->vocabSize = 0;
}