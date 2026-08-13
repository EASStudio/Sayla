#pragma once

typedef struct
{
    int vocabSize;
    int embeddingDim;
    double** vectors; // vectors[tokenId] is a length-embeddingDim vector
} EmbeddingTable;

void initEmbeddingTable(EmbeddingTable* e, int vocabSize, int embeddingDim);
const double* getEmbedding(EmbeddingTable* e, int tokenId);
void updateEmbedding(EmbeddingTable* e, int tokenId, const double* gradient, double lr);
void freeEmbeddingTable(EmbeddingTable* e);