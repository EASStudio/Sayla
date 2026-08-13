#pragma once

// Module includes
#include "BPE.h"
#include "embedding.h"
#include "attention.h"
#include "network.h"

// Core includes
#include <stdbool.h>

#define CACHE_MAGIC 0x53415932

bool loadModelCache(const char* path, const char* corpus, int expectedContextLen, int expectedHiddenSize, Tokenizer* t, EmbeddingTable* e, EmbeddingTable* posEmbeddings, AttentionLayer* attn, Network* net);
bool saveModelCache(const char* path, const char* corpus, Tokenizer* t, EmbeddingTable* e, EmbeddingTable* posEmbeddings, AttentionLayer* attn, Network* net);