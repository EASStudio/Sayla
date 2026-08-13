// Module includes
#include "cache.h"

// Core includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Every int and double value in the cache file goes through
// writePortableInt32/writePortableDouble (and their read counterparts)
// below rather than a raw fwrite of the in-memory value. A raw fwrite
// writes whatever byte order and bit layout the host CPU happens to use
// internally -- on every mainstream platform this project targets
// (Windows, macOS Intel/Apple Silicon, Linux x86/ARM) that happens to
// be little-endian IEEE-754, so it worked, but only by coincidence: the
// file format itself didn't actually define a byte order, it just
// inherited whatever the machine that wrote it used. These helpers make
// the on-disk format explicitly little-endian by construction --
// assembling/disassembling each value byte-by-byte via shifts and masks
// rather than trusting the compiler's in-memory representation -- so a
// cache file is byte-for-byte reproducible and loads correctly
// regardless of which platform wrote or reads it.
//
// Raw byte buffers (corpus text, vocabulary strings) do NOT go through
// these -- a sequence of bytes has no endianness to begin with, only
// multi-byte numeric values do.

static bool writeAll(FILE* f, const void* data, size_t size)
{
    return fwrite(data, 1, size, f) == size;
}

static bool readAll(FILE* f, void* data, size_t size)
{
    return fread(data, 1, size, f) == size;
}

static bool writePortableInt32(FILE* f, int32_t value)
{
    uint32_t u = (uint32_t)value;
    uint8_t bytes[4] = {
        (uint8_t)(u & 0xFF),
        (uint8_t)((u >> 8) & 0xFF),
        (uint8_t)((u >> 16) & 0xFF),
        (uint8_t)((u >> 24) & 0xFF)
    };
    return writeAll(f, bytes, 4);
}

static bool readPortableInt32(FILE* f, int32_t* value)
{
    uint8_t bytes[4];
    if (!readAll(f, bytes, 4)) return false;
    uint32_t u = (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) |
                 ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
    *value = (int32_t)u;
    return true;
}

static bool writePortableDouble(FILE* f, double value)
{
    // memcpy, not a direct cast/reinterpret -- this reads the double's
    // raw IEEE-754 bit pattern into a same-sized integer without
    // violating C's strict aliasing rules (which a `*(uint64_t*)&value`
    // punning cast technically would).
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));

    uint8_t bytes[8];
    for (int i = 0; i < 8; i++)
        bytes[i] = (uint8_t)((bits >> (8 * i)) & 0xFF);

    return writeAll(f, bytes, 8);
}

static bool readPortableDouble(FILE* f, double* value)
{
    uint8_t bytes[8];
    if (!readAll(f, bytes, 8)) return false;

    uint64_t bits = 0;
    for (int i = 0; i < 8; i++)
        bits |= ((uint64_t)bytes[i]) << (8 * i);

    memcpy(value, &bits, sizeof(bits));
    return true;
}

// Writes n consecutive doubles from arr, each through writePortableDouble.
static bool writePortableDoubleArray(FILE* f, const double* arr, size_t n)
{
    bool ok = true;
    for (size_t i = 0; i < n && ok; i++)
        ok = ok && writePortableDouble(f, arr[i]);
    return ok;
}

static bool readPortableDoubleArray(FILE* f, double* arr, size_t n)
{
    bool ok = true;
    for (size_t i = 0; i < n && ok; i++)
        ok = ok && readPortableDouble(f, &arr[i]);
    return ok;
}

// Shared by both the token embedding table and the positional embedding
// table -- structurally identical, so one pair of helpers covers both.
static bool writeEmbeddingTable(FILE* f, EmbeddingTable* e)
{
    bool ok = true;
    ok = ok && writePortableInt32(f, e->vocabSize);
    ok = ok && writePortableInt32(f, e->embeddingDim);

    for (int i = 0; i < e->vocabSize && ok; i++)
        ok = ok && writePortableDoubleArray(f, e->vectors[i], (size_t)e->embeddingDim);

    return ok;
}

// Allocates e->vectors itself (and each row) -- caller should not
// pre-initialize the table before calling this.
static bool readEmbeddingTable(FILE* f, EmbeddingTable* e)
{
    bool ok = true;
    ok = ok && readPortableInt32(f, &e->vocabSize);
    ok = ok && readPortableInt32(f, &e->embeddingDim);
    if (!ok) return false;

    e->vectors = (double**)malloc(sizeof(double*) * (size_t)e->vocabSize);
    for (int i = 0; i < e->vocabSize && ok; i++)
    {
        e->vectors[i] = (double*)malloc(sizeof(double) * (size_t)e->embeddingDim);
        ok = ok && readPortableDoubleArray(f, e->vectors[i], (size_t)e->embeddingDim);
    }

    return ok;
}

// Only Wq/Wk/Wv are trained weights worth persisting -- Q/K/V/alpha/out
// are per-call scratch space, recomputed fresh on every forward() call.
static bool writeAttentionLayer(FILE* f, AttentionLayer* attn)
{
    bool ok = true;
    ok = ok && writePortableInt32(f, attn->embedDim);
    ok = ok && writePortableInt32(f, attn->headDim);

    size_t n = (size_t)attn->embedDim * (size_t)attn->headDim;
    ok = ok && writePortableDoubleArray(f, attn->Wq, n);
    ok = ok && writePortableDoubleArray(f, attn->Wk, n);
    ok = ok && writePortableDoubleArray(f, attn->Wv, n);

    return ok;
}

// attn must already be initialized (initAttentionLayer) so Wq/Wk/Wv are
// already allocated at the right size -- this overwrites their contents
// in place rather than allocating new arrays, and validates the stored
// dimensions match rather than trusting them blindly.
static bool readAttentionLayer(FILE* f, AttentionLayer* attn)
{
    int32_t embedDim = 0, headDim = 0;
    bool ok = true;
    ok = ok && readPortableInt32(f, &embedDim);
    ok = ok && readPortableInt32(f, &headDim);
    if (!ok || embedDim != attn->embedDim || headDim != attn->headDim) return false;

    size_t n = (size_t)attn->embedDim * (size_t)attn->headDim;
    ok = ok && readPortableDoubleArray(f, attn->Wq, n);
    ok = ok && readPortableDoubleArray(f, attn->Wk, n);
    ok = ok && readPortableDoubleArray(f, attn->Wv, n);

    return ok;
}

bool saveModelCache(const char* path, const char* corpus, Tokenizer* t, EmbeddingTable* e,
                     EmbeddingTable* posEmbeddings, AttentionLayer* attn, Network* net)
{
    FILE* f = fopen(path, "wb");
    if (!f) return false;

    int corpusLen = (int)strlen(corpus);

    bool ok = true;
    ok = ok && writePortableInt32(f, CACHE_MAGIC);
    ok = ok && writePortableInt32(f, corpusLen);
    ok = ok && writeAll(f, corpus, (size_t)corpusLen); // raw text bytes, no endianness concern

    // ---- Tokenizer ----
    ok = ok && writePortableInt32(f, t->vocabSize);
    ok = ok && writePortableInt32(f, t->numMerges);

    for (int i = 0; i < t->numMerges && ok; i++)
    {
        ok = ok && writePortableInt32(f, t->merges[i].tokenA);
        ok = ok && writePortableInt32(f, t->merges[i].tokenB);
    }

    for (int i = 0; i < t->vocabSize && ok; i++)
    {
        int len = (int)strlen(t->vocabTable[i]);
        ok = ok && writePortableInt32(f, len);
        ok = ok && writeAll(f, t->vocabTable[i], (size_t)len); // raw text bytes
    }

    // ---- Token embeddings, positional embeddings, attention ----
    ok = ok && writeEmbeddingTable(f, e);
    ok = ok && writeEmbeddingTable(f, posEmbeddings);
    ok = ok && writeAttentionLayer(f, attn);

    // ---- Network ----
    ok = ok && writePortableInt32(f, net->numLayers);

    for (int l = 0; l < net->numLayers && ok; l++)
    {
        Layer* layer = &net->layers[l];
        ok = ok && writePortableInt32(f, layer->numNeurons);
        ok = ok && writePortableInt32(f, layer->numInputs);

        double lr = (layer->numNeurons > 0) ? layer->neurons[0].learningRate : 0.0;
        ok = ok && writePortableDouble(f, lr);

        for (int n = 0; n < layer->numNeurons && ok; n++)
        {
            Neuron* neuron = &layer->neurons[n];
            ok = ok && writePortableDouble(f, neuron->bias);
            ok = ok && writePortableDoubleArray(f, neuron->weights, (size_t)layer->numInputs);
        }
    }

    fclose(f);

    if (!ok) remove(path); // don't leave a half-written cache file behind

    return ok;
}

bool loadModelCache(const char* path, const char* corpus, int expectedContextLen, int expectedHiddenSize,
                     Tokenizer* t, EmbeddingTable* e,
                     EmbeddingTable* posEmbeddings, AttentionLayer* attn, Network* net)
{
    FILE* f = fopen(path, "rb");
    if (!f) return false;

    bool ok = true;

    int32_t magic = 0;
    ok = ok && readPortableInt32(f, &magic);
    if (!ok || magic != CACHE_MAGIC) { fclose(f); return false; }

    int32_t corpusLen = 0;
    ok = ok && readPortableInt32(f, &corpusLen);
    if (!ok || corpusLen != (int32_t)strlen(corpus)) { fclose(f); return false; }

    char* storedCorpus = (char*)malloc((size_t)corpusLen + 1);
    ok = ok && readAll(f, storedCorpus, (size_t)corpusLen);
    storedCorpus[corpusLen] = '\0';
    bool corpusMatches = ok && (memcmp(storedCorpus, corpus, (size_t)corpusLen) == 0);
    free(storedCorpus);

    if (!corpusMatches) { fclose(f); return false; }

    // ---- Tokenizer ----
    ok = ok && readPortableInt32(f, &t->vocabSize);
    ok = ok && readPortableInt32(f, &t->numMerges);

    for (int i = 0; i < t->numMerges && ok; i++)
    {
        ok = ok && readPortableInt32(f, &t->merges[i].tokenA);
        ok = ok && readPortableInt32(f, &t->merges[i].tokenB);
    }

    for (int i = 0; i < t->vocabSize && ok; i++)
    {
        int32_t len = 0;
        ok = ok && readPortableInt32(f, &len);
        if (!ok) break;

        t->vocabTable[i] = (char*)malloc((size_t)len + 1);
        ok = ok && readAll(f, t->vocabTable[i], (size_t)len);
        if (ok) t->vocabTable[i][len] = '\0';
    }

    // Known limitation: if this fails partway through, whatever was
    // already malloc'd above (vocabTable entries, embedding vectors,
    // network layers below) is left allocated but unused -- the caller's
    // fallback path re-inits everything through fresh pointers rather than
    // reusing these. A leak on a rare corrupted-cache path, not a crash;
    // acceptable for a single-run desktop tool.
    if (!ok) { fclose(f); return false; }

    // ---- Token embeddings, positional embeddings, attention ----
    ok = ok && readEmbeddingTable(f, e);
    ok = ok && readEmbeddingTable(f, posEmbeddings);
    if (!ok) { fclose(f); return false; }

    // See this function's doc comment in cache.h: the positional
    // embedding table's row count is really CONTEXT_LEN in disguise, and
    // corpus-content matching alone can't catch a mismatch here -- the
    // same corpus, cached under a different CONTEXT_LEN, would otherwise
    // silently load a table sized for the OLD value while the rest of
    // the code indexes it assuming the current one.
    if (posEmbeddings->vocabSize != expectedContextLen) { fclose(f); return false; }

    ok = ok && readAttentionLayer(f, attn);
    if (!ok) { fclose(f); return false; }

    // ---- Network ----
    ok = ok && readPortableInt32(f, &net->numLayers);

    if (ok)
    {
        net->layers = (Layer*)malloc(sizeof(Layer) * (size_t)net->numLayers);

        for (int l = 0; l < net->numLayers && ok; l++)
        {
            Layer* layer = &net->layers[l];
            ok = ok && readPortableInt32(f, &layer->numNeurons);
            ok = ok && readPortableInt32(f, &layer->numInputs);
            if (!ok) break;

            // See this function's doc comment in cache.h: layer 0 is the
            // hidden layer (sizes[] in engine.c is { headDim, hiddenSize,
            // vocabSize }), and unlike the positional embedding table,
            // nothing else validates this -- the network isn't
            // pre-initialized before loading the way attn is.
            if (l == 0 && layer->numNeurons != expectedHiddenSize) { fclose(f); return false; }

            // Same rule initNetwork uses: only the final layer is softmax.
            // Deriving this from position rather than storing it keeps the
            // cache file format unchanged.
            layer->isSoftmax = (l == net->numLayers - 1);

            double lr = 0.0;
            ok = ok && readPortableDouble(f, &lr);

            layer->neurons = (Neuron*)malloc(sizeof(Neuron) * (size_t)layer->numNeurons);
            layer->outputs = (double*)malloc(sizeof(double) * (size_t)layer->numNeurons);
            layer->deltas  = (double*)malloc(sizeof(double) * (size_t)layer->numNeurons);

            for (int n = 0; n < layer->numNeurons && ok; n++)
            {
                Neuron* neuron = &layer->neurons[n];
                neuron->numInputs = layer->numInputs;
                neuron->learningRate = lr;

                ok = ok && readPortableDouble(f, &neuron->bias);
                neuron->weights = (double*)malloc(sizeof(double) * (size_t)layer->numInputs);
                ok = ok && readPortableDoubleArray(f, neuron->weights, (size_t)layer->numInputs);
            }
        }
    }

    fclose(f);
    return ok;
}