// Module includes
#include "engine.h"
#include "BPE.h"
#include "network.h"
#include "embedding.h"
#include "attention.h"
#include "cache.h"

// Core includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <ctype.h>

#ifdef _WIN32
  #include <windows.h>
  #include <process.h>  

  #define PTHREAD_MUTEX_INITIALIZER SRWLOCK_INIT
  #define pthread_mutex_t SRWLOCK
  #define pthread_mutex_init(mtx, attr) (InitializeSRWLock(mtx), 0)
  #define pthread_mutex_destroy(mtx) ((void)0) // SRWLOCK needs no cleanup call
  #define pthread_mutex_lock(mtx) (AcquireSRWLockExclusive(mtx), 0)
  #define pthread_mutex_unlock(mtx) ReleaseSRWLockExclusive(mtx)

  #define pthread_t HANDLE

  #define pthread_create(thread, attr, start_routine, arg) \
    (((*(thread) = (HANDLE)_beginthreadex( \
        NULL, 0, \
        (unsigned (__stdcall *)(void *))(start_routine), \
        (arg), 0, NULL)) == (HANDLE)-1) ? 1 : 0)
        
   #define pthread_join(thread, retval) \
    ((WaitForSingleObject((thread), INFINITE) == WAIT_OBJECT_0) ? \
    (CloseHandle(thread), 0) : -1)
#endif

#ifndef _WIN32
  #include <pthread.h>
#endif

// ---- Model shape -----------------------------------------------------
// Three independent domain models (math, physics, coding), each built the
// same way the earlier single-domain version was: token + positional
// embeddings over a fixed CONTEXT_LEN window, a single causal
// self-attention head, then a small softmax MLP (see layer.c/network.c).
// Each domain trains on its own specialized corpus *concatenated with*
// the shared base (conversational/explanatory) corpus, so every domain
// keeps general chat and "explain what this is" ability alongside its
// specialty, without diluting any one domain's vocabulary with the other
// two's.
//
// At generation time, a lightweight keyword-based classifier looks at the
// prompt and picks which of the three trained models actually answers
// (see classifyPromptDomain). This is a coarse heuristic, not a learned
// decision -- a real router would be its own small classifier network,
// which is future work, not something worth building before the domains
// even have enough real data to clearly distinguish.

// Fallback seed text, used per-file only if that specific corpus file
// doesn't exist on disk yet -- same pattern the original single-corpus
// SEED_CORPUS used, just split per domain now. These are NOT meant to be
// real training data -- they exist so the app still runs if a corpus file
// is missing. Replace corpus_base.txt / corpus_math.txt /
// corpus_physics.txt / corpus_coding.txt with real content.

static const char* BASE_SEED_CORPUS =
    "hello how can i help you today "
    "i can explain what something is or how it works "
    "i am still learning so my answers are limited right now "
    "thank you for trying me out ";

static const char* MATH_SEED_CORPUS =
    "two plus two equals four "
    "the derivative of x squared is two x "
    "a prime number is only divisible by one and itself ";

static const char* PHYSICS_SEED_CORPUS =
    "force equals mass times acceleration "
    "energy cannot be created or destroyed only transformed "
    "velocity is the rate of change of position over time ";

static const char* CODING_SEED_CORPUS =
    "a function takes inputs and returns an output "
    "a loop repeats a block of code until a condition is met "
    "a variable stores a value that can change while a program runs ";

static const char* domainNames[NUM_DOMAINS]      = { "math", "physics", "coding" };
static const char* domainCorpusPaths[NUM_DOMAINS] = { CORPUS_MATH_PATH, CORPUS_PHYSICS_PATH, CORPUS_CODING_PATH };
static const char* domainCachePaths[NUM_DOMAINS]  = { MODEL_CACHE_MATH_PATH, MODEL_CACHE_PHYSICS_PATH, MODEL_CACHE_CODING_PATH }; 

static int activeModelIndex = 0;

int getModelCount(void) { return NUM_REGISTERED_MODELS; }

const ModelConfig* getModelConfigAt(int index)
{
    if (index < 0 || index >= NUM_REGISTERED_MODELS) index = 0;
    return &MODEL_REGISTRY[index];
}

int getActiveModelIndex(void) { return activeModelIndex; }

static const ModelConfig* activeModel(void) { return &MODEL_REGISTRY[activeModelIndex]; }

// Builds the cache file path for domain d under the given model: the
// legacy unnamespaced path (model_cache_math.bin) when cacheTag is "",
// otherwise a namespaced one (model_cache_math_<tag>.bin) so different
// models' trained weights never collide with each other or with
// Oriole 1's existing files. Writes into a caller-provided buffer
// rather than returning a pointer, since the result is built fresh
// each call (unlike domainCachePaths[], which are compile-time string
// literals).
static void buildCachePath(char* out, size_t outSize, Domain d, const ModelConfig* model)
{
    if (model->cacheTag[0] == '\0')
        snprintf(out, outSize, "%s", domainCachePaths[d]);
    else
        snprintf(out, outSize, "model_cache_%s_%s.bin", domainNames[d], model->cacheTag);
}

static const char* domainSeed(Domain d)
{
    switch (d)
    {
        case DOMAIN_MATH:    return MATH_SEED_CORPUS;
        case DOMAIN_PHYSICS: return PHYSICS_SEED_CORPUS;
        case DOMAIN_CODING:  return CODING_SEED_CORPUS;
        default:             return "";
    }
}

// Reads `path` into a malloc'd, null-terminated buffer. Falls back to a
// copy of `seed` if the file doesn't exist or is empty.
static char* loadFileOrSeed(const char* path, const char* seed)
{
    FILE* f = fopen(path, "rb");
    if (!f)
    {
        char* copy = (char*)malloc(strlen(seed) + 1);
        strcpy(copy, seed);
        return copy;
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (size <= 0)
    {
        fclose(f);
        char* copy = (char*)malloc(strlen(seed) + 1);
        strcpy(copy, seed);
        return copy;
    }

    char* buf = (char*)malloc((size_t)size + 1);
    size_t n = fread(buf, 1, (size_t)size, f);
    buf[n] = '\0';
    fclose(f);

    return buf;
}

static Tokenizer tokenizers[NUM_DOMAINS];
static EmbeddingTable embeddings[NUM_DOMAINS];      // per-token embeddings, per domain
static EmbeddingTable posEmbeddings[NUM_DOMAINS];   // per-slot positional embeddings, per domain
static AttentionLayer attns[NUM_DOMAINS];
static Network networks[NUM_DOMAINS];
static int engineReady = 0;

// ---- Training progress (for the UI's training screen) -------------------
// `progress` is written from the background training thread (see
// startEngineInitAsync) and read from the main/render thread once per
// frame, so every access goes through the mutex -- never read/write the
// struct directly outside updateProgress()/getTrainingProgress().
static TrainingProgress progress = { false, false, "", 0, 0, 0.0 };
static pthread_mutex_t progressMutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_t engineThread;
static bool usedAsyncInit = false;

static void updateProgress(bool isTraining, const char* domainName, long step, long totalSteps, double lastLoss)
{
    pthread_mutex_lock(&progressMutex);
    progress.isTraining = isTraining;
    if (domainName != NULL)
    {
        strncpy(progress.domainName, domainName, sizeof(progress.domainName) - 1);
        progress.domainName[sizeof(progress.domainName) - 1] = '\0';
    }
    progress.step = step;
    progress.totalSteps = totalSteps;
    progress.lastLoss = lastLoss;
    pthread_mutex_unlock(&progressMutex);
}

static void markInitDone(void)
{
    pthread_mutex_lock(&progressMutex);
    progress.isDone = true;
    progress.isTraining = false;
    pthread_mutex_unlock(&progressMutex);
}

TrainingProgress getTrainingProgress(void)
{
    pthread_mutex_lock(&progressMutex);
    TrainingProgress snapshot = progress;
    pthread_mutex_unlock(&progressMutex);
    return snapshot;
}

// Builds X (CONTEXT_LEN x EMBEDDING_DIM, row-major) for domain d: token
// embedding plus positional embedding for each of the CONTEXT_LEN context
// token ids, in order. Positions here are local slots in this fixed
// sliding window (0..CONTEXT_LEN-1), not the token's absolute offset in
// the corpus.
static void buildContextEmbeddings(Domain d, const int* context, double* X)
{
    int contextLen = activeModel()->contextLen;
    int embeddingDim = activeModel()->embeddingDim;
    for (int c = 0; c < contextLen; c++)
    {
        const double* tokVec = getEmbedding(&embeddings[d], context[c]);
        const double* posVec = getEmbedding(&posEmbeddings[d], c);
        for (int e = 0; e < embeddingDim; e++)
            X[c * embeddingDim + e] = tokVec[e] + posVec[e];
    }
}

// Trains domain d on every (context -> next token) window in its
// tokenized corpus, for the given number of passes over the data. Same
// logic as the original single-domain trainOnCorpus, indexed by domain.
static void trainOnCorpus(Domain d, const char* corpus, int epochs)
{
    const ModelConfig* model = activeModel();
    int contextLen = model->contextLen;
    int embeddingDim = model->embeddingDim;
    int headDim = model->headDim;
    double learningRate = model->learningRate;

    int corpusLen = (int)strlen(corpus);
    int* tokens = (int*)malloc(sizeof(int) * (corpusLen + 1));
    int tokenCount = tokenizeEncode(&tokenizers[d], corpus, tokens);

    if (tokenCount <= contextLen)
    {
        free(tokens);
        return; // not enough data to form even one training window
    }

    int vocabSize = tokenizers[d].vocabSize;
    double* target = (double*)malloc(sizeof(double) * vocabSize);

    // Heap-allocated, sized from the active model's config, rather than
    // fixed-size stack arrays keyed to what used to be global constants
    // -- allocated once here (outside the training loop below), not
    // per-step, since these are reused every iteration.
    double* X = (double*)malloc(sizeof(double) * (size_t)contextLen * (size_t)embeddingDim); // context window's token+pos embeddings
    double* input = (double*)malloc(sizeof(double) * (size_t)headDim); // attention output at the last position -> network input
    double* inputGrad = (double*)malloc(sizeof(double) * (size_t)headDim);
    double* dOut = (double*)malloc(sizeof(double) * (size_t)contextLen * (size_t)headDim);
    double* dX = (double*)malloc(sizeof(double) * (size_t)contextLen * (size_t)embeddingDim);
    double* embGrad = (double*)malloc(sizeof(double) * (size_t)embeddingDim); // moved out of the innermost loop below -- was a per-iteration stack array before, now allocated once and reused
    size_t dOutBytes = sizeof(double) * (size_t)contextLen * (size_t)headDim; // explicit byte count for memset below -- sizeof(dOut) would only give the pointer's own size (8 bytes), not the allocated buffer's, now that this is heap-allocated instead of a stack array

    double runningLoss = 0.0;
    long stepsSinceReport = 0;
    long globalStep = 0;
    long totalSteps = (long)epochs * (tokenCount - contextLen);

    updateProgress(true, domainNames[d], 0, totalSteps, 0.0);

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        for (int pos = contextLen; pos < tokenCount; pos++)
        {
            const int* context = &tokens[pos - contextLen];

            buildContextEmbeddings(d, context, X);
            double* attnOut = attentionForward(&attns[d], X, contextLen);
            memcpy(input, attnOut + (contextLen - 1) * headDim, sizeof(double) * (size_t)headDim);

            // One-hot target: the token that actually came next
            for (int v = 0; v < vocabSize; v++) target[v] = 0.0;
            int actualNext = tokens[pos];
            if (actualNext >= 0 && actualNext < vocabSize) target[actualNext] = 1.0;

            trainNetwork(&networks[d], input, target);

            double predictedProb = networks[d].layers[networks[d].numLayers - 1].outputs[actualNext];
            if (predictedProb < 1e-12) predictedProb = 1e-12; // guard log(0)
            runningLoss += -log(predictedProb);
            stepsSinceReport++;
            globalStep++;

            // Cheap relative to the training step itself (a mutex lock
            // plus a few field writes vs. the attention+network forward/
            // backward pass above), so updating every step keeps the
            // training screen's progress smooth rather than jumping in
            // LOSS_REPORT_INTERVAL-sized chunks.
            updateProgress(true, domainNames[d], globalStep, totalSteps,
                            runningLoss / (double)stepsSinceReport);

            if (stepsSinceReport >= LOSS_REPORT_INTERVAL)
            {
                printf("training [%s]: step %ld/%ld (epoch %d/%d) | avg loss (last %ld) = %.4f\n",
                       domainNames[d], globalStep, totalSteps, epoch + 1, epochs,
                       stepsSinceReport, runningLoss / (double)stepsSinceReport);
                fflush(stdout);
                runningLoss = 0.0;
                stepsSinceReport = 0;
            }

            // Backprop one more step into the attention output -- same
            // one-step-stale approximation as before (reads layer 0's
            // weights after trainNetwork already updated them).
            Layer* firstLayer = &networks[d].layers[0];
            for (int i = 0; i < headDim; i++)
            {
                double sum = 0.0;
                for (int k = 0; k < firstLayer->numNeurons; k++)
                    sum += firstLayer->deltas[k] * firstLayer->neurons[k].weights[i];
                inputGrad[i] = sum;
            }

            memset(dOut, 0, dOutBytes);
            for (int k = 0; k < headDim; k++)
                dOut[(contextLen - 1) * headDim + k] = -inputGrad[k];

            attentionBackward(&attns[d], X, dOut, contextLen, dX, learningRate);

            for (int c = 0; c < contextLen; c++)
            {
                for (int e = 0; e < embeddingDim; e++)
                    embGrad[e] = -dX[c * embeddingDim + e];

                updateEmbedding(&embeddings[d], context[c], embGrad, learningRate);
                updateEmbedding(&posEmbeddings[d], c, embGrad, learningRate);
            }
        }
    }

    free(X);
    free(input);
    free(inputGrad);
    free(dOut);
    free(dX);
    free(embGrad);

    if (stepsSinceReport > 0)
    {
        printf("training [%s]: step %ld/%ld (final) | avg loss (last %ld) = %.4f\n",
               domainNames[d], globalStep, totalSteps, stepsSinceReport, runningLoss / (double)stepsSinceReport);
    }

    free(target);
    free(tokens);
}

// The core "load each domain from cache, or train it fresh" sequence,
// parameterized entirely by the currently active model (see
// activeModel()) rather than the removed global constants. Extracted
// out of initEngine() so the same sequence can also run when switching
// which model is active at runtime (see setActiveModelIndex()) --
// switching to an uncached model needs to train it exactly the same
// way the very first startup does.
static void loadOrTrainAllDomains(void)
{
    const ModelConfig* model = activeModel();

    // Explicit UI stage for the shared base corpus, before any domain
    // starts -- without this, the training screen jumps straight to
    // "Training math..." (or "Preparing math...") with no feedback at
    // all during the (usually brief, but not necessarily instant if the
    // base corpus is large) read of corpus_base.txt. isTraining stays
    // false here since this isn't a training step, so window.c's
    // existing "Preparing %s..." fallback branch renders it correctly
    // with no changes needed there.
    updateProgress(false, "base", 0, 0, 0.0);
    char* base = loadFileOrSeed(CORPUS_BASE_PATH, BASE_SEED_CORPUS);

    for (int i = 0; i < NUM_DOMAINS; i++)
    {
        Domain d = (Domain)i;
        char* domainText = loadFileOrSeed(domainCorpusPaths[d], domainSeed(d));

        // What this domain actually trains on: shared base conversational
        // text plus this domain's specialized text, so it keeps general
        // chat/explanation ability alongside its specialty.
        //
        // The domain text is repeated enough times that it isn't
        // completely outweighed by the base corpus -- this addresses a
        // real, measured problem, not a hypothetical one: math's corpus
        // is ~3.6KB against a ~25.6KB base corpus (roughly 1:7), and
        // testing confirmed base-corpus phrasing bleeding into math
        // answers as a direct result (a math prompt's output contained
        // "something built to serve one clear, specific purpose" --
        // straight out of the base corpus's generic-definition style,
        // not math content, and it persisted even after a 50% increase
        // in training epochs, which is what ruled epochs out as the fix
        // in the first place). The repeat count is computed from actual
        // corpus sizes at runtime, not a fixed constant, so a domain
        // whose own corpus already matches or exceeds the base corpus
        // (coding, currently ~285KB) gets repeatCount=1 -- no change at
        // all, since it doesn't need balancing.
        size_t baseLen = strlen(base);
        size_t domainTextLen = strlen(domainText);
        int repeatCount = 1;
        if (domainTextLen > 0 && baseLen / domainTextLen > 1)
        {
            size_t computed = baseLen / domainTextLen;
            repeatCount = (computed > 10) ? 10 : (int)computed; // sanity cap against a pathologically tiny domain corpus
        }

        size_t combinedLen = baseLen + (size_t)repeatCount * (domainTextLen + 1);
        char* combined = (char*)malloc(combinedLen + 1);
        memcpy(combined, base, baseLen);
        size_t pos = baseLen;
        for (int r = 0; r < repeatCount; r++)
        {
            combined[pos++] = '\n';
            memcpy(combined + pos, domainText, domainTextLen);
            pos += domainTextLen;
        }
        combined[pos] = '\0';
        free(domainText);

        char cachePath[256];
        buildCachePath(cachePath, sizeof(cachePath), d, model);

        // attn's scratch buffers (Q/K/V/alpha/out) are fixed-size and
        // independent of the corpus/vocab, and never persisted to the
        // cache -- only Wq/Wk/Wv are. Allocate them up front so a cache
        // hit below can overwrite Wq/Wk/Wv in place.
        initAttentionLayer(&attns[d], model->contextLen, model->embeddingDim, model->headDim);

        if (loadModelCache(cachePath, combined, model->contextLen, model->hiddenSize,
                            &tokenizers[d], &embeddings[d], &posEmbeddings[d], &attns[d], &networks[d]))
        {
            printf("engine: [%s] loaded from cache\n", domainNames[d]);
            updateProgress(false, domainNames[d], 0, 0, 0.0);
            free(combined);
            continue;
        }

        printf("engine: [%s] no cache hit, training now (this can take a while)...\n", domainNames[d]);
        fflush(stdout);

        trainBPE(&tokenizers[d], combined, 1000);
        initEmbeddingTable(&embeddings[d], tokenizers[d].vocabSize, model->embeddingDim);
        initEmbeddingTable(&posEmbeddings[d], model->contextLen, model->embeddingDim);

        int sizes[] = { model->headDim, model->hiddenSize, tokenizers[d].vocabSize };
        initNetwork(&networks[d], sizes, 2, model->learningRate);

        // Blocking here is fine at this corpus size, and only happens on
        // the first run per domain, or whenever that domain's corpus (or
        // the shared base corpus) actually changes, since the result gets
        // cached below.
        trainOnCorpus(d, combined, model->trainEpochs);

        saveModelCache(cachePath, combined, &tokenizers[d], &embeddings[d], &posEmbeddings[d], &attns[d], &networks[d]);

        free(combined);
    }

    free(base);
}

void initEngine()
{
    srand((unsigned int)time(NULL));
    loadOrTrainAllDomains();
    engineReady = 1;
    markInitDone();
}

static void* engineThreadFunc(void* arg)
{
    (void)arg;
    initEngine();
    return NULL;
}

void startEngineInitAsync(void)
{
    usedAsyncInit = true;
    pthread_create(&engineThread, NULL, engineThreadFunc, NULL);
}

// Used internally by generateResponse()/countTokens() instead of the old
// "if (!engineReady) initEngine();" guard. Checking getTrainingProgress()
// (mutex-protected) rather than the raw engineReady flag avoids reading a
// value the background thread might be mid-write on; if async init is
// still running, this waits for it (via pthread_join) instead of racing
// a second concurrent initEngine() call against it. If startEngineInitAsync
// was never called, this falls back to the original synchronous behavior.
static void ensureEngineReady(void)
{
    if (getTrainingProgress().isDone) return;

    if (usedAsyncInit)
    {
        pthread_join(engineThread, NULL);
        usedAsyncInit = false; // joined; freeEngine() must not join again
        return;
    }

    initEngine();
}

// Picks a token id from the network's output distribution. The output
// layer is a real softmax (see layer.c / network.c), so `outputs` already
// sums to ~1.0 -- this still normalizes defensively against floating-point
// drift.
static int sampleFromOutputs(const double* outputs, int vocabSize)
{
    double sum = 0.0;
    for (int i = 0; i < vocabSize; i++) sum += outputs[i];

    if (sum <= 0.0) return rand() % vocabSize; // nothing came out positive; fall back to uniform

    double r = ((double)rand() / (double)RAND_MAX) * sum;
    double acc = 0.0;
    for (int i = 0; i < vocabSize; i++)
    {
        acc += outputs[i];
        if (r <= acc) return i;
    }

    return vocabSize - 1;
}

// ---- Domain classification --------------------------------------------
// A coarse, easily-tunable keyword classifier -- not a learned decision.
// Lower-cases the prompt and counts substring hits against each domain's
// keyword list; picks whichever domain scores highest. Ties (including
// the common "no keywords matched at all" case, e.g. plain small talk)
// default to coding. Every domain's model also carries the base corpus,
// so any of the three can handle general conversation reasonably
// regardless of which one actually answers.
// Expanded after reviewing corpus_math.txt/corpus_physics.txt topic by
// topic against these lists -- "absolute value" and "order of
// operations" were the reported gap, but the same review found several
// more: corpus_math.txt covers circle area, factorials, parallel
// lines, mean/median, greatest common divisor, slope, and perfect
// squares with no matching keyword anywhere; corpus_physics.txt
// likewise covers weight, speed, thermal equilibrium, scalars, tides,
// and pressure uncovered. Each addition below was checked for the same
// substring false-positive risk that caused the "yo"/"your" and
// "hi"/"this" bugs in the greeting keywords: multi-word phrases
// ("perfect square", "order of operations") are safe as substrings
// since they're too long to appear accidentally; single words were
// picked for being reasonably distinctive ("factorial", "equilibrium",
// "scalar") rather than short/common. "mean" was deliberately left out
// in favor of "median" alone -- "mean" is ordinary English vocabulary
// ("what do you mean") and would very likely repeat the exact mistake
// just fixed elsewhere; median alone still catches the corpus's actual
// mean-vs-median topic without that risk.
static const char* mathKeywords[] = {
    "equation", "solve", "integral", "derivative", "algebra", "calculus",
    "theorem", "proof", "matrix", "vector", "polynomial", "factor",
    "prime number", "square root", "sqrt", "geometry", "limit",
    "fraction", "probability", "quadratic",
    "absolute value", "order of operations", "perfect square",
    "area", "circle", "factorial", "parallel", "median", "divisor", "slope"
};
static const char* physicsKeywords[] = {
    "force", "energy", "velocity", "acceleration", "momentum", "quantum",
    "gravity", "mass", "electric", "magnetic", "wave", "particle",
    "thermodynamics", "friction", "newton", "relativity", "voltage",
    "circuit", "kinetic", "potential energy",
    "weight", "speed", "thermal", "equilibrium", "temperature", "scalar",
    "tide", "pressure"
};
// codingKeywords wasn't reviewed as exhaustively -- unlike the Q&A-pair
// math/physics corpora, corpus_coding.txt is real C source (a tokenizer,
// embedding tables, attention layers), so the existing broad programming
// terms already match throughout it naturally. Added a few concrete gaps
// checked directly against the corpus content (TokenNode is a real
// linked list, malloc calls appear throughout, most of the data
// structures are structs) rather than guessing generically.
static const char* codingKeywords[] = {
    "function", "code", "compile", "python", "c++", "variable",
    "loop", "array", "class ", "def ", "int ", "void ", "#include",
    "print(", "return", "bug", "debug", "algorithm", "program", "syntax",
    "pointer", "recursion",
    "struct", "linked list", "malloc", "string"
};

static int countKeywordHits(const char* lowerText, const char** keywords, int numKeywords)
{
    int hits = 0;
    for (int i = 0; i < numKeywords; i++)
        if (strstr(lowerText, keywords[i]) != NULL) hits++;
    return hits;
}

// ---- Fuzzy (typo-tolerant) matching --------------------------------------
// Exact substring matching above is brittle to simple typos: "Dirvative"
// doesn't contain "derivative" as a substring, so it scored 0 against
// every domain and fell through to the all-zero default (coding) even
// though the prompt was clearly a math question. This adds a second pass:
// split the prompt into words and Levenshtein-match each one against the
// single-word, no-space keywords in each list, tolerating a small edit
// distance that scales with word length. Multi-word keywords ("prime
// number") and short/space-padded ones ("int ", "def ") are skipped here
// -- too short or too easily false-matched at word level -- and are left
// to the exact substring pass above, which already handles them fine.
static int levenshtein(const char* a, int aLen, const char* b, int bLen)
{
    // 32x32 comfortably covers any real word or keyword this is ever
    // called with; longer inputs are clamped rather than risking
    // anything larger on the stack.
    static int dp[33][33];
    if (aLen > 32) aLen = 32;
    if (bLen > 32) bLen = 32;

    for (int i = 0; i <= aLen; i++) dp[i][0] = i;
    for (int j = 0; j <= bLen; j++) dp[0][j] = j;

    for (int i = 1; i <= aLen; i++)
    {
        for (int j = 1; j <= bLen; j++)
        {
            int cost = (a[i - 1] == b[j - 1]) ? 0 : 1;
            int del = dp[i - 1][j] + 1;
            int ins = dp[i][j - 1] + 1;
            int sub = dp[i - 1][j - 1] + cost;
            int best = del < ins ? del : ins;
            if (sub < best) best = sub;
            dp[i][j] = best;
        }
    }
    return dp[aLen][bLen];
}

// Verified empirically: "dirvative" vs "derivative" -> distance 2 (falls
// within the 9-letter word's threshold of 2, so it matches); "the" vs
// "theorem" -> distance 4 (and "the" is too short to attempt fuzzy
// matching at all), so short common words don't false-match.
static int fuzzyThreshold(int wordLen)
{
    if (wordLen <= 4) return 0; // too short to safely fuzzy-match at all
    if (wordLen <= 7) return 1;
    return 2;
}

static bool isFuzzyCandidate(const char* keyword)
{
    if (strlen(keyword) < 5) return false;
    return strchr(keyword, ' ') == NULL;
}

static int countFuzzyHits(const char* lowerText, const char** keywords, int numKeywords)
{
    int hits = 0;
    int len = (int)strlen(lowerText);
    int i = 0;

    while (i < len)
    {
        while (i < len && !((lowerText[i] >= 'a' && lowerText[i] <= 'z') || (lowerText[i] >= '0' && lowerText[i] <= '9')))
            i++;

        int wordStart = i;
        while (i < len && ((lowerText[i] >= 'a' && lowerText[i] <= 'z') || (lowerText[i] >= '0' && lowerText[i] <= '9')))
            i++;

        int wordLen = i - wordStart;
        if (wordLen > 0)
        {
            int threshold = fuzzyThreshold(wordLen);
            if (threshold > 0)
            {
                for (int k = 0; k < numKeywords; k++)
                {
                    if (!isFuzzyCandidate(keywords[k])) continue;

                    int kLen = (int)strlen(keywords[k]);
                    if (levenshtein(lowerText + wordStart, wordLen, keywords[k], kLen) <= threshold)
                    {
                        hits++;
                        break; // one fuzzy hit per word; move on to the next word
                    }
                }
            }
        }
    }

    return hits;
}

static Domain classifyPromptDomain(const char* text, bool* outHadDomainMatch)
{
    int len = (int)strlen(text);
    char* lower = (char*)malloc((size_t)len + 1);
    for (int i = 0; i < len; i++) lower[i] = (char)tolower((unsigned char)text[i]);
    lower[len] = '\0';

    int mathScore    = countKeywordHits(lower, mathKeywords,    sizeof(mathKeywords) / sizeof(mathKeywords[0]))
                      + countFuzzyHits(lower, mathKeywords,    sizeof(mathKeywords) / sizeof(mathKeywords[0]));
    int physicsScore = countKeywordHits(lower, physicsKeywords, sizeof(physicsKeywords) / sizeof(physicsKeywords[0]))
                      + countFuzzyHits(lower, physicsKeywords, sizeof(physicsKeywords) / sizeof(physicsKeywords[0]));
    int codingScore  = countKeywordHits(lower, codingKeywords,  sizeof(codingKeywords) / sizeof(codingKeywords[0]))
                      + countFuzzyHits(lower, codingKeywords,  sizeof(codingKeywords) / sizeof(codingKeywords[0]));

    free(lower);

    if (mathScore > physicsScore && mathScore > codingScore)
    {
        if (outHadDomainMatch) *outHadDomainMatch = true;
        return DOMAIN_MATH;
    }
    if (physicsScore > mathScore && physicsScore > codingScore)
    {
        if (outHadDomainMatch) *outHadDomainMatch = true;
        return DOMAIN_PHYSICS;
    }

    // Reaching here covers three cases: coding genuinely won outright,
    // a tie occurred (including an all-zero tie -- plain small talk
    // like "Hey what's up!" matches nothing in any list), or coding
    // won at zero because nothing else scored higher. Only the first
    // is a real classification; the other two are the same structural
    // fallback this function has always had (some domain has to
    // generate a reply). outHadDomainMatch lets a caller -- the Live
    // panel's code-preview display -- tell those apart, so a genuine
    // coding question shows the code panel and a greeting doesn't.
    if (outHadDomainMatch)
        *outHadDomainMatch = (codingScore > mathScore) && (codingScore > physicsScore);
    return DOMAIN_CODING;
}

// Generates up to maxTokens new token ids for domain d, starting from a
// context seeded with the tail of the prompt (padded with spaces if the
// prompt is shorter than CONTEXT_LEN), sliding the window forward one
// token at a time.
static int generateTokens(Domain d, const int* promptTokens, int promptCount, int* outTokens, int maxTokens)
{
    const ModelConfig* model = activeModel();
    int contextLen = model->contextLen;
    int embeddingDim = model->embeddingDim;
    int headDim = model->headDim;

    int vocabSize = tokenizers[d].vocabSize;

    int* context = (int*)malloc(sizeof(int) * (size_t)contextLen);
    for (int i = 0; i < contextLen; i++)
    {
        int srcIdx = promptCount - contextLen + i;
        context[i] = (srcIdx >= 0) ? promptTokens[srcIdx] : PAD_TOKEN_ID;
    }

    double* X = (double*)malloc(sizeof(double) * (size_t)contextLen * (size_t)embeddingDim);
    double* input = (double*)malloc(sizeof(double) * (size_t)headDim);
    int generatedCount = 0;

    for (int step = 0; step < maxTokens; step++)
    {
        buildContextEmbeddings(d, context, X);
        double* attnOut = attentionForward(&attns[d], X, contextLen);
        memcpy(input, attnOut + (contextLen - 1) * headDim, sizeof(double) * (size_t)headDim);

        double* outputs = feedForwardNetwork(&networks[d], input);
        int nextToken = sampleFromOutputs(outputs, vocabSize);

        outTokens[generatedCount++] = nextToken;

        for (int c = 0; c < contextLen - 1; c++)
            context[c] = context[c + 1];
        context[contextLen - 1] = nextToken;
    }

    free(context);
    free(X);
    free(input);

    return generatedCount;
}

// Set by generateResponse(), read by getLastResponseDomain() -- see that
// function's comment in engine.h for why this exists. Plain state, no
// mutex: only ever written/read from the calling thread (generateResponse
// already runs after ensureEngineReady() has synchronized with the
// background training thread, so training is guaranteed finished by the
// time this is touched).
static Domain lastResponseDomain = DOMAIN_CODING;
static bool lastResponseHadDomainMatch = false;

static const char* greetingWords[] = {
    "hi", "hey", "yo", "sup", "howdy", "greetings", "hello"
};
static const char* greetingPhrases[] = {
    "good morning", "good afternoon", "good evening",
    "how are you", "how's it going", "hows it going",
    "what's up", "whats up", "what is up"
};
static const char* greetingResponses[] = {
    "Hello! How can I help -- math, physics, or code?",
    "Hi there! What can I help you with today?",
    "Hey! Ask me about math, physics, or coding, happy to help.",
};
#define GREETING_RESPONSE_COUNT (int)(sizeof(greetingResponses) / sizeof(greetingResponses[0]))

// Checks whether any individual word in lowerText (already lowercased,
// split on non-alphanumeric characters) exactly equals one of the
// short, substring-risky greeting words -- not a fuzzy match, not a
// substring match, an exact whole-word match only.
static bool hasGreetingWord(const char* lowerText)
{
    int len = (int)strlen(lowerText);
    int i = 0;

    while (i < len)
    {
        while (i < len && !((lowerText[i] >= 'a' && lowerText[i] <= 'z') || (lowerText[i] >= '0' && lowerText[i] <= '9')))
            i++;

        int wordStart = i;
        while (i < len && ((lowerText[i] >= 'a' && lowerText[i] <= 'z') || (lowerText[i] >= '0' && lowerText[i] <= '9')))
            i++;

        int wordLen = i - wordStart;
        if (wordLen > 0)
        {
            for (size_t k = 0; k < sizeof(greetingWords) / sizeof(greetingWords[0]); k++)
            {
                if ((int)strlen(greetingWords[k]) == wordLen &&
                    strncmp(lowerText + wordStart, greetingWords[k], (size_t)wordLen) == 0)
                    return true;
            }
        }
    }
    return false;
}

// Only treated as a genuine greeting when classifyPromptDomain() found
// no real domain match either -- "hi, what's the derivative of x
// squared" should still get a real math answer, not a canned hello,
// just because it happens to start with a greeting word.
static bool isGreeting(const char* text)
{
    int len = (int)strlen(text);
    char* lower = (char*)malloc((size_t)len + 1);
    for (int i = 0; i < len; i++) lower[i] = (char)tolower((unsigned char)text[i]);
    lower[len] = '\0';

    bool result = hasGreetingWord(lower)
                || countKeywordHits(lower, greetingPhrases, sizeof(greetingPhrases) / sizeof(greetingPhrases[0])) > 0
                || countFuzzyHits(lower, greetingPhrases, sizeof(greetingPhrases) / sizeof(greetingPhrases[0])) > 0;
    free(lower);
    return result;
}

// Cuts outputBuffer at the first literal "User:" or "Assistant:" label,
// if either appears. These strings show up in generated output because
// the training corpus itself is formatted as "User: ...\nAssistant:
// ...\n\n" pairs, so the tiny model has learned them as ordinary
// vocabulary -- with only 16 tokens of context, it can lose track of
// whose turn it's supposedly on partway through generating and drift
// into hallucinating a whole fake follow-up exchange rather than
// stopping after answering. Real, observed example: a math question's
// output ran on past a genuine (if garbled) attempt at an answer into
// a fabricated "User: What is..." / "User: What is an..." continuation
// that was never actually asked. Everything from the first such label
// onward is that hallucinated continuation, not part of the real
// answer, so it's cut rather than shown or spoken.
static void truncateAtRoleLabel(char* text, int maxLen)
{
    char* userMarker = strstr(text, "User:");
    char* assistantMarker = strstr(text, "Assistant:");

    char* cutPoint = NULL;
    if (userMarker && assistantMarker)
        cutPoint = (userMarker < assistantMarker) ? userMarker : assistantMarker;
    else if (userMarker)
        cutPoint = userMarker;
    else if (assistantMarker)
        cutPoint = assistantMarker;

    if (cutPoint != NULL)
        *cutPoint = '\0';

    // Trim trailing whitespace/newlines left dangling by the cut (or
    // already present even without one).
    int len = (int)strlen(text);
    while (len > 0 && (text[len - 1] == ' ' || text[len - 1] == '\n' || text[len - 1] == '\t' || text[len - 1] == '\r'))
    {
        text[len - 1] = '\0';
        len--;
    }

    // If the hallucination started immediately -- nothing real came
    // before it -- don't leave an empty string; substitute something
    // honest rather than show or speak nothing at all.
    if (text[0] == '\0')
    {
        strncpy(text, "I don't have a clear answer for that.", (size_t)maxLen - 1);
        text[maxLen - 1] = '\0';
    }
}

static EffortLevel currentEffort = EFFORT_STANDARD;

void setEffortLevel(EffortLevel level) { currentEffort = level; }
EffortLevel getEffortLevel(void) { return currentEffort; }

const char* effortLevelDisplayName(EffortLevel level)
{
    switch (level)
    {
        case EFFORT_LOW:      return "Low";
        case EFFORT_HIGH:     return "High";
        case EFFORT_STANDARD:
        default:              return "Standard";
    }
}

// A real, if honestly imperfect, proxy for how garbled a generated
// response looks: the fraction of space-separated "words" that consist
// only of letters (plus basic punctuation) at a plausible length.
// Verified against actual garbled output from this project's own
// testing: it reliably scores obviously-broken tokens lower (anything
// with a stray digit or hyphen mid-word, like "dition-5ur" or
// "-erthe-0ce", fails immediately), but it is NOT a true coherence or
// dictionary check -- a nonsense-but-alphabetic fragment like "ens" or
// "valub" passes exactly as if it were a real word, since checking
// actual word validity would need something like a dictionary, well
// beyond what's reasonable to build here. Directionally correct
// (real testing showed clean, coherent sentences reliably score higher
// than garbled output), not a precise judge of quality.
static float scoreResponseQuality(const char* text)
{
    int totalWords = 0;
    int cleanWords = 0;
    int len = (int)strlen(text);
    int wordStart = -1;

    for (int i = 0; i <= len; i++)
    {
        bool isSpace = (i == len) || text[i] == ' ' || text[i] == '\n' || text[i] == '\t';
        if (!isSpace && wordStart < 0) wordStart = i;
        if (isSpace && wordStart >= 0)
        {
            int wordLen = i - wordStart;
            totalWords++;

            bool clean = true;
            for (int j = wordStart; j < i; j++)
            {
                char c = text[j];
                bool okChar = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                              c == '\'' || c == ',' || c == '.' || c == '!' || c == '?' || c == ';' || c == ':';
                if (!okChar) { clean = false; break; }
            }
            if (clean && wordLen >= 2 && wordLen <= 15) cleanWords++;

            wordStart = -1;
        }
    }

    if (totalWords == 0) return 0.0f;
    return (float)cleanWords / (float)totalWords;
}

// Low effort trades quality for speed (fewer tokens, one attempt);
// high effort spends real additional compute (three full generation
// passes, each independently stochastic -- see sampleFromOutputs())
// for a genuine, if imperfect, chance at a cleaner result via
// scoreResponseQuality() below. Standard is the original, unchanged
// single-attempt behavior.
static void getEffortParams(EffortLevel level, int* outMaxTokens, int* outNumAttempts)
{
    switch (level)
    {
        case EFFORT_LOW:
            *outMaxTokens = 30;
            *outNumAttempts = 1;
            break;
        case EFFORT_HIGH:
            *outMaxTokens = MAX_GENERATED_TOKENS;
            *outNumAttempts = 3;
            break;
        case EFFORT_STANDARD:
        default:
            *outMaxTokens = MAX_GENERATED_TOKENS;
            *outNumAttempts = 1;
            break;
    }
}

void generateResponse(const char* userText, char* outputBuffer, int maxLen)
{
    ensureEngineReady();

    bool hadMatch = false;
    Domain d = classifyPromptDomain(userText, &hadMatch);

    if (!hadMatch && isGreeting(userText))
    {
        static int greetingRotation = 0;
        const char* reply = greetingResponses[greetingRotation % GREETING_RESPONSE_COUNT];
        greetingRotation++;

        strncpy(outputBuffer, reply, (size_t)maxLen - 1);
        outputBuffer[maxLen - 1] = '\0';

        lastResponseDomain = d;
        lastResponseHadDomainMatch = false;
        return;
    }

    // Not a greeting either -- genuinely outside math/physics/coding.
    // Say so honestly instead of running the tiny model anyway: with no
    // real keyword match, classifyPromptDomain()'s documented fallback
    // still returns a domain (coding, currently), and generating from
    // that domain's model for a totally unrelated prompt produces
    // output that's closer to noise than an answer -- the kind of
    // garbled, code-flavored text already documented elsewhere in this
    // file. This used to only affect what got *spoken* (a patch in
    // live_view.c's TTS selection); fixing it here instead means the
    // chat history, the on-screen text, and speech are all consistent,
    // including for typed Chat-panel input, which never went through
    // that TTS-only patch at all.
    if (!hadMatch)
    {
        static const char* outOfRangeResponses[] = {
            "I'm not sure about that one -- try asking me about math, physics, or code.",
            "I don't have a good answer for that yet -- I'm best with math, physics, and code questions.",
            "That's outside what I can answer well right now -- ask me about math, physics, or coding.",
        };
        static int outOfRangeRotation = 0;
        const char* reply = outOfRangeResponses[outOfRangeRotation % 3];
        outOfRangeRotation++;

        strncpy(outputBuffer, reply, (size_t)maxLen - 1);
        outputBuffer[maxLen - 1] = '\0';

        lastResponseDomain = d;
        lastResponseHadDomainMatch = false;
        return;
    }

    lastResponseDomain = d;
    lastResponseHadDomainMatch = hadMatch;

    int len = (int)strlen(userText);
    int* promptTokens = (int*)malloc(sizeof(int) * (len + 1));
    int promptCount = tokenizeEncode(&tokenizers[d], userText, promptTokens);

    int maxTokensThisAttempt, numAttempts;
    getEffortParams(currentEffort, &maxTokensThisAttempt, &numAttempts);
    if (maxTokensThisAttempt > MAX_GENERATED_TOKENS) maxTokensThisAttempt = MAX_GENERATED_TOKENS; // safety cap, matches generated[]'s fixed size below

    char* candidate = (char*)malloc((size_t)maxLen);
    float bestScore = -1.0f;

    for (int attempt = 0; attempt < numAttempts; attempt++)
    {
        int generated[MAX_GENERATED_TOKENS];
        int generatedCount = generateTokens(d, promptTokens, promptCount, generated, maxTokensThisAttempt);

        tokenizeDecode(&tokenizers[d], generated, generatedCount, candidate, maxLen);
        truncateAtRoleLabel(candidate, maxLen);

        float score = scoreResponseQuality(candidate);
        if (attempt == 0 || score > bestScore)
        {
            bestScore = score;
            strncpy(outputBuffer, candidate, (size_t)maxLen - 1);
            outputBuffer[maxLen - 1] = '\0';
        }
    }

    free(candidate);
    free(promptTokens);
}

Domain getLastResponseDomain(void)
{
    return lastResponseDomain;
}

// True only when the last response's domain was a genuine keyword
// match (math or physics winning outright, or coding winning outright
// -- not a tie, not the all-zero fallback). The Live panel's code
// preview uses this to avoid showing up for plain small talk that
// happened to land on coding by default.
bool getLastResponseHadDomainMatch(void)
{
    return lastResponseHadDomainMatch;
}

const char* domainDisplayName(Domain d)
{
    return domainNames[d];
}

int countTokens(const char* text)
{
    ensureEngineReady();

    int len = (int)strlen(text);
    if (len == 0) return 0;

    Domain d = classifyPromptDomain(text, NULL);

    int* tokens = (int*)malloc(sizeof(int) * (len + 1));
    if (!tokens) return -1;

    int count = tokenizeEncode(&tokenizers[d], text, tokens);
    free(tokens);
    return count;
}

// Frees the currently-loaded per-domain state (tokenizers, embeddings,
// attention layers, networks) for all domains -- shared by freeEngine()
// (app shutdown) and setActiveModelIndex() (switching to a different
// model while the app keeps running), so a model switch doesn't leak
// the previously-active model's memory.
static void freeAllDomainState(void)
{
    for (int i = 0; i < NUM_DOMAINS; i++)
    {
        freeTokenizer(&tokenizers[i]);
        freeNetwork(&networks[i]);
        freeEmbeddingTable(&embeddings[i]);
        freeEmbeddingTable(&posEmbeddings[i]);
        freeAttentionLayer(&attns[i]);
    }
}

void freeEngine()
{
    if (usedAsyncInit)
    {
        // Block until training actually finishes rather than freeing
        // shared model state out from under a still-running background
        // thread. If the user closes the window mid-training, this means
        // shutdown waits for the current domain to finish -- abrupt
        // cancellation mid-training isn't implemented, since the training
        // loop doesn't currently check for a stop signal.
        pthread_join(engineThread, NULL);
        usedAsyncInit = false;
    }

    if (engineReady)
    {
        freeAllDomainState();
        engineReady = 0;
    }
}

// Switches which registered model is active -- see engine.h's doc
// comment for the full contract. Blocking: frees the previously-active
// model's state, then runs the same load-or-train sequence initEngine()
// runs on first startup, just for the newly-selected model. A no-op if
// the requested model is already the active, loaded one, so repeated
// clicks on the same "current" model in the UI don't needlessly
// retrain or reload anything.
void setActiveModelIndex(int index)
{
    if (index < 0 || index >= NUM_REGISTERED_MODELS) return;
    if (index == activeModelIndex && engineReady) return;

    if (engineReady)
    {
        freeAllDomainState();
        engineReady = 0;
    }

    activeModelIndex = index;

    // Re-arms the training screen if the newly-selected model needs
    // training (a cache miss) -- markInitDone() below only flips this
    // back once loadOrTrainAllDomains() actually finishes, the same way
    // it does on the very first startup.
    pthread_mutex_lock(&progressMutex);
    progress.isDone = false;
    pthread_mutex_unlock(&progressMutex);

    loadOrTrainAllDomains();
    engineReady = 1;
    markInitDone();
}