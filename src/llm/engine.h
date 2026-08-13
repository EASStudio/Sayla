#pragma once

// Core includes
#include <stdbool.h>

#define CORPUS_BASE_PATH     "corpus_base.txt"
#define CORPUS_MATH_PATH     "corpus_math.txt"
#define CORPUS_PHYSICS_PATH  "corpus_physics.txt"
#define CORPUS_CODING_PATH   "corpus_coding.txt"

#define MODEL_CACHE_MATH_PATH    "model_cache_math.bin"
#define MODEL_CACHE_PHYSICS_PATH "model_cache_physics.bin"
#define MODEL_CACHE_CODING_PATH  "model_cache_coding.bin"

#define LOSS_REPORT_INTERVAL 2000 // print average cross-entropy loss every N training steps, per domain
#define MAX_GENERATED_TOKENS 60
#define PAD_TOKEN_ID   32 // ASCII space, used to pad a context shorter than the active model's contextLen

typedef struct
{
    const char* name;      // display name, e.g. "Oriole 1"
    const char* cacheTag;  // "" for the legacy unnamespaced path, otherwise a short filesystem-safe tag
    int contextLen;
    int embeddingDim;
    int headDim;
    int hiddenSize;
    int trainEpochs;
    double learningRate;
} ModelConfig;

typedef enum
{
    DOMAIN_MATH,
    DOMAIN_PHYSICS,
    DOMAIN_CODING,
    NUM_DOMAINS
} Domain;

typedef enum
{
    EFFORT_LOW,
    EFFORT_STANDARD,
    EFFORT_HIGH
} EffortLevel;

typedef struct
{
    bool isTraining;        // true while some domain is actively training (not just loading from cache)
    bool isDone;             // true once initEngine has fully finished every domain
    char domainName[16];      // e.g. "math", "physics", "coding"
    long step;
    long totalSteps;
    double lastLoss;
} TrainingProgress;

// 1. name
// 2. cacheTag
// 3. contextLen
// 4. embeddingDim
// 5. headDim
// 6. hiddenSize
// 7. trainEpochs
// 8. learningRate
static const ModelConfig MODEL_REGISTRY[] = 
{
    { "Oriole 1", "", 32, 128, 128, 512, 10, 0.02 },
};
#define NUM_REGISTERED_MODELS (int)(sizeof(MODEL_REGISTRY) / sizeof(MODEL_REGISTRY[0]))

int getModelCount(void);
const ModelConfig* getModelConfigAt(int index);
int getActiveModelIndex(void);
void setActiveModelIndex(int index);
void setEffortLevel(EffortLevel level);
EffortLevel getEffortLevel(void);
const char* effortLevelDisplayName(EffortLevel level);
void startEngineInitAsync(void);
TrainingProgress getTrainingProgress(void);
void initEngine(void);
void generateResponse(const char* userText, char* outputBuffer, int maxLen);
int countTokens(const char* text);
void freeEngine(void);
Domain getLastResponseDomain(void);
bool getLastResponseHadDomainMatch(void);
const char* domainDisplayName(Domain d);