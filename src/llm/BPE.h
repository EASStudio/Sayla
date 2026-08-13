#pragma once
 
// Core includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_VOCAB_SIZE 50257
#define MAX_PAIR_TRACK 20000
#define INF 99999999

// BPE murge rule
typedef struct 
{
    int tokenA;
    int tokenB;
}MergeRule;

// Tokenizer state 
typedef struct 
{
    int vocabSize;
    char* vocabTable[MAX_VOCAB_SIZE];
    MergeRule merges[MAX_VOCAB_SIZE];
    int numMerges;
}Tokenizer;

// Node for a dynamic linked list representing the text sequence during merging
typedef struct TokenNode 
{
    int id;
    struct TokenNode* next;
    struct TokenNode* prev;
} TokenNode;

// One (pair, frequency) observation while scanning the corpus for merge candidates
typedef struct
{
    int a;
    int b;
    int count;
} PairCount;

void initTokenizer(Tokenizer* t);
void trainBPE(Tokenizer* t, const char* corpus, int targetVocabSize);
int getMergeRank(Tokenizer* t, int idA, int idB);
int tokenizeEncode(Tokenizer* t, const char* text, int* ouputTokens);
int tokenizeDecode(Tokenizer* t, const int* tokens, int tokenCount, char* ouput, int maxLen);
void freeTokenizer(Tokenizer* t);