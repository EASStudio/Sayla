// Module includes
#include "BPE.h"

// Helpers:

// Open-addressing hash table for pair counts, used only inside
// findBestPair. 4x MAX_PAIR_TRACK keeps the load factor low (<=25%) so
// probe chains stay short even when the tracked-pair cap is nearly hit.
#define PAIR_HASH_SIZE (MAX_PAIR_TRACK * 4)

typedef struct
{
    int a, b, count;
    int used;
} PairSlot;

// 64-bit mix (splitmix64-style finalizer) so two tokens packed into one
// key spread out well across a hash table instead of clustering.
static unsigned int mixHash(int a, int b, unsigned int tableSize)
{
    unsigned long long key = ((unsigned long long)(unsigned int)a << 32) | (unsigned int)(unsigned)b;
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    key *= 0xc4ceb9fe1a85ec53ULL;
    key ^= key >> 33;
    return (unsigned int)(key % tableSize);
}

// Scans the whole linked-list sequence, tallies every adjacent pair, and
// returns the most frequent one. Returns 0 if nothing appears more than once
// (i.e. no merge is worth making).
//
// Pair counts are tracked in a hash table rather than a linear-scan array:
// with a linear array, every single token pair encountered has to be
// compared against every distinct pair seen so far just to find (or rule
// out) a match, which is O(numPairs) per token and O(numPairs x corpus
// length) for the whole scan -- fine on a small corpus, but it dominates
// runtime badly once the corpus and vocabulary grow (quadratic-ish blowup
// in practice, since numPairs itself grows as training progresses). The
// hash table makes each pair lookup O(1) on average, so one scan is
// O(corpus length), and finding the max afterward is a single O(numPairs)
// pass over the table -- much cheaper since numPairs is capped by
// MAX_PAIR_TRACK regardless of corpus size.
static int findBestPair(TokenNode* head, int* outA, int* outB)
{
    static PairSlot table[PAIR_HASH_SIZE];
    memset(table, 0, sizeof(table)); // used=0 marks every slot empty
    int numPairs = 0;

    TokenNode* cur = head;
    while (cur != NULL && cur->next != NULL)
    {
        int a = cur->id;
        int b = cur->next->id;

        unsigned int idx = mixHash(a, b, PAIR_HASH_SIZE);
        while (table[idx].used && !(table[idx].a == a && table[idx].b == b))
            idx = (idx + 1) % PAIR_HASH_SIZE;

        if (table[idx].used)
        {
            table[idx].count++;
        }
        else if (numPairs < MAX_PAIR_TRACK)
        {
            table[idx].used = 1;
            table[idx].a = a;
            table[idx].b = b;
            table[idx].count = 1;
            numPairs++;
        }
        // If the MAX_PAIR_TRACK cap is already hit and this is a pair we
        // haven't seen before, it's silently dropped -- same behavior the
        // original linear-scan version had via its `numPairs < MAX_PAIR_TRACK`
        // guard.

        cur = cur->next;
    }

    if (numPairs == 0) return 0;

    int bestCount = 0;
    int bestA = 0, bestB = 0;
    for (int i = 0; i < PAIR_HASH_SIZE; i++)
    {
        if (table[i].used && table[i].count > bestCount)
        {
            bestCount = table[i].count;
            bestA = table[i].a;
            bestB = table[i].b;
        }
    }

    if (bestCount < 2) return 0; // merging a one-off pair isn't useful

    *outA = bestA;
    *outB = bestB;
    return bestCount;
}
 
// Replaces every occurrence of the adjacent pair (a, b) in the list with a
// single node carrying newId.
static void applyMergeToList(TokenNode* head, int a, int b, int newId)
{
    TokenNode* cur = head;
    while (cur != NULL && cur->next != NULL)
    {
        if (cur->id == a && cur->next->id == b)
        {
            TokenNode* nodeB = cur->next;
            cur->id = newId;
            cur->next = nodeB->next;
 
            if (nodeB->next != NULL)
            {
                nodeB->next->prev = cur;
            }
 
            free(nodeB);
        }
 
        cur = cur->next;
    }
}

void initTokenizer(Tokenizer* t)
{
    t->vocabSize = 256;
    t->numMerges = 0;
 
    // Base vocab: every single byte value is its own token
    for (int i = 0; i < 256; i++)
    {
        t->vocabTable[i] = (char*)malloc(2);
        t->vocabTable[i][0] = (char)i;
        t->vocabTable[i][1] = '\0';
    }
}

void trainBPE(Tokenizer* t, const char* corpus, int targetVocabSize)
{
    initTokenizer(t);
 
    int len = strlen(corpus);
    if (len == 0) return;
 
    // Build the initial byte-level linked list for the whole corpus
    TokenNode* head = (TokenNode*)malloc(sizeof(TokenNode));
    head->id = (unsigned char)corpus[0];
    head->prev = NULL;
    head->next = NULL;
 
    TokenNode* current = head;
    for (int i = 1; i < len; i++)
    {
        TokenNode* newNode = (TokenNode*)malloc(sizeof(TokenNode));
        newNode->id = (unsigned char)corpus[i];
        newNode->prev = current;
        newNode->next = NULL;
        current->next = newNode;
        current = newNode;
    }
 
    while (t->vocabSize < targetVocabSize && t->numMerges < MAX_VOCAB_SIZE)
    {
        int a, b;
        int count = findBestPair(head, &a, &b);
        if (count == 0) break; // nothing left worth merging
 
        int newId = 256 + t->numMerges;
 
        t->merges[t->numMerges].tokenA = a;
        t->merges[t->numMerges].tokenB = b;
        t->numMerges++;
 
        // The new token decodes to whatever its two parts decode to, concatenated
        int lenA = strlen(t->vocabTable[a]);
        int lenB = strlen(t->vocabTable[b]);
        t->vocabTable[newId] = (char*)malloc(lenA + lenB + 1);
        memcpy(t->vocabTable[newId], t->vocabTable[a], lenA);
        memcpy(t->vocabTable[newId] + lenA, t->vocabTable[b], lenB);
        t->vocabTable[newId][lenA + lenB] = '\0';
 
        t->vocabSize++;
 
        applyMergeToList(head, a, b, newId);
    }
 
    current = head;
    while (current != NULL)
    {
        TokenNode* temp = current;
        current = current->next;
        free(temp);
    }
}

int getMergeRank(Tokenizer* t, int idA, int idB) 
{
    for (int i = 0; i < t->numMerges; i++) 
    {
        if (t->merges[i].tokenA == idA && t->merges[i].tokenB == idB) 
        {
            return i;            
        }
    }

    return -1;
}

// Fast rank lookup used internally by tokenizeEncode, since numMerges can
// reach into the hundreds or thousands and getMergeRank's linear scan
// through all of them, repeated for every pair on every pass, was the
// other major cost on real-sized corpora (see tokenizeEncode's comment).
// 4x headroom over MAX_VOCAB_SIZE keeps the load factor low even at the
// largest possible merge count.
#define MERGE_RANK_HASH_SIZE (MAX_VOCAB_SIZE * 4)

typedef struct
{
    int a, b, rank;
    int used;
} MergeRankSlot;

static void buildMergeRankTable(Tokenizer* t, MergeRankSlot* table)
{
    memset(table, 0, sizeof(MergeRankSlot) * MERGE_RANK_HASH_SIZE);

    for (int i = 0; i < t->numMerges; i++)
    {
        int a = t->merges[i].tokenA;
        int b = t->merges[i].tokenB;

        unsigned int idx = mixHash(a, b, MERGE_RANK_HASH_SIZE);
        while (table[idx].used)
            idx = (idx + 1) % MERGE_RANK_HASH_SIZE;

        table[idx].used = 1;
        table[idx].a = a;
        table[idx].b = b;
        table[idx].rank = i;
    }
}

static int lookupMergeRank(MergeRankSlot* table, int a, int b)
{
    unsigned int idx = mixHash(a, b, MERGE_RANK_HASH_SIZE);
    while (table[idx].used)
    {
        if (table[idx].a == a && table[idx].b == b) return table[idx].rank;
        idx = (idx + 1) % MERGE_RANK_HASH_SIZE;
    }
    return -1;
}

int tokenizeEncode(Tokenizer* t, const char* text, int* outputTokens)
{
    int len = strlen(text);
    if (len == 0) return 0;
 
    TokenNode* head = (TokenNode*)malloc(sizeof(TokenNode));
    head->id = (unsigned char)text[0];
    head->prev = NULL;
    head->next = NULL;
 
    TokenNode* current = head;
    for (int i = 1; i < len; i++)
    {
        TokenNode* newNode = (TokenNode*)malloc(sizeof(TokenNode));
        newNode->id = (unsigned char)text[i];
        newNode->prev = current;
        newNode->next = NULL;
        current->next = newNode;
        current = newNode;
    }
 
    // Builds a hash lookup for merge ranks once per call, then repeatedly:
    // scans the list to find the lowest-rank pair *type* present anywhere,
    // and merges every occurrence of that exact pair in one pass (reusing
    // applyMergeToList, the same helper trainBPE uses). This produces the
    // same final tokenization as merging occurrences strictly one at a
    // time in global rank order -- since two non-overlapping occurrences
    // of the same pair type don't affect each other's merge -- but needs
    // at most numMerges scans instead of potentially tens of thousands
    // (one per individual merge occurrence in the whole corpus).
    static MergeRankSlot rankTable[MERGE_RANK_HASH_SIZE];
    buildMergeRankTable(t, rankTable);

    while (1)
    {
        int bestRank = INF;
        int bestA = 0, bestB = 0;
        int foundAny = 0;

        current = head;
        while (current != NULL && current->next != NULL)
        {
            int rank = lookupMergeRank(rankTable, current->id, current->next->id);

            if (rank != -1 && rank < bestRank)
            {
                bestRank = rank;
                bestA = current->id;
                bestB = current->next->id;
                foundAny = 1;
            }

            current = current->next;
        }

        if (!foundAny)
            break;

        int mergedId = 256 + bestRank;
        applyMergeToList(head, bestA, bestB, mergedId);
    }
 
    int tokenCount = 0;
    current = head;
 
    while (current != NULL)
    {
        outputTokens[tokenCount++] = current->id;
        TokenNode* temp = current;
        current = current->next;
        free(temp);
    }
 
    return tokenCount;
}
 
int tokenizeDecode(Tokenizer* t, const int* tokens, int tokenCount, char* output, int maxLen)
{
    int pos = 0;
 
    for (int i = 0; i < tokenCount; i++)
    {
        int id = tokens[i];
        if (id < 0 || id >= t->vocabSize || t->vocabTable[id] == NULL) continue;
 
        int pieceLen = strlen(t->vocabTable[id]);
        if (pos + pieceLen >= maxLen) break; // avoid overflowing the output buffer
 
        memcpy(output + pos, t->vocabTable[id], pieceLen);
        pos += pieceLen;
    }
 
    output[pos] = '\0';
    return pos;
}
 
void freeTokenizer(Tokenizer* t)
{
    for (int i = 0; i < t->vocabSize; i++)
    {
        free(t->vocabTable[i]);
        t->vocabTable[i] = NULL;
    }
 
    t->vocabSize = 0;
    t->numMerges = 0;
}