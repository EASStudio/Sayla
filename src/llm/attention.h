#pragma once

// Single-head causal self-attention over a sequence of embeddings.
//
// Forward: given X (seqLen x embedDim), projects to Q/K/V (seqLen x headDim),
// computes causal-masked softmax(QK^T / sqrt(headDim)) as attention weights,
// and returns out = alpha * V (seqLen x headDim). "Causal" means position i
// only attends to positions 0..i -- it can't see future tokens, same
// constraint a real language model needs at generation time.
//
// Backward: standard backprop through softmax attention, accumulating
// gradients into Wq/Wk/Wv and writing dX (gradient w.r.t. the input
// embeddings) so a caller can chain this into an embedding table the way
// engine.c already does for the existing MLP.

typedef struct
{
    int maxSeqLen;
    int embedDim;   // input embedding dimension (d_model)
    int headDim;    // Q/K/V projection dimension (d_k)

    double* Wq;     // embedDim x headDim
    double* Wk;     // embedDim x headDim
    double* Wv;     // embedDim x headDim

    // Forward-pass caches, sized for maxSeqLen and overwritten each call.
    // Kept around because backward() needs them.
    double* Q;      // maxSeqLen x headDim
    double* K;      // maxSeqLen x headDim
    double* V;      // maxSeqLen x headDim
    double* alpha;  // maxSeqLen x maxSeqLen, row i valid for columns 0..i
    double* out;    // maxSeqLen x headDim -- this layer's output

    int lastSeqLen; // sequence length used in the most recent forward() call
} AttentionLayer;

void initAttentionLayer(AttentionLayer* attn, int maxSeqLen, int embedDim, int headDim);
double* attentionForward(AttentionLayer* attn, const double* X, int seqLen);
void attentionBackward(AttentionLayer* attn, const double* X, const double* dOut, int seqLen, double* dX, double lr);
void freeAttentionLayer(AttentionLayer* attn);