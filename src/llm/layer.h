#pragma once

// Module include
#include "neuron.h"

// Core includes
#include <stdbool.h>

typedef struct 
{
    Neuron *neurons;
    int numNeurons;
    int numInputs;
    double* outputs;
    double* deltas;
    bool isSoftmax; // true only for the network's final (output) layer
}Layer;

void initLayer(Layer* layer, int numNeurons, int numInputs, double lr);
double* feedForwardLayer(Layer* layer, const double* inputs);
void freeLayer(Layer* layer);