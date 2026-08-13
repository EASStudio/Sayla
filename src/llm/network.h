#pragma once

// Module includes
#include "layer.h"

typedef struct
{
    Layer* layers;
    int numLayers;
}Network;

void initNetwork(Network* net, const int* layerSizes, int numLayers, double lr);
double* feedForwardNetwork(Network* net, const double* inputs);
void trainNetwork(Network* net, const double* inputs, const double* targets);
void freeNetwork(Network* net);