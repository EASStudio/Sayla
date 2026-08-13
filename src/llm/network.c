// Module includes
#include "network.h"

// Core includes
#include <stdlib.h>

void initNetwork(Network* net, const int* layerSizes, int numLayers, double lr) 
{
    net->numLayers = numLayers;
    net->layers = (Layer*)malloc(sizeof(Layer) * numLayers);

    for (int i = 0; i < numLayers; i++) 
    {
        int inputsForThisLayer = layerSizes[i];
        int neuronsInThisLayer = layerSizes[i + 1];
        initLayer(&net->layers[i], neuronsInThisLayer, inputsForThisLayer, lr);
    }

    // The output layer predicts a probability distribution over the
    // vocabulary; hidden layers keep the sigmoid activation they've
    // always used.
    net->layers[numLayers - 1].isSoftmax = true;
}

double* feedForwardNetwork(Network* net, const double* inputs) 
{
    const double* current = inputs;

    for (int i = 0; i < net->numLayers; i++) 
        current = feedForwardLayer(&net->layers[i], current);

    return net->layers[net->numLayers - 1].outputs;
}

void trainNetwork(Network* net, const double* inputs, const double* targets) 
{
    feedForwardNetwork(net, inputs);

    Layer* outputLayer = &net->layers[net->numLayers - 1];

    if (outputLayer->isSoftmax)
    {
        // Softmax + cross-entropy: the softmax Jacobian and the
        // cross-entropy derivative cancel out algebraically, so the
        // gradient w.r.t. each output logit is simply (output - target).
        // To match this file's sign convention -- delta is defined as the
        // negative of that gradient, since weights are updated by adding
        // lr * delta * input -- that's (target - output), with no extra
        // derivative term needed (unlike the sigmoid+MSE case below).
        for (int j = 0; j < outputLayer->numNeurons; j++)
            outputLayer->deltas[j] = targets[j] - outputLayer->outputs[j];
    }
    else
    {
        for (int j = 0; j < outputLayer->numNeurons; j++) 
        {
            double output = outputLayer->outputs[j];
            double error = targets[j] - output;
            outputLayer->deltas[j] = error * output * (1.0 - output);
        }
    }

    for (int l = net->numLayers - 2; l >= 0; l--)
    {
        Layer* layer = &net->layers[l];
        Layer* nextLayer = &net->layers[l + 1];
 
        for (int j = 0; j < layer->numNeurons; j++)
        {
            double sum = 0.0;
            for (int k = 0; k < nextLayer->numNeurons; k++)
                sum += nextLayer->deltas[k] * nextLayer->neurons[k].weights[j];
 
            double output = layer->outputs[j];
            layer->deltas[j] = sum * output * (1.0 - output);
        }
    }

    for (int l = 0; l < net->numLayers; l++)
    {
        Layer* layer = &net->layers[l];
        const double* layerInputs = (l == 0) ? inputs : net->layers[l - 1].outputs;
 
        for (int j = 0; j < layer->numNeurons; j++)
        {
            Neuron* n = &layer->neurons[j];
 
            for (int i = 0; i < layer->numInputs; i++)
                n->weights[i] += n->learningRate * layer->deltas[j] * layerInputs[i];
 
            n->bias += n->learningRate * layer->deltas[j];
        }
    }

}

void freeNetwork(Network* net) 
{
    for (int i = 0; i < net->numLayers; i++) 
        freeLayer(&net->layers[i]);

    free(net->layers);
}