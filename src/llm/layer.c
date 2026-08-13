// Module includes
#include "layer.h"

// Core includes
#include <stdlib.h>
#include <math.h>

void initLayer(Layer* layer, int numNeurons, int numInputs, double lr) 
{
    layer->numNeurons = numNeurons;
    layer->numInputs = numInputs;
    layer->isSoftmax = false; // hidden layers stay sigmoid; initNetwork flips the output layer

    layer->neurons = (Neuron*)malloc(sizeof(Neuron) * numNeurons);
    layer->outputs = (double*)malloc(sizeof(double) * numNeurons);
    layer->deltas = (double*)malloc(sizeof(double) * numNeurons);

    for (int i = 0; i < numNeurons; i++) 
        initNeuron(numInputs, lr, &layer->neurons[i]);
}

double* feedForwardLayer(Layer* layer, const double* inputs) 
{
    if (layer->isSoftmax)
    {
        // Raw logits first (no activation), stored directly into outputs
        // as scratch space -- normalized into real probabilities below.
        double maxRaw = -1.0e300;
        for (int j = 0; j < layer->numNeurons; j++)
        {
            Neuron* n = &layer->neurons[j];
            double sum = n->bias;
            for (int i = 0; i < layer->numInputs; i++)
                sum += inputs[i] * n->weights[i];

            layer->outputs[j] = sum;
            if (sum > maxRaw) maxRaw = sum;
        }

        // Subtracting the max before exponentiating keeps this numerically
        // stable without changing the result (softmax is shift-invariant).
        double expSum = 0.0;
        for (int j = 0; j < layer->numNeurons; j++)
        {
            double e = exp(layer->outputs[j] - maxRaw);
            layer->outputs[j] = e;
            expSum += e;
        }

        for (int j = 0; j < layer->numNeurons; j++)
            layer->outputs[j] /= expSum;

        return layer->outputs;
    }

    for (int i = 0; i < layer->numNeurons; i++)
        layer->outputs[i] = feedForward(inputs, layer->numInputs, &layer->neurons[i]);

    return layer->outputs;
}

void freeLayer(Layer* layer) 
{
    for (int i = 0; i < layer->numNeurons; i++)
        freeNeuron(&layer->neurons[i]);

    free(layer->neurons);
    free(layer->outputs);
    free(layer->deltas);
}