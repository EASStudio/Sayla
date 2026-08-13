// Library includes
#include "raylib.h"

// Core includes
#include <math.h>
#include <stdlib.h>

// Module includes
#include "neuron.h"

// Returns a random double in [-1.0, 1.0]
static double randomWeight(void)
{
    return GetRandomValue(-1000, 1000) / 1000.0;
}

void initNeuron(int numInputs, double lr, Neuron* neuron)
{
    neuron->learningRate = lr;
    neuron->numInputs = numInputs;
    neuron->bias = randomWeight();

    neuron->weights = (double*)malloc(sizeof(double) * numInputs);
    for (int i = 0; i < numInputs; ++i)
        neuron->weights[i] = randomWeight();
}

double feedForward(const double* inputs, int numInputs, Neuron* neuron)
{
    double sum = neuron->bias;

    for (int i = 0; i < numInputs; ++i)
        sum += inputs[i] * neuron->weights[i];

    return sigmoid(sum);
}

void trainNeuron(const double* inputs, int numInputs, double target, Neuron* neuron)
{
    double output = feedForward(inputs, numInputs, neuron);

    // Calculate error
    double error = target - output;

    // Derivative of sigmoid function
    double errorGradient = error * output * (1.0 - output);

    for (int i = 0; i < numInputs; ++i)
        neuron->weights[i] += neuron->learningRate * errorGradient * inputs[i];

    neuron->bias += neuron->learningRate * errorGradient;
}

double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

void freeNeuron(Neuron* neuron)
{
    free(neuron->weights);
    neuron->weights = NULL;
}