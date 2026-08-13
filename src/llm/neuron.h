#pragma once

typedef struct
{
    double* weights;
    int numInputs;
    double bias;
    double learningRate;
} Neuron;
 
void initNeuron(int numInputs, double lr, Neuron* neuron);
double feedForward(const double* inputs, int numInputs, Neuron* neuron);
void trainNeuron(const double* inputs, int numInputs, double target, Neuron* neuron);
double sigmoid(double x);
void freeNeuron(Neuron* neuron);