#pragma once
#ifndef SIGMOID_H
#define SIGMOID_H

#include <layer.h>

class Sigmoid : public Layer
{
public:
  Sigmoid();
  Sigmoid(unsigned int input_size);
  unsigned int InputSize;
  unsigned int OutputSize;
  void Forward();
  void CalcDeltas(std::vector<double> nextLayerDeltas);
  void UpdateParams(double learning_rate);
};

#endif
