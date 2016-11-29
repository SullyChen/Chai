#pragma once
#ifndef RELU_H
#define RELU_H

#include <layer.h>

class ReLU : public Layer
{
public:
  ReLU();
  ReLU(unsigned int input_size);
  unsigned int InputSize;
  unsigned int OutputSize;
  void Forward();
  void CalcDeltas(std::vector<double> nextLayerDeltas);
  void UpdateParams(double learning_rate);
};

#endif
