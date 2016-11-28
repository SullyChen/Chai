#pragma once
#ifndef FCL_H
#define FCL_H

#include <layer.h>

class FCL : public Layer
{
public:
  FCL();
  FCL(unsigned int input_size, unsigned int output_size);
  unsigned int InputSize;
  unsigned int OutputSize;
  std::vector<double> W;
  std::vector<double> b;
  void Forward();
  void CalcDeltas(std::vector<double> nextLayerDeltas);
  void UpdateParams(double learning_rate);
};

#endif
