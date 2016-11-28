#pragma once
#ifndef LAYER_H
#define LAYER_H

#include <iostream>
#include <math.h>
#include <vector>

class Layer
{
public:
  Layer();
  std::vector<double> Input;
  std::vector<double> Output;
  std::vector<double> LayerDeltas;
  std::vector<double> NextLayerDeltas;
  virtual void Forward() = 0;
  virtual void CalcDeltas(std::vector<double> nextLayerDeltas) = 0;
  virtual void UpdateParams(double learning_rate) = 0;
};

#endif
