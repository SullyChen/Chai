#ifndef LAYER_H
#define LAYER_H

#include <iostream>
#include <math.h>
#include <vector>

class Layer
{
public:
  Layer();
  virtual ~Layer();
  std::vector<float> Input;
  std::vector<float> Output;
  std::vector<float> LayerDeltas;
  std::vector<float> NextLayerDeltas;
  virtual void Forward() = 0;
  virtual void CalcDeltas(std::vector<float> nextLayerDeltas) = 0;
  virtual void UpdateParams(float learning_rate) = 0;
};

#endif
