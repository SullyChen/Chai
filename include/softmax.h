#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <layer.h>

class Softmax : public Layer
{
public:
  Softmax();
  Softmax(unsigned int input_size);
  unsigned int InputSize;
  unsigned int OutputSize;
  void Forward();
  void CalcDeltas(std::vector<float> nextLayerDeltas);
  void UpdateParams(float learning_rate);
};

#endif
