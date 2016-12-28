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
  std::vector<float> W;
  std::vector<float> b;
  void Forward();
  void CalcDeltas(std::vector<float> nextLayerDeltas);
  void UpdateParams(float learning_rate);
};

#endif
