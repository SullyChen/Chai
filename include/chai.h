#ifndef CHAI_H
#define CHAI_H

#include <fcl.h>
#include <sigmoid.h>
#include <softmax.h>
#include <relu.h>

class ChaiModel
{
public:
  ChaiModel();
  ~ChaiModel();
  std::vector<float> Evaluate(std::vector<float> input);
  float Train(std::vector<float> input, std::vector<float> output, float learning_rate);
  void AddFCL(unsigned int input_size, unsigned int output_size);
  void AddSigmoid(unsigned int input_size);
  void AddSoftmax(unsigned int input_size);
  void AddReLU(unsigned int input_size);
  std::vector<Layer*> Layers;
};

#endif
