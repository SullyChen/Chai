#pragma once
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
  std::vector<double> Evaluate(std::vector<double> input);
  double Train(std::vector<double> input, std::vector<double> output, double learning_rate);
  void AddFCL(unsigned int input_size, unsigned int output_size);
  void AddSigmoid(unsigned int input_size);
  void AddSoftmax(unsigned int input_size);
  void AddReLU(unsigned int input_size);
  std::vector<Layer*> Layers;
};

#endif
