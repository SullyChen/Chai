#include "relu.h"

ReLU::ReLU()
{
}

ReLU::ReLU(unsigned int input_size)
{
  InputSize = input_size;
}

void ReLU::Forward()
{
  //make sure the batch fits the input size
  if (Input.size() % InputSize != 0)
    std::cout << "Error! Input size mismatch: given " << Input.size() << " instead of multiple of "
              << InputSize << std::endl;
  else
  {
    Output.resize(Input.size());
    for (unsigned int i = 0; i < Input.size(); i++)
      if (Input[i] < 0)
        Output[i] = 0;
      else
        Input[i] = Output[i];
  }
}

void ReLU::CalcDeltas(std::vector<float> nextLayerDeltas)
{
  NextLayerDeltas = nextLayerDeltas;
  LayerDeltas.clear();
  LayerDeltas.resize(nextLayerDeltas.size());
  for (unsigned int i = 0; i < nextLayerDeltas.size(); i++)
    if (Output[i] < 0)
      LayerDeltas[i] = 0;
    else
      LayerDeltas[i] = nextLayerDeltas[i];
}

void ReLU::UpdateParams(float learning_rate)
{
}
