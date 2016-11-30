#include "softmax.h"

Softmax::Softmax()
{
}

Softmax::Softmax(unsigned int input_size)
{
  InputSize = input_size;
}

void Softmax::Forward()
{
  //make sure the batch fits the input size
  if (Input.size() % InputSize != 0)
    std::cout << "Error! Input size mismatch: given " << Input.size() << " instead of multiple of "
              << InputSize << std::endl;
  else
  {
    Output.resize(Input.size());
    for (unsigned int i = 0; i < Input.size() / InputSize; i++)
    {
      float denominator = 0.0f;
      for (unsigned int j = 0; j < InputSize; j++)
        denominator += exp(Input[j + i * InputSize]);
      for (unsigned int j = 0; j < InputSize; j++)
        Output[j + i * InputSize] = exp(Input[j + i * InputSize]) / denominator;
    }
  }
}

void Softmax::CalcDeltas(std::vector<float> nextLayerDeltas)
{
  NextLayerDeltas = nextLayerDeltas;
  LayerDeltas.clear();
  LayerDeltas.resize(nextLayerDeltas.size());
  for (unsigned int i = 0; i < nextLayerDeltas.size(); i++)
    LayerDeltas[i] = nextLayerDeltas[i] * Output[i] * (1.0f - Output[i]);
  for (unsigned int i = 0; i < Input.size() / InputSize; i++)
  {
    for (unsigned int j = 0; j < InputSize; j++)
      for (unsigned int k = 0; k < InputSize; k++)
      {
        if (j != k)
          LayerDeltas[j + i * InputSize] += -1.0f * nextLayerDeltas[j + i * InputSize] * Output[j] * Output[k];
        else
          LayerDeltas[j + i * InputSize] += nextLayerDeltas[j + i * InputSize] * Output[j] * (1.0f - Output[k]);
      }
  }
}

void Softmax::UpdateParams(float learning_rate)
{
}
