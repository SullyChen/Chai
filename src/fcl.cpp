#include "fcl.h"

FCL::FCL()
{
}

FCL::FCL(unsigned int input_size, unsigned int output_size)
{
  InputSize = input_size;
  OutputSize = output_size;
  W.resize(InputSize * OutputSize);
  for (unsigned int i = 0; i < InputSize * OutputSize; i++)
    W[i] = (rand() % 2000 - 1000) / 100000.0f;
  b.resize(OutputSize);
  for (unsigned int i = 0; i < OutputSize; i++)
    b[i] = (rand() % 2000 - 1000) / 100000.0f;
}

void FCL::Forward()
{
  //make sure the batch fits the input
  if (Input.size() % InputSize != 0)
    std::cout << "Error! Input size mismatch: given " << Input.size() << " instead of multiple of "
              << InputSize << std::endl;
  else
  {
    //resize the output to the batch size
    Output.clear();
    Output.resize(Input.size() / InputSize * OutputSize);

    //optimization
    unsigned int _iSize = Input.size();
    unsigned int _oSize = Output.size();

    //Weight calculation
    for (unsigned int i = 0; i < _iSize; i++)
      for (unsigned int j = 0; j < OutputSize; j++)
        Output[j + (i / InputSize) * OutputSize] += Input[i] * W[j + (i  % InputSize) * OutputSize];

    //Bias calculation
    for (unsigned int i = 0; i < _oSize; i++)
      Output[i] += b[i % OutputSize];
  }
}

void FCL::CalcDeltas(std::vector<float> nextLayerDeltas)
{
  NextLayerDeltas = nextLayerDeltas;
  LayerDeltas.clear();
  LayerDeltas.resize(nextLayerDeltas.size() / OutputSize * InputSize);

  //optimization
  unsigned int _iSize = Input.size();

  //Weight calculation
  for (unsigned int i = 0; i < _iSize; i++)
    for (unsigned int j = 0; j < OutputSize; j++)
      LayerDeltas[i] += NextLayerDeltas[j + (i / InputSize) * OutputSize] * W[j + (i % InputSize) * OutputSize];
}

void FCL::UpdateParams(float learning_rate)
{
  //optimization
  unsigned int _iSize = Input.size();
  unsigned int _oSize = Output.size();
  unsigned int batch_size = _iSize / InputSize;

  //Weight update
  for (unsigned int i = 0; i < _iSize; i++)
    for (unsigned int j = 0; j < OutputSize; j++)
      W[j + (i % InputSize) * OutputSize] += -1.0f / batch_size * learning_rate * Input[i]
                                              * NextLayerDeltas[j + (i / InputSize) * OutputSize];

  //Bias update
  for (unsigned int i = 0; i < _oSize; i++)
    b[i % OutputSize] += -1.0f / batch_size * NextLayerDeltas[i] * learning_rate;
}
