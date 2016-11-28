#include <iostream>
#include <time.h>
#include <chai.h>

using namespace std;

int main()
{
  //srand(time(NULL));
  ChaiModel m;
  m.AddFCL(2, 3);
  m.AddSigmoid(3);
  m.AddFCL(3, 2);
  m.AddSigmoid(2);

  //generate training data
  vector<double> inputs;
  vector<double> outputs;
  for (int i = 0; i < 100000; i++)
    inputs.push_back((rand() % 2000 - 1000) / 100.0f);
  for (int i = 0; i < 99999; i += 2)
  {
    if (inputs[i] > inputs[i + 1])
    {
      outputs.push_back(1.0f);
      outputs.push_back(0.0f);
    }
    else
    {
      outputs.push_back(0.0f);
      outputs.push_back(1.0f);
    }
  }

  unsigned int batch_size = 100;

  for (int i = 0; i < inputs.size() / batch_size / 2; i++)
  {
    vector<double>::const_iterator first = inputs.begin() + i * batch_size * 2;
    vector<double>::const_iterator last = inputs.begin() + (i * batch_size + batch_size) * 2;
    vector<double> input(first, last);

    first = outputs.begin() + i * batch_size * 2;
    last = outputs.begin() + (i * batch_size + batch_size) * 2;
    vector<double> output(first, last);

    m.Train(input, output, 0.1f);
  }

  std::vector<double> test_input;
  test_input.push_back(10.0f);
  test_input.push_back(-10.0f);
  test_input.push_back(-10.0f);
  test_input.push_back(10.0f);

  std::vector<double> test_output = m.Evaluate(test_input);

  for (unsigned int i = 0; i < test_output.size(); i++)
    cout << test_output[i] << endl;

  return 0;
}
