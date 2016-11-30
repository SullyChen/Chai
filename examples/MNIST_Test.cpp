#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <time.h>
#include <fstream>
#include "chai.h"

unsigned int MaxElement(std::vector<float> input);
std::vector<std::vector<float> > LoadMNIST();

unsigned int MaxElement(std::vector<float> input)
{
  if (input.size() == 0)
    return -1;

  float max = input[0];
  int index = 0;
  for (unsigned int i = 1; i < input.size(); i++)
    if (input[i] > max)
    {
      max = input[i];
      index = i;
    }

  return index;
}

std::vector<std::vector<float> > LoadMNIST()
{
  //open the MNIST dataset
  std::ifstream data0;
  data0.open("MNIST_Dataset/data0.txt", std::ios::binary);
  std::ifstream data1;
  data1.open("MNIST_Dataset/data1.txt", std::ios::binary);
  std::ifstream data2;
  data2.open("MNIST_dataset/data2.txt", std::ios::binary);
  std::ifstream data3;
  data3.open("MNIST_Dataset/data3.txt", std::ios::binary);
  std::ifstream data4;
  data4.open("MNIST_Dataset/data4.txt", std::ios::binary);
  std::ifstream data5;
  data5.open("MNIST_Dataset/data5.txt", std::ios::binary);
  std::ifstream data6;
  data6.open("MNIST_Dataset/data6.txt", std::ios::binary);
  std::ifstream data7;
  data7.open("MNIST_Dataset/data7.txt", std::ios::binary);
  std::ifstream data8;
  data8.open("MNIST_Dataset/data8.txt", std::ios::binary);
  std::ifstream data9;
  data9.open("MNIST_Dataset/data9.txt", std::ios::binary);

  std::vector<float> input0;
  std::vector<float> input1;
  std::vector<float> input2;
  std::vector<float> input3;
  std::vector<float> input4;
  std::vector<float> input5;
  std::vector<float> input6;
  std::vector<float> input7;
  std::vector<float> input8;
  std::vector<float> input9;

  //create the input vectors
  while (data0.good())
  {
    int data_byte = data0.get();
    input0.push_back(data_byte / 255.0f);
  }
  while (data1.good())
  {
    int data_byte = data1.get();
    input1.push_back(data_byte / 255.0f);
  }
  while (data2.good())
  {
    int data_byte = data2.get();
    input2.push_back(data_byte / 255.0f);
  }
  while (data3.good())
  {
    int data_byte = data3.get();
    input3.push_back(data_byte / 255.0f);
  }
  while (data4.good())
  {
    int data_byte = data4.get();
    input4.push_back(data_byte / 255.0f);
  }
  while (data5.good())
  {
    int data_byte = data5.get();
    input5.push_back(data_byte / 255.0f);
  }
  while (data6.good())
  {
    int data_byte = data6.get();
    input6.push_back(data_byte / 255.0f);
  }
  while (data7.good())
  {
    int data_byte = data7.get();
    input7.push_back(data_byte / 255.0f);
  }
  while (data8.good())
  {
    int data_byte = data8.get();
    input8.push_back(data_byte / 255.0f);
  }
  while (data9.good())
  {
    int data_byte = data9.get();
    input9.push_back(data_byte / 255.0f);
  }

  data0.close();
  data1.close();
  data2.close();
  data3.close();
  data4.close();
  data5.close();
  data6.close();
  data7.close();
  data8.close();
  data9.close();

  std::vector<std::vector<float> > training_data;
  training_data.push_back(input0);
  training_data.push_back(input1);
  training_data.push_back(input2);
  training_data.push_back(input3);
  training_data.push_back(input4);
  training_data.push_back(input5);
  training_data.push_back(input6);
  training_data.push_back(input7);
  training_data.push_back(input8);
  training_data.push_back(input9);

  return training_data;
}

int main()
{
    srand(time(NULL));
    std::vector<std::vector<float> > training_data = LoadMNIST();

    //create the labels
    std::vector<std::vector<float> > labels;
    for (int i = 0; i < 10; i++)
    {
        std::vector<float> label;
        label.resize(10);
        label[i] = 1.0f;
        labels.push_back(label);
    }

    ChaiModel MNISTModel;
    MNISTModel.AddFCL(784, 10);
    MNISTModel.AddSoftmax(10);
    const unsigned int INPUT_SIZE = 784;
    const unsigned int NUM_CLASSES = 10;
    const unsigned int OUTPUT_SIZE = 10;
    const unsigned int NUM_EPOCHS = 10;
    const float LEARNING_RATE = 0.1f;
    const unsigned int NUM_EXAMPLES = training_data[0].size() / INPUT_SIZE;

    for (unsigned int epoch = 0; epoch < NUM_EPOCHS; epoch++)
      for (unsigned int i = 0; i < NUM_EXAMPLES; i++)
        for (unsigned int j = 0; j < NUM_CLASSES; j++)
        {
          std::vector<float> input;
          input.reserve(INPUT_SIZE);
          for (int k = i * INPUT_SIZE; k < i * INPUT_SIZE + INPUT_SIZE; k++)
            input.push_back(training_data[j][k]);

          float loss = MNISTModel.Train(input, labels[j], LEARNING_RATE / pow(10, epoch / 10.0f));

          //train the model
          if (i % 100 == 0)
          std::cout << "Epoch " << epoch + 1 << ", loss: " << loss << std::endl;
        }

  std::cout << "Testing trained model... " << std::endl;

  unsigned int num_tests = 0;
  unsigned int num_correct = 0;
  for (unsigned int i = 0; i < NUM_EXAMPLES; i++)
  {
    for (unsigned int j = 0; j < NUM_CLASSES; j++)
    {
      std::vector<float> input;
      input.reserve(INPUT_SIZE);
      for (int k = i * INPUT_SIZE; k < i * INPUT_SIZE + INPUT_SIZE; k++)
        input.push_back(training_data[j][k]);

      num_tests++;

      if (MaxElement(MNISTModel.Evaluate(input)) == j)
        num_correct++;
    }
  }

  std::cout << "Accuracy: " << (float)num_correct / num_tests * 100.0f << "%" << std::endl;

	return 0;
}
