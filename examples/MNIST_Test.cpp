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

std::vector<std::vector<float>> LoadMNIST() {
  std::vector<std::vector<float>> training_data;

  // The MNIST dataset file list
  const std::vector< std::string > fileList = {
      "MNIST_Dataset/data0.txt", "MNIST_Dataset/data1.txt",
      "MNIST_Dataset/data2.txt", "MNIST_Dataset/data3.txt",
      "MNIST_Dataset/data4.txt", "MNIST_Dataset/data5.txt",
      "MNIST_Dataset/data6.txt", "MNIST_Dataset/data7.txt",
      "MNIST_Dataset/data8.txt", "MNIST_Dataset/data9.txt"};

  for (auto currentFile : fileList) {
    std::vector<float> theData;

    // open the data file
    std::ifstream dataFile(currentFile.c_str(), std::ios::binary);

    // Try to load the data if the file was able to be opened
    if (dataFile.good()) {

      // Load and scale the data in the file
      char dataByte;
      while (dataFile.get(dataByte)) {
        theData.push_back(dataByte / 255.0f);
      }

      // Save the data to the training set
      training_data.push_back(theData);

    } else {

      std::cout << "Could not open file " << currentFile << std::endl;

    }
  }

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
