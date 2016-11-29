#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <time.h>
#include <fstream>
#include <SFML/Graphics.hpp>
#include <chai.h>

unsigned int MaxElement(std::vector<double> input);
std::vector<std::vector<double> > LoadMNIST();

unsigned int MaxElement(std::vector<double> input)
{
  if (input.size() == 0)
    return -1;

  double max = input[0];
  int index = 0;
  for (unsigned int i = 1; i < input.size(); i++)
    if (input[i] > max)
    {
      max = input[i];
      index = i;
    }

  return index;
}

std::vector<std::vector<double> > LoadMNIST()
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

  std::vector<double> input0;
  std::vector<double> input1;
  std::vector<double> input2;
  std::vector<double> input3;
  std::vector<double> input4;
  std::vector<double> input5;
  std::vector<double> input6;
  std::vector<double> input7;
  std::vector<double> input8;
  std::vector<double> input9;

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

  std::vector<std::vector<double> > training_data;
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
    std::vector<std::vector<double> > training_data = LoadMNIST();

    //create the labels
    std::vector<std::vector<double> > labels;
    for (int i = 0; i < 10; i++)
    {
        std::vector<double> label;
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
    const double LEARNING_RATE = 0.1f;
    const unsigned int NUM_EXAMPLES = training_data[0].size() / INPUT_SIZE;

    for (unsigned int epoch = 0; epoch < NUM_EPOCHS; epoch++)
      for (unsigned int i = 0; i < NUM_EXAMPLES; i++)
        for (unsigned int j = 0; j < NUM_CLASSES; j++)
        {
          std::vector<double> input;
          input.reserve(INPUT_SIZE);
          for (int k = i * INPUT_SIZE; k < i * INPUT_SIZE + INPUT_SIZE; k++)
            input.push_back(training_data[j][k]);

          double loss = MNISTModel.Train(input, labels[j], LEARNING_RATE / pow(10, epoch / 10.0f));

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
      std::vector<double> input;
      input.reserve(INPUT_SIZE);
      for (int k = i * INPUT_SIZE; k < i * INPUT_SIZE + INPUT_SIZE; k++)
        input.push_back(training_data[j][k]);

      num_tests++;

      if (MaxElement(MNISTModel.Evaluate(input)) == j)
        num_correct++;
    }
  }

  std::cout << "Accuracy: " << (double)num_correct / num_tests * 100.0f << "%" << std::endl;

  // Create the main window
  sf::RenderWindow window(sf::VideoMode(280, 280), "Handwriting Recognition");
  sf::VertexArray pointmap(sf::Points, 280 * 280);
  for (int i = 0; i < 280; i++)
      for (int j = 0; j < 280; j++)
      {
          pointmap[i * 280 + j].position.x = j;
          pointmap[i * 280 + j].position.y = i;
          pointmap[i * 280 + j].color = sf::Color::Black;
      }

  // Start the game loop
  while (window.isOpen())
  {
      // Process events
      sf::Event event;
      while (window.pollEvent(event))
      {
          // Close window: exit
          if (event.type == sf::Event::Closed) {
              window.close();
          }

          // Escape pressed: exit
          if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Escape) {
              window.close();
          }
      }

      //zoom into area that is left clicked
      if (sf::Mouse::isButtonPressed(sf::Mouse::Left))
      {
          sf::Vector2i position = sf::Mouse::getPosition(window);
          if (position.x + position.y * 280 < 280*280 && position.x + position.y * 280 >= 0)
          {
              for (int i = -16; i < 17; i++)
              {
                  for (int j = -16; j < 17; j++)
                  {
                      if (position.x + i + (position.y + j) * 280 < 280*280 && position.x + i + (position.y + j) * 280 >= 0)
                      {
                          double distance_squared = i * i + j * j + 1;
                          sf::Color color(255 / distance_squared, 255 / distance_squared, 255 / distance_squared);
                          pointmap[position.x + i + (position.y + j) * 280].position.x = position.x + i;
                          pointmap[position.x + i + (position.y + j) * 280].position.y = position.y + j;
                          pointmap[position.x + i + (position.y + j) * 280].color += color;
                          pointmap[position.x + i + (position.y + j) * 280].color += color;
                          pointmap[position.x + i + (position.y + j) * 280].color += color;
                          pointmap[position.x + i + (position.y + j) * 280].color += color;
                          pointmap[position.x + i + (position.y + j) * 280].color += color;
                      }
                  }
              }
          }
      }

      if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::C))
      {
          for (int i = 0; i < 280*280; i++)
              pointmap[i].color = sf::Color::Black;
      }

      if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Return))
      {
          std::vector<double> input;
          input.reserve(784);
          for (int k = 0; k < 28; k++)
          {
              for (int l = 0; l < 28; l++)
              {
                  double average = 0.0f;
                  for (int i = 0; i < 10; i++)
                      for (int j = 0; j < 10; j++)
                      {
                          double temp_average = 0.0f;
                          temp_average += pointmap[k * 10 * 280 + l * 10 + i * 280 + j].color.r;
                          temp_average += pointmap[k * 10 * 280 + l * 10 + i * 280 + j].color.g;
                          temp_average += pointmap[k * 10 * 280 + l * 10 + i * 280 + j].color.b;
                          temp_average /= 3.0f;
                          average += temp_average;
                      }
                  average /= 100.0f;
                  average /= 255.0f; //normalize
                  input.push_back(average);
              }
          }
          std::vector<double> output = MNISTModel.Evaluate(input);
          int prediction = MaxElement(output);
          std::cout << "This number is predicted to be a: " << prediction << std::endl;
      }

      // Clear screen
      window.clear();
      window.draw(pointmap);
      // Update the window
      window.display();
  }

  return EXIT_SUCCESS;
}
